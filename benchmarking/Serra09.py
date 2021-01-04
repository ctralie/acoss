# -*- coding: utf-8 -*-
from pySeqAlign import qmax, dmax
from CoverAlgorithm import *
from CRPUtils import *
import numpy as np
import matplotlib.pyplot as plt
import argparse
import librosa
from skimage.transform import resize

COMMON_SIZE = -1
RES = 128
scattering = None
DO_SCATTERING = True
if DO_SCATTERING:
    from kymatio.numpy import Scattering2D
    scattering = Scattering2D(shape=(RES, RES), J=4, L=8)

def global_chroma(chroma):
    """Computes global chroma of a input chroma vector"""
    if chroma.shape[1] not in [12, 24, 36]:
        raise IOError("Wrong axis for the input chroma array. Expected shape '(frame_size, bin_size)'")
    return np.divide(chroma.sum(axis=0), np.max(chroma.sum(axis=0)))

def get_ssm_sequence(mfcc, downsample_fac, m):
    """
    Get a sequence of SSMs of mfccs
    Parameters
    ----------
    mfcc: ndarray(12, N)
        MFCC features
    downsample_fac: int
        The factor by which to downsample the MFCCs, treated both
        as twice the averaging parameter and as the hop length between blocks
    m: int
        Number of delays to take.  Total block length should encompass 
        2*m*downsample_fac audio
    """
    if COMMON_SIZE > -1:
        mfcc = resize(mfcc, (mfcc.shape[0], COMMON_SIZE*downsample_fac), anti_aliasing=True)
    mfcc = mfcc.T
    ssms = []
    idx = 0
    win = int(downsample_fac/2)
    while idx + m*downsample_fac <= mfcc.shape[0]:
        x = mfcc[idx:idx+m*downsample_fac, :]
        # Smooth out mfcc
        x = np.cumsum(x, axis=0)
        x = x[win::, :] - x[0:-win, :]
        # Z-normalize block
        x -= np.mean(x, 0)[None, :]
        norm = np.sqrt(np.sum(x**2, 1))
        norm[norm == 0] = 1
        x /= norm[:, None]
        xsqr = np.sum(x**2, 1)
        # Compute resized SSM
        D = xsqr[:, None] + xsqr[None, :] - 2*x.dot(x.T)
        D[D < 0] = 0
        D = np.sqrt(D)
        D = resize(D, (RES, RES), anti_aliasing=True)
        if DO_SCATTERING:
            D = scattering(D)
        idx += downsample_fac
        ssms.append(D.flatten())
    return np.array(ssms)



class Serra09(CoverAlgorithm):
    """
    Attributes
    ----------
    Same as CoverAlgorithms, plus
    chroma_type: string
        Type of chroma to use (key into features)
    downsample_fac: int
        The factor by which to downsample the HPCPs with
        median aggregation
    all_feats: {int: dictionary}
        Cached features
    """
    def __init__(self, datapath="../features_covers80", chroma_type='hpcp', shortname='benchmark', 
                oti=True, kappa=0.095, m=9, downsample_fac=40, do_memmaps=True):
        self.oti = oti
        self.m = m
        self.chroma_type = chroma_type
        self.kappa = kappa
        self.m = m
        self.downsample_fac = downsample_fac
        self.all_feats = {} # For caching features (global chroma and stacked chroma)
        CoverAlgorithm.__init__(self, "Serra09", datapath=datapath, shortname=shortname, do_memmaps=do_memmaps, similarity_types=["ssms_scatter_qmax", "ssms_scatter_dmax", "chroma_qmax", "chroma_dmax", "mfcc_qmax", "mfcc_dmax"])

    def load_features(self, i):
        if not i in self.all_feats:
            feats = CoverAlgorithm.load_features(self, i)
            ## Step 1: Compute aggregated chroma
            # First compute global chroma (used for OTI later)
            chroma = feats[self.chroma_type]
            gchroma = global_chroma(chroma)
            # Now downsample the chromas using median aggregation
            chroma = librosa.util.sync(chroma.T, np.arange(0, chroma.shape[0], self.downsample_fac), aggregate=np.median)
            
            ## Step 2: Compute aggregated mfcc features
            mfcc_orig = feats['mfcc_htk']
            mfcc_orig[np.isnan(mfcc_orig)] = 0
            mfcc_orig[np.isinf(mfcc_orig)] = 0
            mfcc = librosa.util.sync(mfcc_orig, np.arange(0, mfcc_orig.shape[1], self.downsample_fac), aggregate=np.mean)
            N = min(chroma.shape[1], mfcc.shape[1])
            chroma = chroma[:, 0:N]
            mfcc = mfcc[:, 0:N]
            ## Step 3: Compute MFCC SSMs
            ssms = get_ssm_sequence(mfcc_orig[0:N*self.downsample_fac], self.downsample_fac, self.m)
            

            ## Step 4: Do a uniform scaling
            if COMMON_SIZE > -1:
                chroma = resize(chroma, (chroma.shape[0], COMMON_SIZE), anti_aliasing=True)
                mfcc = resize(mfcc, (mfcc.shape[0], COMMON_SIZE), anti_aliasing=True)

            ## Step 5: Do a stacked delay embedding of mfcc and save away features
            mfcc_stacked = sliding_window(mfcc.T, self.m)
            if ssms.shape[0] < mfcc_stacked.shape[0]:
                ssms2 = np.zeros((mfcc_stacked.shape[0], ssms.shape[1]))
                ssms2[0:ssms.shape[0], :] = ssms
                ssms2[ssms.shape[0]::, :] = ssms[-1, :]
                ssms = ssms2
            ssms = ssms[0:mfcc_stacked.shape[0], :]

            feats = {'gchroma':gchroma, 'chroma':chroma, 'mfcc_stacked':mfcc_stacked, 'ssms':ssms}
            self.all_feats[i] = feats
        return self.all_feats[i]

    def similarity(self, idxs):
        N = idxs.shape[0]
        similarities = {'ssms_scatter_qmax':np.zeros(N), 'ssms_scatter_dmax':np.zeros(N), 'chroma_qmax':np.zeros(N), 'chroma_dmax':np.zeros(N), 'mfcc_qmax':np.zeros(N), 'mfcc_dmax':np.zeros(N)}
        for idx, (i,j) in enumerate(zip(idxs[:, 0], idxs[:, 1])):
            Si = self.load_features(i)
            Sj = self.load_features(j)

            ## Step 1: Do chroma similarities
            oti = get_oti(Si['gchroma'], Sj['gchroma'])
            C1 = np.roll(Si['chroma'], oti, axis=0)
            C2 = Sj['chroma']
            csm = get_csm_cosine(C1.T, C2.T)
            csm = sliding_csm(csm, self.m)
            csm = csm_to_binary(csm, self.kappa)
            M, N = csm.shape[0], csm.shape[1]
            D = np.zeros(M*N, dtype=np.float32)
            similarities['chroma_qmax'][idx] = qmax(csm.flatten(), D, M, N) / (M+N)
            similarities['chroma_dmax'][idx] = dmax(csm.flatten(), D, M, N) / (M+N)

            ## Step 2: Do MFCC similarities
            csm = get_csm(Si['mfcc_stacked'], Sj['mfcc_stacked'])
            csm = csm_to_binary(csm, self.kappa)
            M, N = csm.shape[0], csm.shape[1]
            D = np.zeros(M*N, dtype=np.float32)
            similarities['mfcc_qmax'][idx] = qmax(csm.flatten(), D, M, N) / (M+N)
            similarities['mfcc_dmax'][idx] = dmax(csm.flatten(), D, M, N) / (M+N)

            ## Step 3: Do SSM Similarities
            csm = get_csm(Si['ssms'], Sj['ssms'])
            csm = csm_to_binary(csm, self.kappa)
            M, N = csm.shape[0], csm.shape[1]
            D = np.zeros(M*N, dtype=np.float32)
            similarities['ssms_scatter_qmax'][idx] = qmax(csm.flatten(), D, M, N) / (M+N)
            similarities['ssms_scatter_dmax'][idx] = dmax(csm.flatten(), D, M, N) / (M+N)
            if self.do_memmaps:
                for key in self.Ds.keys():
                    self.Ds[key][i][j] = similarities[key][idx]
        return similarities

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmarking with Joan Serra's Cover id algorithm",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", '--datapath', type=str, action="store", default='../features_covers80',
                        help="Path to data files")
    parser.add_argument("-s", "--shortname", type=str, action="store", default="covers80", help="Short name for dataset")
    parser.add_argument("-c", '--chroma_type', type=str, action="store", default='hpcp',
                        help="Type of chroma to use for experiments")
    parser.add_argument("-p", '--parallel', type=int, choices=(0, 1), action="store", default=0,
                        help="Parallel computing or not")
    parser.add_argument("-n", '--n_cores', type=int, action="store", default=1,
                        help="No of cores required for parallelization")
    parser.add_argument("-r", "--range", type=str, action="store", default="")
    parser.add_argument("-b", "--batch_path", type=str, action="store", default="")

    cmd_args = parser.parse_args()
    
    do_memmaps = True
    if (len(cmd_args.range) > 0):
        do_memmaps = False
    serra09 = Serra09(cmd_args.datapath, cmd_args.chroma_type, cmd_args.shortname, do_memmaps=do_memmaps)
    
    if len(cmd_args.batch_path) > 0:
        # Aggregrate precomputed similarities
        serra09.load_batches(cmd_args.batch_path)
        for similarity_type in serra09.Ds.keys():
            serra09.getEvalStatistics(similarity_type)
    else:
        if do_memmaps:
            # Do the whole thing
            serra09.all_pairwise(cmd_args.parallel, cmd_args.n_cores, symmetric=True)
            for similarity_type in serra09.Ds.keys():
                print(similarity_type)
                serra09.getEvalStatistics(similarity_type)
            serra09.cleanup_memmap()
        else:
            # Do only a range and save it
            [w, idx] = [int(s) for s in cmd_args.range.split("-")]
            serra09.do_batch(w, idx, "cache/serra")
    
    print("... Done ....")

