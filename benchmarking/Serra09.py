# -*- coding: utf-8 -*-
from pySeqAlign import qmax, dmax
from CoverAlgorithm import *
from CRPUtils import *
import numpy as np
import argparse
import librosa


def global_chroma(chroma):
    """Computes global chroma of a input chroma vector"""
    if chroma.shape[1] not in [12, 24, 36]:
        raise IOError("Wrong axis for the input chroma array. Expected shape '(frame_size, bin_size)'")
    return np.divide(chroma.sum(axis=0), np.max(chroma.sum(axis=0)))

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
                oti=True, kappa=0.095, tau=1, m=9, downsample_fac=40, do_memmaps=True):
        self.oti = oti
        self.tau = tau
        self.m = m
        self.chroma_type = chroma_type
        self.kappa = kappa
        self.tau = tau
        self.m = m
        self.downsample_fac = downsample_fac
        self.all_feats = {} # For caching features (global chroma and stacked chroma)
        CoverAlgorithm.__init__(self, "Serra09", datapath=datapath, shortname=shortname, do_memmaps=do_memmaps, similarity_types=["chroma_qmax", "chroma_dmax", "mfcc_qmax", "mfcc_dmax"])

    def load_features(self, i):
        if not i in self.all_feats:
            feats = CoverAlgorithm.load_features(self, i)
            ## Step 1: Compute chroma embeddings
            # First compute global chroma (used for OTI later)
            chroma = feats[self.chroma_type]
            gchroma = global_chroma(chroma)
            # Now downsample the chromas using median aggregation
            chroma = librosa.util.sync(chroma.T, np.arange(0, chroma.shape[0], self.downsample_fac), aggregate=np.median)
            # Finally, do a stacked delay embedding
            chroma_stacked = librosa.feature.stack_memory(chroma, self.tau, self.m).T
            
            ## Step 2: Compute MFCC Embeddings
            mfcc = feats['mfcc_htk']
            mfcc[np.isnan(mfcc)] = 0
            mfcc[np.isinf(mfcc)] = 0
            mfcc = librosa.util.sync(mfcc, np.arange(0, mfcc.shape[1], self.downsample_fac), aggregate=np.mean)
            mfcc_stacked = librosa.feature.stack_memory(mfcc, self.tau, self.m).T
            mag = np.sqrt(np.sum(mfcc_stacked**2, 1))
            mag[mag == 0] = 1
            mfcc_stacked /= mag[:, None]

            ## Step 3: Save away features
            feats = {'gchroma':gchroma, 'chroma_stacked':chroma_stacked, 'mfcc_stacked':mfcc_stacked}
            self.all_feats[i] = feats
        return self.all_feats[i]

    def similarity(self, idxs):
        N = idxs.shape[0]
        similarities = {'chroma_qmax':np.zeros(N), 'chroma_dmax':np.zeros(N), 'mfcc_qmax':np.zeros(N), 'mfcc_dmax':np.zeros(N)}
        for idx, (i,j) in enumerate(zip(idxs[:, 0], idxs[:, 1])):
            Si = self.load_features(i)
            Sj = self.load_features(j)
            ## Step 1: Do chroma similarities
            csm = get_csm_blocked_oti(Si['chroma_stacked'], Sj['chroma_stacked'], Si['gchroma'], Sj['gchroma'], get_csm_euclidean)
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

