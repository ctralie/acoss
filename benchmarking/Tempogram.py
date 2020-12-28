# -*- coding: utf-8 -*-
from pySeqAlign import qmax, dmax
from CoverAlgorithm import *
from CRPUtils import *
import numpy as np
import argparse
import librosa
from skimage.transform import resize
import scipy.signal

COMMON_SIZE = -1 #20000 # Roughly 4 minutes of audio at 44100hz and a 512 hop size

class TGAlg(CoverAlgorithm):
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
    def __init__(self, datapath="../features_covers80", shortname='benchmark', 
                oti=True, kappa=0.095, tau=1, m=9, downsample_fac=40, do_memmaps=True):
        self.oti = oti
        self.tau = tau
        self.m = m
        self.kappa = kappa
        self.tau = tau
        self.m = m
        self.downsample_fac = downsample_fac
        self.all_feats = {} # For caching features (global chroma and stacked chroma)
        CoverAlgorithm.__init__(self, "TGAlg", datapath=datapath, shortname=shortname, do_memmaps=do_memmaps, similarity_types=["tempogram_rnn_qmax", "tempogram_rnn_dmax", "tempogram_sflux_qmax", "tempogram_sflux_dmax"])

    def load_features(self, i):
        if not i in self.all_feats:
            feats = {}
            m = CoverAlgorithm.load_features(self, i)['madmom_features']
            for name, novfn in zip(["tempogram_rnn", "tempogram_sflux"], [m['novfn'], m['snovfn']]):
                x = novfn
                if COMMON_SIZE > -1:
                    l = np.lcm(novfn.size, COMMON_SIZE)
                    x = scipy.signal.resample_poly(novfn, int(l/novfn.size), int(l/COMMON_SIZE))
                tempogram = librosa.feature.tempogram(onset_envelope=x, sr=44100, hop_length=512)
                tempogram = librosa.util.sync(tempogram, np.arange(0, tempogram.shape[1], self.downsample_fac), aggregate=np.mean).T
                feats[name] = tempogram
            self.all_feats[i] = feats
        return self.all_feats[i]

    def similarity(self, idxs):
        N = idxs.shape[0]
        similarities = {f:np.zeros(N) for f in ["tempogram_rnn_qmax", "tempogram_rnn_dmax", "tempogram_sflux_qmax", "tempogram_sflux_dmax"]}
        for idx, (i,j) in enumerate(zip(idxs[:, 0], idxs[:, 1])):
            Si = self.load_features(i)
            Sj = self.load_features(j)
            ## Step 1: Do chroma similarities
            for f in ["tempogram_rnn", "tempogram_sflux"]:
                csm = get_csm(Si[f], Sj[f])
                csm = csm_to_binary(csm, self.kappa)
                M, N = csm.shape[0], csm.shape[1]
                D = np.zeros(M*N, dtype=np.float32)
                similarities["{}_qmax".format(f)][idx] = qmax(csm.flatten(), D, M, N) / (M+N)
                similarities["{}_dmax".format(f)][idx] = dmax(csm.flatten(), D, M, N) / (M+N)
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
    tgalg = TGAlg(cmd_args.datapath, cmd_args.shortname, do_memmaps=do_memmaps)
    
    if len(cmd_args.batch_path) > 0:
        # Aggregrate precomputed similarities
        tgalg.load_batches(cmd_args.batch_path)
        for similarity_type in tgalg.Ds.keys():
            tgalg.getEvalStatistics(similarity_type)
    else:
        if do_memmaps:
            # Do the whole thing
            tgalg.all_pairwise(cmd_args.parallel, cmd_args.n_cores, symmetric=True)
            for similarity_type in tgalg.Ds.keys():
                print(similarity_type)
                tgalg.getEvalStatistics(similarity_type)
            tgalg.cleanup_memmap()
        else:
            # Do only a range and save it
            [w, idx] = [int(s) for s in cmd_args.range.split("-")]
            tgalg.do_batch(w, idx, "cache/tempogram")
    
    print("... Done ....")

