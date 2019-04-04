# -*- coding: utf-8 -*-
from pySeqAlign import qmax, dmax
from CoverAlgorithm import *
from CRPUtils import *
import numpy as np
import argparse
import librosa
from SimilarityFusion import *


def global_chroma(chroma):
    """Computes global chroma of a input chroma vector"""
    if chroma.shape[1] not in [12, 24, 36]:
        raise IOError("Wrong axis for the input chroma array. Expected shape '(frame_size, bin_size)'")
    return np.divide(chroma.sum(axis=0), np.max(chroma.sum(axis=0)))

class ChenFusion(CoverAlgorithm):
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
                oti=True, kappa=0.095, tau=1, m=9, downsample_fac=40):
        self.oti = oti
        self.tau = tau
        self.m = m
        self.chroma_type = chroma_type
        self.kappa = kappa
        self.tau = tau
        self.m = m
        self.downsample_fac = downsample_fac
        self.all_feats = {} # For caching features (global chroma and stacked chroma)
        CoverAlgorithm.__init__(self, "ChenFusion", similarity_types=["qmax", "dmax"], datapath=datapath, shortname=shortname)

    def load_features(self, i):
        if not i in self.all_feats:
            feats = CoverAlgorithm.load_features(self, i)
            # First compute global chroma (used for OTI later)
            chroma = feats[self.chroma_type]
            gchroma = global_chroma(chroma)
            # Now downsample the chromas using median aggregation
            chroma = librosa.util.sync(chroma.T, np.arange(0, chroma.shape[0], self.downsample_fac), aggregate=np.median)
            # Finally, do a stacked delay embedding
            stacked = librosa.feature.stack_memory(chroma, self.tau, self.m).T
            feats = {'gchroma':gchroma, 'stacked':stacked}
            self.all_feats[i] = feats
        return self.all_feats[i]

    def similarity(self, i, j):
        Si = self.load_features(i)
        Sj = self.load_features(j)
        csm = get_csm_blocked_oti(Si['stacked'], Sj['stacked'], Si['gchroma'], Sj['gchroma'], get_csm_euclidean)
        csm = csm_to_binary(csm, self.kappa)
        M, N = csm.shape[0], csm.shape[1]
        D = np.zeros(M*N, dtype=np.float32)
        scores = {}
        scores["qmax"] = qmax(csm.flatten(), D, M, N)
        scores["dmax"] = dmax(csm.flatten(), D, M, N)
        for s in scores:
            if not os.path.exists('cache/distances'):
                os.mkdir('cache/distances')
            if not os.path.exists('cache/distances/{}'.format(s)):
                os.mkdir('cache/distances/{}'.format(s))
            np.savetxt('cache/distances/{}/{}_{}.txt'.format(s, i, j), np.array([scores[s]]).astype('float16'), fmt='%1.3f')
            #self.Ds[s][i, j] = scores[s]
    
    def normalize_by_length(self):
        """
        Do a non-symmetric normalization by length
        """
        N = len(self.filepaths)
        for j in range(N):
            f = self.load_features(j)
            norm_fac = np.sqrt(f['stacked'].shape[0])
            for i in range(N):
                for key in self.Ds:     
                    self.Ds[key][i, j] = norm_fac/self.Ds[key][i, j]
    
    def do_late_fusion(self):
        DLate = doSimilarityFusion([self.Ds[s] for s in self.Ds], K=20, niters=20, reg_diag=1)[1]
        for key in self.Ds:
            self.Ds[key] *= -1 # Switch back to larger scores being closer
        self.Ds["Late"] = DLate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmarking with Joan Serra's Cover id algorithm",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", '--datapath', type=str, action="store", default='../features_covers80',
                        help="Path to data files")
    parser.add_argument("-s", "--shortname", type=str, action="store", default="Covers80", help="Short name for dataset")
    parser.add_argument("-c", '--chroma_type', type=str, action="store", default='hpcp',
                        help="Type of chroma to use for experiments")
    parser.add_argument("-p", '--parallel', type=int, choices=(0, 1), action="store", default=0,
                        help="Parallel computing or not")
    parser.add_argument("-n", '--n_cores', type=int, action="store", default=1,
                        help="No of cores required for parallelization")
    parser.add_argument("-i", '--idx', type=int, action="store", default=0,
                        help="Index of pairs")

    cmd_args = parser.parse_args()

    from itertools import combinations
    import time

    start = time.monotonic()
    cf = ChenFusion(cmd_args.datapath, cmd_args.chroma_type, cmd_args.shortname)
    N = len(cf.filepaths)
    blockdim = 100
    batch_size = blockdim**2
    blockres = int(N/blockdim) # Number of blocks across an axis
    NPairs = int(blockdim*blockdim*blockres*(blockres-1)/2)
    print("%i Pairs total across %i batches"%(NPairs, NPairs/batch_size))

    ## Setup pairs
    all_pairs = np.zeros((NPairs, 2), dtype=int)
    I, J = np.meshgrid(np.arange(blockdim), np.arange(blockdim))
    blockidx = np.array([I.flatten(), J.flatten()]).T
    idx = 0
    for blocki in range(blockdim):
        for blockj in range(blocki, blockdim):
            all_pairs[idx*batch_size:(idx+1)*batch_size, :] = blockidx + np.array([[blocki*blockdim, blockj*blockdim]])
            idx += 1

    ## Run the appropriate batch
    tic = time.time()
    for index in range(cmd_args.idx*batch_size,(cmd_args.idx+1)*batch_size):
        print(all_pairs[index, 0], all_pairs[index, 1])
        cf.similarity(all_pairs[index, 0], all_pairs[index, 1])
        if index == 10000:
            print('hit 10000 {}'.format(time.monotonic()-start))
    print("Elapsed Time Batch %i: %.3g"%(cmd_args.idx, time.time()-tic))
