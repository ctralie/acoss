# -*- coding: utf-8 -*-
from CoverAlgorithm import *
import numpy as np
import argparse
from kymatio.numpy import Scattering1D
import scipy.signal

DOWNSAMPLE_FAC = 16

class ANFScattering(CoverAlgorithm):
    """
    Attributes
    ----------
    all_feats: {int: dictionary}
        Cached features
    """
    def __init__(self, datapath="../features_covers80", shortname='benchmark', J = 6, T = 2**14, Q = 8, do_memmaps=True):
        self.all_feats = {} # For caching features
        self.J = J
        self.T = T
        self.Q = Q
        self.S = Scattering1D(J, T, Q)
        self.S2 = Scattering1D(J, int(T/DOWNSAMPLE_FAC), Q)
        CoverAlgorithm.__init__(self, "ANFScattering", datapath=datapath, shortname=shortname, do_memmaps=do_memmaps, similarity_types=["anfrnn", "anfrnn_shingle", "anfsuperflux", "anfsuperflux_shingle"])

    def load_features(self, i):
        if not i in self.all_feats:
            m = CoverAlgorithm.load_features(self, i)['madmom_features']
            feats = {}
            for name, novfn in zip(['anfrnn', 'anfsuperflux'], [m['novfn'], m['snovfn']]):
                l = np.lcm(novfn.size, self.T)
                x = scipy.signal.resample_poly(novfn, int(l/novfn.size), int(l/self.T))
                ## Step 1: Do global scattering
                y = x - np.mean(x)
                y = y/np.sqrt(np.sum(y**2))
                y = self.S.scattering(y)
                feats[name] = y.flatten()
                ## Step 2: Do shingled scattering
                win = int(x.size/DOWNSAMPLE_FAC)
                X = []
                for k in range(DOWNSAMPLE_FAC):
                    y = x[k*win:(k+1)*win]
                    y = y - np.mean(y)
                    y = y/np.sqrt(np.sum(y**2))
                    y = self.S2.scattering(y)
                    X.append(y.flatten())
                X = np.array(X)
                y = np.median(X, 0)
                y = y/np.sqrt(np.sum(y**2))
                feats["{}_shingle".format(name)] = y
            self.all_feats[i] = feats
        return self.all_feats[i]

    def similarity(self, idxs):
        N = idxs.shape[0]
        similarities = {n:np.zeros(N) for n in self.similarity_types}
        for idx, (i,j) in enumerate(zip(idxs[:, 0], idxs[:, 1])):
            Si = self.load_features(i)
            Sj = self.load_features(j)
            for name in Si:
                diff = Si[name] - Sj[name]
                d = np.sqrt(np.sum(diff**2))
                similarities[name][idx] = d
            if self.do_memmaps:
                for key in self.Ds.keys():
                    self.Ds[key][i][j] = similarities[key][idx]
        return similarities

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmarking with audio novelty function scattering",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", '--datapath', type=str, action="store", default='../features_covers80',
                        help="Path to data files")
    parser.add_argument("-s", '--shortname', type=str, action="store", default="covers80", help="Short name for dataset")
    parser.add_argument("-j", '--J', type=int, action="store", default=6, help="Number of levels in scattering transform")
    parser.add_argument("-t", '--T', type=int, action="store", default=2**13, help="Length of uniformly scaled ANF (must be power of 2)")
    parser.add_argument("-q", '--Q', type=int, action="store", default=8, help="Number of wavelets per level")
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
    anf = ANFScattering(cmd_args.datapath, cmd_args.shortname, J=cmd_args.J, T=cmd_args.T, Q=cmd_args.Q, do_memmaps=do_memmaps)
    
    if len(cmd_args.batch_path) > 0:
        # Aggregrate precomputed similarities
        anf.load_batches(cmd_args.batch_path)
        for similarity_type in anf.Ds.keys():
            anf.getEvalStatistics(similarity_type)
    else:
        if do_memmaps:
            # Do the whole thing
            anf.all_pairwise(cmd_args.parallel, cmd_args.n_cores, symmetric=True)
            for similarity_type in anf.Ds.keys():
                print(similarity_type)
                anf.getEvalStatistics(similarity_type)
            anf.cleanup_memmap()
        else:
            # Do only a range and save it
            [w, idx] = [int(s) for s in cmd_args.range.split("-")]
            anf.do_batch(w, idx, "cache/anfscattering")
    
    print("... Done ....")

