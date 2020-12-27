# -*- coding: utf-8 -*-
from pySeqAlign import qmax, dmax
from CoverAlgorithm import *
from Serra09 import *
from CRPUtils import *
from SimilarityFusion import *
import numpy as np
import argparse
import librosa


class EarlySNF(Serra09):
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
        CoverAlgorithm.__init__(self, "EarlySNF", datapath=datapath, shortname=shortname, do_memmaps=do_memmaps, similarity_types=["chroma_qmax", "chroma_dmax", "mfcc_qmax", "mfcc_dmax", "snf_qmax", "snf_dmax"])

    def similarity(self, idxs):
        N = idxs.shape[0]
        similarities = {'chroma_qmax':np.zeros(N), 'chroma_dmax':np.zeros(N), 'mfcc_qmax':np.zeros(N), 'mfcc_dmax':np.zeros(N), "snf_qmax":np.zeros(N), "snf_dmax":np.zeros(N)}
        for idx, (i,j) in enumerate(zip(idxs[:, 0], idxs[:, 1])):
            Si = self.load_features(i)
            Sj = self.load_features(j)
            Ws = []
            ## Step 1: Get chroma matrices
            csm = get_csm_blocked_oti(Si['chroma_stacked'], Sj['chroma_stacked'], Si['gchroma'], Sj['gchroma'], get_csm_euclidean)
            M, N = csm.shape[0], csm.shape[1]
            K = int(self.kappa*(M+N))
            ssma = get_ssm(Si['chroma_stacked'])
            ssmb = get_ssm(Sj['chroma_stacked'])
            Ws.append(get_WCSMSSM(ssma, ssmb, csm, K))
            # Might as well do Serra09 while we're at it
            csm = csm_to_binary(csm, self.kappa)
            D = np.zeros(M*N, dtype=np.float32)
            similarities['chroma_qmax'][idx] = qmax(csm.flatten(), D, M, N) / (M+N)
            similarities['chroma_dmax'][idx] = dmax(csm.flatten(), D, M, N) / (M+N)

            ## Step 2: Get mfcc matrices
            csm = get_csm(Si['mfcc_stacked'], Sj['mfcc_stacked'])
            ssma = get_ssm(Si['mfcc_stacked'])
            ssmb = get_ssm(Sj['mfcc_stacked'])
            Ws.append(get_WCSMSSM(ssma, ssmb, csm, K))
            # Might as well do Serra09 while we're at it
            csm = csm_to_binary(csm, self.kappa)
            D = np.zeros(M*N, dtype=np.float32)
            similarities['mfcc_qmax'][idx] = qmax(csm.flatten(), D, M, N) / (M+N)
            similarities['mfcc_dmax'][idx] = dmax(csm.flatten(), D, M, N) / (M+N)

            ## Step 3: Do early fusion
            csm = snf_ws(Ws, K = K, niters = 5, reg_diag = True, verbose_times=False)
            csm = -csm[0:M, M::] # Do negative since this is a similarity but binary csm expects difference
            csm = csm_to_binary(csm, self.kappa)
            D = np.zeros(M*N, dtype=np.float32)
            similarities['snf_qmax'][idx] = qmax(csm.flatten(), D, M, N) / (M+N)
            similarities['snf_dmax'][idx] = dmax(csm.flatten(), D, M, N) / (M+N)

            if self.do_memmaps:
                for key in self.Ds.keys():
                    self.Ds[key][i][j] = similarities[key][idx]
        return similarities

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmarking with early fusion + QMax/DMax",
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
    earlySNF = EarlySNF(cmd_args.datapath, cmd_args.chroma_type, cmd_args.shortname, do_memmaps=do_memmaps)
    
    if len(cmd_args.batch_path) > 0:
        # Aggregrate precomputed similarities
        earlySNF.load_batches(cmd_args.batch_path)
        for similarity_type in earlySNF.Ds.keys():
            earlySNF.getEvalStatistics(similarity_type)
    else:
        if do_memmaps:
            # Do the whole thing
            earlySNF.all_pairwise(cmd_args.parallel, cmd_args.n_cores, symmetric=True)
            for similarity_type in earlySNF.Ds.keys():
                print(similarity_type)
                earlySNF.getEvalStatistics(similarity_type)
            earlySNF.cleanup_memmap()
        else:
            # Do only a range and save it
            [w, idx] = [int(s) for s in cmd_args.range.split("-")]
            earlySNF.do_batch(w, idx, "cache/earlysnf")
    
    print("... Done ....")

