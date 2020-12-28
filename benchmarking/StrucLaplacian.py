# -*- coding: utf-8 -*-
from pySeqAlign import qmax, dmax
from CoverAlgorithm import *
from CRPUtils import *
from SimilarityFusion import *
import numpy as np
import argparse
import librosa
import librosa.util
import librosa.display
from skimage.transform import resize
from Laplacian import *
import mir_eval
import scipy.signal
import os
from CurvatureTools import *

## Define a function that can plot the similarity images with higher contrast
def getHighContrastImage(W, noisefloor = 0.1):
    floor = np.quantile(W.flatten(), noisefloor)
    WShow = np.log(W+floor)
    np.fill_diagonal(WShow, 0)
    return WShow

class StrucHash(CoverAlgorithm):
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
    def __init__(self, datapath="../features_covers80", chroma_type='hpcp', shortname='benchmark', kappa=0.095, tau=1, m=10,
                 wins_per_block=20, K=10, niters=10, downsample_fac=40, do_sync=True, do_memmaps=True):
        self.wins_per_block = wins_per_block
        self.chroma_type = chroma_type
        self.kappa = kappa
        self.tau = tau
        self.m = m
        self.K=K
        self.niters = niters
        self.downsample_fac = downsample_fac
        self.do_sync = do_sync
        self.all_feats = {} # For caching features
        CoverAlgorithm.__init__(self, "StructureLaplacian", datapath=datapath, shortname=shortname, do_memmaps=do_memmaps, similarity_types=["snovfn_qmax", "snovfn_dmax"])

    def get_cacheprefix(self):
        """
        Return a descriptive file prefix to use for caching features
        and distance matrices
        """
        return "%s/%s_%s_%s"%(self.cachedir, self.name, self.shortname, self.chroma_type)

    def load_features(self, i, do_plot=False):
        if i in self.all_feats:
            return self.all_feats[i]
        filepath = "{}_{}.h5".format(self.get_cacheprefix(), i)
        if os.path.exists(filepath):
            self.all_feats[i] = dd.io.load(filepath)['X']
            return self.all_feats[i]

        feats = CoverAlgorithm.load_features(self, i)
        hop_length=512
        sr=44100
        hpcp_orig = feats[self.chroma_type]
        mfcc_orig = feats['mfcc_htk'].T
        tempogram_orig = librosa.feature.tempogram(onset_envelope=feats['madmom_features']['snovfn'], sr=sr, hop_length=hop_length).T
        if self.do_sync:
            # Beat-track
            onsets = feats['madmom_features']['onsets']
        else:
            # Uniformly downsample
            onsets = np.arange(0, mfcc_orig.shape[0], self.downsample_fac)
        
        hpcp_sync = librosa.util.sync(hpcp_orig.T, onsets, aggregate=np.median)
        hpcp_sync[np.isnan(hpcp_sync)] = 0
        hpcp_sync[np.isinf(hpcp_sync)] = 0
        hpcp_stack = librosa.feature.stack_memory(hpcp_sync, n_steps = self.wins_per_block)
        
        mfcc_sync = librosa.util.sync(mfcc_orig.T, onsets, aggregate=np.mean)
        mfcc_sync[np.isnan(mfcc_sync)] = 0
        mfcc_sync[np.isinf(mfcc_sync)] = 0
        mfcc_stack = librosa.feature.stack_memory(mfcc_sync, n_steps = self.wins_per_block)

        tempogram_sync = librosa.util.sync(tempogram_orig.T, onsets, aggregate=np.mean)
        tempogram_sync[np.isnan(tempogram_sync)] = 0
        tempogram_sync[np.isinf(tempogram_sync)] = 0
        tempogram_stack = librosa.feature.stack_memory(tempogram_sync, n_steps = self.wins_per_block)

        # MFCC use straight Euclidean SSM
        Dmfcc_stack = get_ssm(mfcc_stack.T)
        # Chroma 
        Dhpcp_stack = get_csm_cosine(hpcp_stack.T, hpcp_stack.T)
        # Tempogram with Euclidean
        Dtempogram_stack = get_ssm(tempogram_stack.T)

        N = min(min(Dhpcp_stack.shape[0], Dmfcc_stack.shape[0]), Dtempogram_stack.shape[0])
        Dhpcp_stack = Dhpcp_stack[0:N, 0:N]
        Dmfcc_stack = Dmfcc_stack[0:N, 0:N]
        Dtempogram_stack = Dtempogram_stack[0:N, 0:N]
        onsets = onsets[0:N]

        Ds = [Dmfcc_stack, Dhpcp_stack, Dtempogram_stack]  
        # Edge case: If it's too small, zeropad SSMs
        for i, Di in enumerate(Ds):
            if Di.shape[0] < 2*self.K:
                D = np.zeros((2*self.K, 2*self.K))
                D[0:Di.shape[0], 0:Di.shape[1]] = Di
                Ds[i] = D

        pK = self.K
        if self.K == -1:
            pK = int(np.round(2*np.log(Ds[0].shape[0])/np.log(2)))
            print("Autotuned K = %i"%pK)
        ## Do fusion on all features
        Ws, WFused = snf([Dmfcc_stack, Dhpcp_stack, Dtempogram_stack], K=pK, niters=self.niters, verbose_times=False)

        ## Create the meet matrix
        times = onsets*hop_length/sr
        lapfn = getRandomWalkLaplacianEigsDense
        specfn = lambda v, dim, times: spectralClusterSequential(v, dim, times, rownorm=False)
        neigs=10
        vs = lapfn(WFused)
        labels = [specfn(vs, k, times) for k in range(2, neigs+1)]
        specintervals_hier = [res['intervals_hier'] for res in labels]
        speclabels_hier = [res['labels_hier'] for res in labels]
        interval = np.mean(times[1::]-times[0:-1])
        L = np.asarray(mir_eval.hierarchy._meet(specintervals_hier, speclabels_hier, interval).todense())
        ## Create Euclidean features for the meet matrix by using an SVD
        U, s, _ = linalg.svd(L)
        s = s[0:neigs]
        s /= s[0]
        X = U[:, 0:neigs]*s[None, :]
        
        X = getCurvVectors(X, 3, 2)
        X = np.array(X)
        X = np.sqrt(np.sum(X**2, axis=2)).T
        X = X[:, 1]
        X = sliding_window(X[:, None], self.m)
        
        if do_plot:
            plt.clf()
            plt.subplot(221)
            plt.imshow(L)
            plt.title("Meet matrix")
            plt.subplot(223)
            plt.plot(X)
            plt.title("Structural Novelty")
            plt.subplot(222)
            plt.plot(s)
            plt.subplot(224)
            plt.imshow(-get_ssm(X))
            plt.colorbar()
            plt.show()
            plt.savefig("{}_{}.png".format(self.get_cacheprefix(), i), bbox_inches='tight')
        
        self.all_feats[i] = X
        dd.io.save(filepath, {"X":X})
        return X
    
    def similarity(self, idxs):
        N = idxs.shape[0]
        similarities = {s:np.zeros(N) for s in ['snovfn_qmax', 'snovfn_dmax']}
        for idx, (i,j) in enumerate(zip(idxs[:, 0], idxs[:, 1])):
            Si = self.load_features(i)
            Sj = self.load_features(j)
            csm = get_csm(Si, Sj)
            csm = csm_to_binary(csm, self.kappa)
            M, N = csm.shape[0], csm.shape[1]
            D = np.zeros(M*N, dtype=np.float32)
            similarities['snovfn_qmax'][idx] = qmax(csm.flatten(), D, M, N) / (M+N)
            similarities['snovfn_dmax'][idx] = dmax(csm.flatten(), D, M, N) / (M+N)
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
    parser.add_argument("-w", '--wins_per_block', type=int, action="store", default=20,
                        help="The number of windows per block.")
    parser.add_argument("-t", '--niters', type=int, action="store", default=5,
                        help="Number of iterations for similarity fusion.")
    parser.add_argument("-k", '--K', type=int, action="store", default=5,
                        help="The number of nearest neighbors for similarity fusion.")
    parser.add_argument("-y", '--synchronous', type=bool, action="store", default=False,
                        help="Do beat synchronous tracking or not.")
    parser.add_argument("-p", '--parallel', type=int, choices=(0, 1), action="store", default=0,
                        help="Parallel computing or not")
    parser.add_argument("-n", '--n_cores', type=int, action="store", default=1,
                        help="No of cores required for parallelization")
    parser.add_argument("-r", "--range", type=str, action="store", default="")
    parser.add_argument("-b", "--batch_path", type=str, action="store", default="")

    cmd_args = parser.parse_args()
    
    plt.figure(figsize=(12, 12))
    do_memmaps = True
    if (len(cmd_args.range) > 0):
        do_memmaps = False
    strucHash = StrucHash(cmd_args.datapath, cmd_args.chroma_type, cmd_args.shortname, wins_per_block=cmd_args.wins_per_block, niters=cmd_args.niters, K=cmd_args.K, do_sync=cmd_args.synchronous, do_memmaps=do_memmaps)
    
    if len(cmd_args.batch_path) > 0:
        # Aggregrate precomputed similarities
        strucHash.load_batches(cmd_args.batch_path)
        for similarity_type in strucHash.Ds.keys():
            strucHash.getEvalStatistics(similarity_type)
    else:
        if do_memmaps:
            # Do the whole thing
            strucHash.all_pairwise(cmd_args.parallel, cmd_args.n_cores, symmetric=True)
            for similarity_type in strucHash.Ds.keys():
                print(similarity_type)
                strucHash.getEvalStatistics(similarity_type)
            strucHash.cleanup_memmap()
        else:
            # Do only a range and save it
            [w, idx] = [int(s) for s in cmd_args.range.split("-")]
            strucHash.do_batch(w, idx, "cache/struc")
    
    print("... Done ....")

