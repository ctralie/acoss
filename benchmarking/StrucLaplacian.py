import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.interpolate
import os
import librosa
import librosa.display
import argparse
from SimilarityFusion import doSimilarityFusionWs, getW
import subprocess
from scipy import sparse
from sklearn.metrics.pairwise import euclidean_distances

from skimage.transform import resize
from FTM2D import *
from CoverAlgorithm import *
from CRPUtils import *
from SimilarityFusion import *
from Laplacian import *

REC_SMOOTH = 9
PAD_LEN  = 2000

WIN_FAC = 10
FINAL_SIZE = 512

## Define a function that can plot the similarity images with higher contrast
def getHighContrastImage(W, noisefloor = 0.1):
    floor = np.quantile(W.flatten(), noisefloor)
    WShow = np.log(W+floor)
    np.fill_diagonal(WShow, 0)
    return WShow

class StrucHash(CoverAlgorithm):
    def __init__(self, datapath="../features_covers80", chroma_type='hcpc', shortname='Covers80', wins_per_block=20, K=10, niters=10, do_sync=True):
        """
        Attributes
        """

        self.wins_per_block = wins_per_block
        self.chroma_type = chroma_type
        self.K=K
        self.niters = niters
        self.do_sync = do_sync
        self.shingles = {}
        CoverAlgorithm.__init__(self, "StructureLaplacian", datapath=datapath, shortname=shortname)
    
    def get_cacheprefix(self):
        """
        Return a descriptive file prefix to use for caching features
        and distance matrices
        """
        return "%s/%s_%s_%s"%(self.cachedir, self.name, self.shortname, self.chroma_type)
    
    def load_features(self, i, do_plot=False):
        num = self.filepaths[i].split("/")[-1].split(".h5")[0]
        print(i, num)
        filepath = "%s_%s.h5"%(self.get_cacheprefix(), num)
        figpath = "%s_%s.png"%(self.get_cacheprefix(), num)
        if i in self.shingles:
            # If the result has already been cached in memory, 
            # return the cache
            return self.shingles[i]
        elif os.path.exists(filepath):
            # If the result has already been cached on disk, 
            # load it, save it in memory, and return
            self.shingles[i] = dd.io.load(filepath)['shingle']
            # Make sure to also load clique info as a side effect
            feats = CoverAlgorithm.load_features(self, i)
            return self.shingles[i]

        # Otherwise, compute the shingle
        import librosa.util
        feats = CoverAlgorithm.load_features(self, i)       

        hop_length=512
        sr=44100
        hpcp_orig = feats[self.chroma_type]
        mfcc_orig = feats['mfcc_htk'].T
        tempogram_orig = librosa.feature.tempogram(onset_envelope=feats['madmom_features']['snovfn'], sr=sr, hop_length=hop_length).T

        # Downsample
        onsets = feats['madmom_features']['onsets']
        # Beat-track
        #nHops = hpcp_orig.shape[0]-WIN_FAC*self.wins_per_block
        #onsets = np.arange(0, nHops, WIN_FAC)
        
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

        FeatureNames = ['DMFCCs', 'Chroma', 'Tempogram']
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
        # Do fusion on all features
        Ws, WFused = doSimilarityFusion([Dmfcc_stack, Dhpcp_stack, Dtempogram_stack], K=pK, niters=self.niters)
        w, v = getRandomWalkLaplacianEigsDense(WFused)
        

        if do_plot:
            plt.clf()
            for i, W in enumerate(Ws):
                plt.subplot(2, 3, i+1)
                plt.imshow(getHighContrastImage(W), cmap='magma_r')
                plt.title(["MFCC", self.chroma_type, "Tempogram"][i])
                plt.colorbar()
            plt.subplot(234)
            plt.imshow(getHighContrastImage(WFused), cmap='magma_r')
            plt.title("Fused")
            plt.subplot(235)
            plt.plot(w)
            plt.xlim([0, 30])
            plt.title("Eigenvalues")
            plt.subplot(236)
            plt.imshow(v, cmap='magma_r', aspect='auto')
            plt.xlim([0, 30])
            plt.title("Eigenvectors")
            plt.savefig(figpath, bbox_inches='tight')
        

        dd.io.save(filepath, {'WFused':WFused, 'w':w, 'v':v})
        shingle = w
        self.shingles[i] = shingle
        return shingle

    def similarity(self, idxs):
        (a, b) = idxs.shape
        for k in range(a):
            i = idxs[k][0]
            j = idxs[k][1]
            s1 = self.load_features(i)
            s2 = self.load_features(j)
            dSqr = np.sum((s1-s2)**2)
            # Since similarity should be high for two things
            # with a small distance, take the negative exponential
            sim = np.exp(-dSqr)
            self.Ds['main'][i, j] = sim
    
    def all_pairwise(self, parallel=0, n_cores=12, symmetric=False):
        N = len(self.filepaths)
        d = self.load_features(0).shape[0]
        X = np.zeros((N,d))
        for i in range(N):
            X[i, :] = self.load_features(i)
        tic = time.time()
        X = X[:, 0:100]
        X -= np.min(X)
        X += 0.01*np.max(X)
        plt.subplot(131)
        plt.imshow(X)
        plt.subplot(132)
        X = np.log(X)
        plt.imshow(X)
        XSqr = np.sum(X**2, 1)
        DsSqr = XSqr[:, None] + XSqr[None, :] - 2*(X.dot(X.T))
        plt.subplot(133)
        plt.imshow(DsSqr)
        plt.show()
        self.Ds['main'] = np.exp(-DsSqr)
        print("Elapsed Time All Pairwise Fast: %.3g"%(time.time()-tic))


if __name__ == '__main__':
    #ftm2d_allpairwise_covers80(chroma_type='crema')
    parser = argparse.ArgumentParser(description="Benchmarking with 2D Fourier Transform Magnitude Coefficients",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", '--datapath', type=str, action="store", default='../features_covers80',
                        help="Path to data files")
    parser.add_argument("-s", "--shortname", type=str, action="store", default="Covers80", help="Short name for dataset")
    parser.add_argument("-c", '--chroma_type', type=str, action="store", default='crema',
                        help="Type of chroma to use for experiments")
    parser.add_argument("-w", '--wins_per_block', type=int, action="store", default=20,
                        help="The number of windows per block.")
    parser.add_argument("-t", '--niters', type=int, action="store", default=3,
                        help="Number of iterations for similarity fusion.")
    parser.add_argument("-k", '--K', type=int, action="store", default=5,
                        help="The number of nearest neighbors for similarity fusion.")
    parser.add_argument("-y", '--synchronous', type=bool, action="store", default=True,
                        help="Do beat synchronous tracking or not.")
    parser.add_argument("-p", '--parallel', type=int, choices=(0, 1), action="store", default=0,
                        help="Parallel computing or not")
    parser.add_argument("-n", '--n_cores', type=int, action="store", default=1,
                        help="No of cores required for parallelization")

    cmd_args = parser.parse_args()

    strucHash = StrucHash(cmd_args.datapath, cmd_args.chroma_type, cmd_args.shortname, \
        cmd_args.wins_per_block, cmd_args.K, cmd_args.niters)
    plt.figure(figsize=(15, 10))
    for i in range(len(strucHash.filepaths)):
        strucHash.load_features(i, do_plot=True)
    print('Feature loading done.')
    strucHash.all_pairwise(cmd_args.parallel, cmd_args.n_cores, symmetric=True)
    for similarity_type in strucHash.Ds.keys():
        strucHash.getEvalStatistics(similarity_type)
    strucHash.cleanup_memmap()

    print("... Done ....")