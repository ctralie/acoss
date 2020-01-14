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

REC_SMOOTH = 9
PAD_LEN  = 2000

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
        CoverAlgorithm.__init__(self, "StrucFTM2D", datapath=datapath, shortname=shortname)
    
    def get_cacheprefix(self):
        """
        Return a descriptive file prefix to use for caching features
        and distance matrices
        """
        return "%s/%s_%s_%s"%(self.cachedir, self.name, self.shortname, self.chroma_type)
    def load_features(self, i, do_plot=False):
        filepath = "%s_%i.h5"%(self.get_cacheprefix(), i)
        figpath = "%s_%i.png"%(self.get_cacheprefix(), i)
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
        hpcp_orig = feats['hpcp']
        mfcc_orig = feats['mfcc_htk'].T
        tempogram_orig = librosa.feature.tempogram(onset_envelope=feats['madmom_features']['snovfn'], sr=sr, hop_length=hop_length).T

        # Downsample
        onsets = feats['madmom_features']['onsets']
        
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
        Ws, WFused = doSimilarityFusion([Dmfcc_stack, Dhpcp_stack], K=pK, niters=self.niters)

        if do_plot:
            plt.clf()
            for i, W in enumerate(Ws):
                plt.subplot(2, 2, i+1)
                plt.imshow(getHighContrastImage(W))
                plt.title(["MFCC", "HPCP"][i])
                plt.colorbar()
            plt.subplot(223)
            plt.imshow(getHighContrastImage(WFused))
            plt.title("Fused")
            plt.savefig(figpath, bbox_inches='tight')
        
        N = min(PAD_LEN, WFused.shape[0])
        Wres = np.zeros((PAD_LEN, PAD_LEN))
        Wres[0:N, 0:N] = WFused[0:N, 0:N]
        fft_mag = np.abs(scipy.fftpack.fft2(Wres))
        flat = fft_mag.flatten()
        shingle = np.log(flat/(np.sqrt(np.sum(flat**2))) + 1)
        # Set all elements except for the 5*PAD_LEN largest elements to zero
        cutoff = -np.partition(-shingle, PAD_LEN*5)[PAD_LEN*5-1]
        shingle[shingle < cutoff] = 0
        # Set to a sparse matrix to save memory and computation
        shingle = sparse.csr_matrix(shingle)

        # Get all 2D FFT magnitude shingles
        
        """
        shingles = btchroma_to_fftmat(chroma, self.WIN).T
        Norm = np.sqrt(np.sum(shingles**2, 1))
        Norm[Norm == 0] = 1
        shingles = np.log(self.C*shingles/Norm[:, None] + 1)
        shingle = np.median(shingles, 0) # Median aggregate
        shingle = shingle/np.sqrt(np.sum(shingle**2))
        """

        self.shingles[i] = shingle
        dd.io.save(filepath, {'shingle':shingle})
        return shingle

    def similarity(self, idxs):
        (a, b) = idxs.shape
        for k in range(a):
            i = idxs[k][0]
            j = idxs[k][1]
            s1 = self.load_features(i)
            s2 = self.load_features(j)
            d = euclidean_distances(s1, s2)
            # Since similarity should be high for two things
            # with a small distance, take the negative exponential
            sim = np.exp(-d*d)
            self.Ds['main'][i, j] = sim
    
    def all_pairwise(self, parallel=0, n_cores=12, symmetric=False):
        print("All Pairwise Fast")
        N = len(self.filepaths)
        x = self.load_features(0).toarray()
        X = sparse.csr_matrix((N, x.size))
        for i in range(N):
            if i%100 == 0:
                print(i)
            X[i, :] = self.load_features(i)
        print("Sparsity Factor: %.3g"%(X.size/(X.shape[0]*X.shape[1])))
        tic = time.time()
        XSqr = X.power(2)
        XSqr = np.array(np.sum(XSqr, 1)).flatten()
        DsSqr = XSqr[:, None] + XSqr[None, :] - 2*(X.dot(X.T)).toarray()
        self.Ds['main'] = np.exp(-DsSqr)
        self.get_all_clique_ids()
        print("Elapsed Time All Pairwise Fast: %.3g"%(time.time()-tic))


if __name__ == '__main__':
    #ftm2d_allpairwise_covers80(chroma_type='crema')
    parser = argparse.ArgumentParser(description="Benchmarking with 2D Fourier Transform Magnitude Coefficients",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", '--datapath', type=str, action="store", default='../features_covers80',
                        help="Path to data files")
    parser.add_argument("-s", "--shortname", type=str, action="store", default="Covers80", help="Short name for dataset")
    parser.add_argument("-c", '--chroma_type', type=str, action="store", default='hpcp',
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
    plt.figure(figsize=(15, 15))
    for i in range(len(strucHash.filepaths)):
        strucHash.load_features(i, do_plot=False)
    print('Feature loading done.')
    strucHash.all_pairwise(cmd_args.parallel, cmd_args.n_cores, symmetric=True)
    for similarity_type in strucHash.Ds.keys():
        strucHash.getEvalStatistics(similarity_type)
    strucHash.cleanup_memmap()

    print("... Done ....")