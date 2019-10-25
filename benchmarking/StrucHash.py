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

from skimage.transform import resize
from FTM2D import *
from CoverAlgorithm import *
from CRPUtils import *
from SimilarityFusion import *

REC_SMOOTH = 9

class StrucHash(CoverAlgorithm):
    def __init__(self, datapath="../features_covers80", chroma_type='hcpc', shortname='Covers80', wins_per_block=20, K=5, niters=3, do_sync=True):
        """
        Attributes
        """

        self.wins_per_block = wins_per_block
        self.chroma_type = chroma_type
        self.K=K
        self.niters = niters
        self.do_sync = do_sync
        self.shingles = {}
        CoverAlgorithm.__init__(self, "FTM2D", datapath=datapath, shortname=shortname)
    
    def get_cacheprefix(self):
        """
        Return a descriptive file prefix to use for caching features
        and distance matrices
        """
        return "%s/%s_%s_%s"%(self.cachedir, self.name, self.shortname, self.chroma_type)
    def load_features(self, i, do_plot=False):
        filepath = "%s_%i.h5"%(self.get_cacheprefix(), i)
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
        f = FTM2D()
        feats = CoverAlgorithm.load_features(f, i)

        hpcp_orig = feats['crema']
        mfcc_orig = feats['mfcc_htk'].T

        # Synchronize HPCP to the beats
        onsets = feats['madmom_features']['onsets']
        hpcp_sync = librosa.util.sync(hpcp_orig, onsets, aggregate=np.median)
        hpcp_sync[np.isnan(hpcp_sync)] = 0
        hpcp_sync[np.isinf(hpcp_sync)] = 0
        mfcc_sync = librosa.util.sync(mfcc_orig.T, onsets, aggregate=np.mean)
        mfcc_sync[np.isnan(mfcc_sync)] = 0
        mfcc_sync[np.isinf(mfcc_sync)] = 0

        hpcp_stack = librosa.feature.stack_memory(hpcp_sync, n_steps = self.wins_per_block)
        mfcc_stack = librosa.feature.stack_memory(mfcc_sync, n_steps = self.wins_per_block)

        # MFCC use straight Euclidean SSM
        Dmfcc_sync = get_ssm(mfcc_sync.T)
        Dmfcc_stack = get_ssm(mfcc_stack.T)
        # Chroma 
        Dhpcp_sync = get_csm_cosine(hpcp_sync.T, hpcp_sync.T)
        Dhpcp_stack = get_csm_cosine(hpcp_stack.T, hpcp_stack)

        FeatureNames = ['DMFCCs', 'Chroma']
        Ds = [Dmfcc_sync, Dhpcp_sync]   

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
        Ws = [getW(D, pK) for D in Ds]
        if REC_SMOOTH > 0:
            df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
            Ws = [df(W, size=(1, REC_SMOOTH)) for W in Ws]

        WFused_sync = doSimilarityFusionWs(Ws, K=pK, niters=self.niters)

        N = min(Dhpcp_stack.shape[0], Dmfcc_stack.shape[0])
        Dhpcp_stack = Dhpcp_stack[0:N, 0:N]
        Dmfcc_stack = Dmfcc_stack[0:N, 0:N]
        
        # Get all 2D FFT magnitude shingles
        
        """
        shingles = btchroma_to_fftmat(chroma, self.WIN).T
        Norm = np.sqrt(np.sum(shingles**2, 1))
        Norm[Norm == 0] = 1
        shingles = np.log(self.C*shingles/Norm[:, None] + 1)
        shingle = np.median(shingles, 0) # Median aggregate
        shingle = shingle/np.sqrt(np.sum(shingle**2))
        """

        if do_plot:
            import librosa.display
            plt.subplot(311)
            librosa.display.specshow(librosa.amplitude_to_db(hpcp_orig, ref=np.max))
            plt.title("Original")
            plt.subplot(312)
            librosa.display.specshow(librosa.amplitude_to_db(hpcp, ref=np.max))
            plt.title("Beat-synchronous Median Aggregated")
            plt.subplot(313)
            plt.imshow(np.reshape(shingle, (hpcp.shape[0], self.wins_per_block)))
            plt.title("Median FFT2D Mag Shingle")
            plt.show()
        self.shingles[i] = shingle
        dd.io.save(filepath, {'shingle':shingle})
        return shingle

if __name__ == '__main__':
    #ftm2d_allpairwise_covers80(chroma_type='crema')
    parser = argparse.ArgumentParser(description="Benchmarking with 2D Fourier Transform Magnitude Coefficients",
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

    cmd_args = parser.parse_args()

    strucHash = StrucHash(cmd_args.datapath, cmd_args.chroma_type, cmd_args.shortname)
    for i in range(len(strucHash.filepaths)):
        CoverAlgorithm.load_features(i)
    print('Feature loading done.')
    ftm2d.all_pairwise(cmd_args.parallel, cmd_args.n_cores, symmetric=True)
    for similarity_type in ftm2d.Ds.keys():
        ftm2d.getEvalStatistics(similarity_type)
    ftm2d.cleanup_memmap()

    print("... Done ....")