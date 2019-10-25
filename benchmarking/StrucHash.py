import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.interpolate
import os
import librosa
import librosa.display
import argparse
from CSMSSMTools import getCSM, getCSMCosine
from SimilarityFusion import doSimilarityFusionWs, getW
from SongStructureGUI import saveResultsJSON
import subprocess

class StrucHash(CoverAlgorithm):

    def __init__(self, datapath="../features_covers80", chroma_type='hcpc', shortname='Covers80', PWR=1.96, WIN=75, C=5, K, reg_diag, reg_neighbs, niters, do_animation, plot_result, do_crema=True):
        """
        Attributes
        """
        
        self.PWR = PWR
        self.WIN = WIN
        self.C = C
        self.chroma_type = chroma_type
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
        feats = CoverAlgorithm.load_features(self, i)
        hpcp_orig = feats[self.chroma_type].T
        # Synchronize HPCP to the beats
        onsets = feats['madmom_features']['onsets']
        hpcp = librosa.util.sync(hpcp_orig, onsets, aggregate=np.median)
        chroma = chrompwr(hpcp, self.PWR)
        
        XChroma = librosa.feature.stack_memory(chroma, n_steps=WIN, mode='edge').T
        DChroma = getCSMCosine(XChroma, XChroma) #Cosine distance
        XMFCC = librosa.feature.stack_memory(mfcc, n_steps=WIN, mode='edge').T
        DMFCC = getCSM(XMFCC, XMFCC) #Euclidean distance 

        FeatureNames = ['MFCCs', 'Chromas']
        Ds = [DMFCC, DChroma]   

        # Edge case: If it's too small, zeropad SSMs
        for i, Di in enumerate(Ds):
            if Di.shape[0] < 2*K:
                D = np.zeros((2*K, 2*K))
                D[0:Di.shape[0], 0:Di.shape[1]] = Di
                Ds[i] = D

        pK = K
        if K == -1:
            pK = int(np.round(2*np.log(Ds[0].shape[0])/np.log(2)))
            print("Autotuned K = %i"%pK)
        # Do fusion on all features
        Ws = [getW(D, pK) for D in Ds]
        if REC_SMOOTH > 0:
            df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
            Ws = [df(W, size=(1, REC_SMOOTH)) for W in Ws]

        WFused = doSimilarityFusionWs(Ws, K=pK, niters=niters, \
            reg_diag=reg_diag, reg_neighbs=reg_neighbs, \
            do_animation=do_animation, PlotNames=FeatureNames, \
            PlotExtents=[times[0], times[-1]])
        
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
            plt.imshow(np.reshape(shingle, (hpcp.shape[0], self.WIN)))
            plt.title("Median FFT2D Mag Shingle")
            plt.show()
        self.shingles[i] = shingle
        dd.io.save(filepath, {'shingle':shingle})
        return shingle