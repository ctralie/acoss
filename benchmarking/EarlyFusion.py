import numpy as np
from CoverAlgorithm import *
from SimilarityFusion import *
import argparse
from pySeqAlign import swconstrained as alignment_fn
from CRPUtils import *

"""====================================================
                UTILITY FUNCTIONS
===================================================="""

def resize_block(X, i1, i2, frames_per_block, median_aggregate = False):
    """
    Median aggregate features into a coarser list
    Parameters
    ----------
    X: ndarray(n_frames, n_feats)
        An array of features
    i1: int
        Index at beginning of block
    i2: int
        Index at end of block
    frames_per_block: int
        Number of frames to which to downsample
    """
    if median_aggregate:
        import librosa
        idxs = np.linspace(i1, i2, frames_per_block-1)
        idxs = np.array(np.floor(idxs), dtype=int)
        res = librosa.util.sync(X.T, idxs, aggregate=np.median).T
        ret = res
        if res.shape[0] > frames_per_block:
            ret = res[0:frames_per_block, :]
        elif res.shape[0] < frames_per_block:
            ret = np.zeros((frames_per_block, res.shape[1]))
            ret[0:res.shape[0], :] = res
        return ret
    else:
        import skimage.transform
        x = X[i1:i2, :]
        x = x.astype('float64')
        ret = skimage.transform.resize(x, (frames_per_block, x.shape[1]), anti_aliasing=True, mode='constant')
        ret[np.isinf(ret)] = 0
        ret[np.isnan(ret)] = 0
        return ret


"""====================================================
            FEATURE COMPUTATION/COMPARISON
===================================================="""

class EarlyFusion(CoverAlgorithm):
    """
    Attributes
    ----------
    Same as CoverAlgorithms, plus
    chroma_type: string
        Type of chroma to use (key into features)
    blocksize: int
        Number of beats to take in each block
    mfccs_per_block: int
        Resize to this number of MFCC frames in each block
    chroma_blocksize: int
        Number of beats to take in a chroma block
    chromas_per_block: int
        Resize to this number of chroma frames in each block
    kappa: float
        Neighborhood factor for binary thresholding
    K: int
        Number of nearest neighbors to use in SNF
    niters: int
        Number of iterations in SNF
    all_block_feats: dict
        A cache of features computed by load_features
    """
    def __init__(self, datapath="../features_covers80", chroma_type='hpcp', shortname='Covers80', blocksize=20, mfccs_per_block=50, ssm_res=50, chromas_per_block=40, kappa=0.1, K=10, niters=5, log_times=False):
        self.chroma_type = chroma_type
        self.blocksize = blocksize
        self.mfccs_per_block = mfccs_per_block
        self.chromas_per_block = chromas_per_block
        self.kappa = kappa
        self.K = K
        self.niters = niters
        self.all_block_feats = {} # Cached features
        self.log_times = log_times
        if log_times:
            self.times = {'features':[], 'raw':[]}
        CoverAlgorithm.__init__(self, "EarlyFusion", datapath=datapath, shortname=shortname, \
                                similarity_types=["mfccs", "ssms", "chromas", "early"])

    def get_cacheprefix(self):
        """
        Return a descriptive file prefix to use for caching features
        and distance matrices
        """
        return "%s/%s_%s_%s"%(self.cachedir, self.name, self.shortname, self.chroma_type)

    def load_features(self, i, do_plot=False):
        """
        Return a dictionary of all of the beat-synchronous blocked features
        Parameters
        ----------
        i: int
            Index of the song in the corpus for which to compute features
        Returns
        -------
        block_features: dict {
            'mfccs': ndarray(n_blocks, 20*mfccs_per_block)
                Array of blocked Z-normalized raw MFCCs
            'ssms': ndarray(n_blocks, mfccs_per_block*(mfccs_per_block-1)/2)
                Upper triangular part of all SSMs for blocked
                Z-normalized MFCCs
            'chromas': ndarray(n_blocks, 12*chroma_dim)
                Array of blocked chromas
            'chroma_med': ndarray(12)
                Median of all chroma frames across song (for OTI)
        }
        """
        filepath = "%s_%i.h5"%(self.get_cacheprefix(), i)
        if i in self.all_block_feats:
            # If the result has already been cached in memory,
            # return the cache
            return self.all_block_feats[i]
        elif os.path.exists(filepath):
            # If the result has already been cached on disk,
            # load it, save it in memory, and return
            #self.all_block_feats[i] = dd.io.load(filepath)
            temp_feat = dd.io.load(filepath)
            # Make sure to also load clique info as a side effect
            #feats = CoverAlgorithm.load_features(self, i)
            return temp_feat
        tic = time.time()
        block_feats = {}
        feats = CoverAlgorithm.load_features(self, i)
        chroma = feats[self.chroma_type]
        mfcc = feats['mfcc_htk'].T
        mfcc[np.isnan(mfcc)] = 0

        onsets = feats['madmom_features']['onsets']
        n_beats = len(onsets)
        n_blocks = n_beats - self.blocksize

        ## Step 1: Compute raw MFCC and MFCC SSM blocked features
        # Allocate space for MFCC-based features
        block_feats['mfccs'] = np.zeros((n_blocks, self.mfccs_per_block*mfcc.shape[1]), dtype=np.float32)
        pix = np.arange(self.mfccs_per_block)
        I, J = np.meshgrid(pix, pix)
        dpixels = int(self.mfccs_per_block*(self.mfccs_per_block-1)/2)
        block_feats['ssms'] = np.zeros((n_blocks, dpixels), dtype=np.float32)
        # Compute MFCC-based features
        for b in range(n_blocks):
            i1 = onsets[b]
            i2 = onsets[b+self.blocksize-1]
            x = resize_block(mfcc, i1, i2, self.mfccs_per_block)
            # Z-normalize
            x -= np.mean(x, 0)[None, :]
            xnorm = np.sqrt(np.sum(x**2, 1))[:, None]
            xnorm[xnorm == 0] = 1
            xn = x / xnorm
            block_feats['mfccs'][b, :] = xn.flatten()
            # Create SSM, resize, and save
            D = get_ssm(xn)
            block_feats['ssms'][b, :] = D[I < J] # Upper triangular part

        ## Step 2: Compute chroma blocks
        block_feats['chromas'] = np.zeros((n_blocks, self.chromas_per_block*chroma.shape[1]), dtype=np.float32)
        block_feats['chroma_med'] = np.median(chroma, axis=0)
        for b in range(n_blocks):
            i1 = onsets[b]
            i2 = onsets[b+self.blocksize]
            x = resize_block(chroma, i1, i2, self.chromas_per_block)
            block_feats['chromas'][b, :] = x.flatten()

        ## Step 3: Precompute Ws for each features
        """ Skip this since I'm doing a simpler, accelerated early fusion
        ssm_fns = {'chromas':lambda x: get_csm_cosine(x, x), 'mfccs':get_ssm, 'ssms':get_ssm}
        for feat in ssm_fns:
            d = ssm_fns[feat](block_feats[feat])
            block_feats['%s_W'%feat] = getW(d, self.K)
        """

        self.all_block_feats[i] = block_feats # Cache features
        dd.io.save(filepath, block_feats)
        if self.log_times:
            self.times['features'].append(time.time()-tic)
        return block_feats


    def load_and_write(self, i):
        filepath = "%s_%i.h5"%(self.get_cacheprefix(), i)
        if not os.path.exists(filepath):
            block_feats = {}
            feats = CoverAlgorithm.load_features(self, i)
            chroma = feats[self.chroma_type]
            mfcc = feats['mfcc_htk'].T
            mfcc[np.isnan(mfcc)] = 0

            onsets = feats['madmom_features']['onsets']
            n_beats = len(onsets)
            n_blocks = n_beats - self.blocksize

            ## Step 1: Compute raw MFCC and MFCC SSM blocked features
            # Allocate space for MFCC-based features
            block_feats['mfccs'] = np.zeros((n_blocks, self.mfccs_per_block*mfcc.shape[1]), dtype=np.float32)
            pix = np.arange(self.mfccs_per_block)
            I, J = np.meshgrid(pix, pix)
            dpixels = int(self.mfccs_per_block*(self.mfccs_per_block-1)/2)
            block_feats['ssms'] = np.zeros((n_blocks, dpixels), dtype=np.float32)
            # Compute MFCC-based features
            for b in range(n_blocks):
                i1 = onsets[b]
                i2 = onsets[b+self.blocksize-1]
                x = resize_block(mfcc, i1, i2, self.mfccs_per_block)
                # Z-normalize
                x -= np.mean(x, 0)[None, :]
                xnorm = np.sqrt(np.sum(x**2, 1))[:, None]
                xnorm[xnorm == 0] = 1
                xn = x / xnorm
                block_feats['mfccs'][b, :] = xn.flatten()
                # Create SSM, resize, and save
                D = get_ssm(xn)
                block_feats['ssms'][b, :] = D[I < J] # Upper triangular part
            
            ## Step 2: Compute chroma blocks
            block_feats['chromas'] = np.zeros((n_blocks, self.chromas_per_block*chroma.shape[1]), dtype=np.float32)
            block_feats['chroma_med'] = np.median(chroma, axis=0)
            for b in range(n_blocks):
                i1 = onsets[b]
                i2 = onsets[b+self.blocksize]
                x = resize_block(chroma, i1, i2, self.chromas_per_block)
                block_feats['chromas'][b, :] = x.flatten()
            
            ## Step 3: Precompute Ws for each features
            """ Skip this since I'm doing a simpler, accelerated early fusion
            ssm_fns = {'chromas':lambda x: get_csm_cosine(x, x), 'mfccs':get_ssm, 'ssms':get_ssm}
            for feat in ssm_fns:
                d = ssm_fns[feat](block_feats[feat])
                block_feats['%s_W'%feat] = getW(d, self.K)
            """

            self.all_block_feats[i] = block_feats # Cache features
            dd.io.save(filepath, block_feats)


    def similarity(self, i, j, do_plot=False):
        feats1 = self.load_features(i)
        feats2 = self.load_features(j)
        ## Step 1: Create all of the parent SSMs
        Ws = {}
        scores = {}
        CSMs = {}
        tic = time.time()
        CSMs['mfccs'] = get_csm(feats1['mfccs'], feats2['mfccs'])
        M, N = CSMs['mfccs'].shape[0], CSMs['mfccs'].shape[1]
        D = np.zeros((M+1)*(N+1), dtype=np.float32)
        scores['mfccs'] = alignment_fn(csm_to_binary(CSMs['mfccs'], self.kappa).flatten(), D, M, N)
        CSMs['ssms'] = get_csm(feats1['ssms'], feats2['ssms'])
        D *= 0
        scores['ssms'] = alignment_fn(csm_to_binary(CSMs['ssms'], self.kappa).flatten(), D, M, N)
        CSMs['chromas'] = get_csm_blocked_oti(feats1['chromas'], feats2['chromas'], \
                                                    feats1['chroma_med'], feats2['chroma_med'],\
                                                    get_csm_cosine)
        D *= 0
        scores['chromas'] = alignment_fn(csm_to_binary(CSMs['chromas'], self.kappa).flatten(), D, M, N)

        ## Step 2: Compute Ws for each CSM
        W_CSMs = {s:getWCSM(CSMs[s], self.K, self.K) for s in CSMs}
        WCSM_sum = np.zeros_like(CSMs['mfccs'])
        for s in W_CSMs:
            WCSM_sum += W_CSMs[s]
        WCSM_sum = np.exp(-WCSM_sum) # Binary thresholding uses "distances" so switch back
        D *= 0
        scores['early'] = alignment_fn(csm_to_binary(WCSM_sum, self.kappa).flatten(), D, M, N)
        if do_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.imshow(csm_to_binary(WCSM_sum, self.kappa))
            plt.subplot(122)
            plt.imshow(np.reshape(D, (M+1, N+1)))
            plt.title("%.3g"%scores['early'])
            plt.show()
        if self.log_times:
            self.times['raw'].append(time.time()-tic)
        return scores

    def do_late_fusion(self):
        """
        Perform late fusion after all different pairwise similarity scores
        have been computed
        """
        self.Ds["late"] = doSimilarityFusion([1.0/(1.0+self.Ds[s]) for s in ["chromas", "ssms", "mfccs"]], K=20, niters=20, reg_diag=1)[1]
        self.Ds["early+late"] = doSimilarityFusion([1.0/(1.0+self.Ds[s]) for s in ["chromas", "ssms", "mfccs", "early"]], K=20, niters=20, reg_diag=1)[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmarking with Early Similarity Network Fusion of HPCP, MFCC, and MFCC SSMs",
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
    parser.add_argument("-l", '--log_times', type=int, choices=(0, 1), action="store", default=0,
                        help="Whether to log times to a file")
    parser.add_argument("-i", '--idx', type=int, action="store", default=0,
                        help="Index of pairs")

    cmd_args = parser.parse_args()

    from itertools import combinations
    import time
    import sys

    filename = 'cache/distances_batch/early_%s/%i.h5'%(cmd_args.shortname, cmd_args.idx)
    if os.path.exists(filename):
        print("Batch %i already done for %s; skipping..."%(cmd_args.idx, cmd_args.shortname))
        sys.exit(0)
    if not os.path.exists('cache/distances_batch'):
        os.mkdir('cache/distances_batch')
    if not os.path.exists('cache/distances_batch/early_%s'%cmd_args.shortname):
        os.mkdir('cache/distances_batch/early_%s'%cmd_args.shortname)

    ef = EarlyFusion(cmd_args.datapath, cmd_args.chroma_type, cmd_args.shortname, log_times=bool(cmd_args.log_times))
    N = len(ef.filepaths)
    blockdim = 100
    batch_size = blockdim**2
    blockres = int(N/blockdim) # Number of blocks across an axis
    NPairs = int(blockdim*blockdim*(blockres+blockres*(blockres-1)/2))

    ## Setup pairs
    if os.path.exists('pairs_map'):
        all_pairs = np.memmap('pairs_map', dtype=int, shape=(NPairs, 2), mode='r')
    else:
        all_pairs = np.memmap('pairs_map', dtype=int, shape=(NPairs, 2), mode='w+')
        I, J = np.meshgrid(np.arange(blockdim), np.arange(blockdim))
        blockidx = np.array([I.flatten(), J.flatten()]).T
        idx = 0
        for blocki in range(blockres):
            for blockj in range(blocki, blockres):
                all_pairs[idx*batch_size:(idx+1)*batch_size, :] = blockidx + np.array([[blocki*blockdim, blockj*blockdim]])
                idx += 1
    ## Run the appropriate batch
    tic = time.time()
    scores = {'mfccs':np.zeros((batch_size, 3), dtype=np.float32), \
              'ssms':np.zeros((batch_size, 3), dtype=np.float32), \
              'chromas':np.zeros((batch_size, 3), dtype=np.float32), \
              'early':np.zeros((batch_size, 3), dtype=np.float32)}
    for bidx, index in enumerate(range(cmd_args.idx*batch_size,(cmd_args.idx+1)*batch_size)):
        #print(all_pairs[index, 0], all_pairs[index, 1])
        i = all_pairs[index, 0]
        j = all_pairs[index, 1]
        res = ef.similarity(i, j)
        for s in res:
            scores[s][bidx, :] = [i, j, res[s]]
    print("Elapsed Time Batch %i: %.3g"%(cmd_args.idx, time.time()-tic))
    
    dd.io.save(filename, scores)
