"""
Programmer: Chris Tralie, 12/2016 (ctralie@alumni.princeton.edu)
Purpose: To implement similarity network fusion approach described in
[1] Wang, Bo, et al. "Unsupervised metric fusion by cross diffusion." Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012.
[2] Wang, Bo, et al. "Similarity network fusion for aggregating data types on a genomic scale." Nature methods 11.3 (2014): 333-337.
[3] Tralie, Christopher et. al. "Enhanced Hierarchical Music Structure Annotations via Feature Level Similarity Fusion." ICASSP 2019
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse
import scipy.io as sio
import time
import os
import librosa
import subprocess


def csm_binary(D, Kappa):
    """
    Turn a cross-similarity matrix into a binary cross-simlarity matrix
    If Kappa = 0, take all neighbors
    If Kappa < 1 it is the fraction of mutual neighbors to consider
    Otherwise Kappa is the number of mutual neighbors to consider
    """
    N = D.shape[0]
    M = D.shape[1]
    if Kappa == 0:
        return np.ones((N, M))
    elif Kappa < 1:
        NNeighbs = int(np.round(Kappa*M))
    else:
        NNeighbs = Kappa
    J = np.argpartition(D, NNeighbs, 1)[:, 0:NNeighbs]
    I = np.tile(np.arange(N)[:, None], (1, NNeighbs))
    V = np.ones(I.size)
    [I, J] = [I.flatten(), J.flatten()]
    ret = sparse.coo_matrix((V, (I, J)), shape=(N, M))
    return ret.toarray()

def csm_binary_mutual(D, Kappa):
    """
    Take the binary AND between the nearest neighbors in one direction
    and the other
    """
    B1 = csm_binary(D, Kappa)
    B2 = csm_binary(D.T, Kappa).T
    return B1*B2

def get_W(D, K, Mu = 0.5):
    """
    Return affinity matrix
    :param D: Self-similarity matrix
    :param K: Number of nearest neighbors
    :param Mu: Nearest neighbor hyperparameter (default 0.5)
    """
    #W(i, j) = exp(-Dij^2/(mu*epsij))
    DSym = 0.5*(D + D.T)
    np.fill_diagonal(DSym, 0)

    Neighbs = np.partition(DSym, K+1, 1)[:, 0:K+1]
    MeanDist = np.mean(Neighbs, 1)*float(K+1)/float(K) #Need this scaling
    #to exclude diagonal element in mean
    #Equation 1 in SNF paper [2] for estimating local neighborhood radii
    #by looking at k nearest neighbors, not including point itself
    Eps = MeanDist[:, None] + MeanDist[None, :] + DSym
    Eps = Eps/3
    Denom = (2*(Mu*Eps)**2)
    Denom[Denom == 0] = 1
    W = np.exp(-DSym**2/Denom)
    return W

def get_P(W, reg_diag = False):
    """
    Turn a similarity matrix into a proability matrix,
    with each row sum normalized to 1
    :param W: (MxM) Similarity matrix
    :param reg_diag: Whether or not to regularize
    the diagonal of this matrix
    :returns P: (MxM) Probability matrix
    """
    if reg_diag:
        P = 0.5*np.eye(W.shape[0])
        WNoDiag = np.array(W)
        np.fill_diagonal(WNoDiag, 0)
        RowSum = np.sum(WNoDiag, 1)
        RowSum[RowSum == 0] = 1
        P = P + 0.5*WNoDiag/RowSum[:, None]
        return P
    else:
        RowSum = np.sum(W, 1)
        RowSum[RowSum == 0] = 1
        P = W/RowSum[:, None]
        return P

def get_S(W, K):
    """
    Same thing as P but restricted to K nearest neighbors
        only (using partitions for fast nearest neighbor sets)
    (**note that nearest neighbors here include the element itself)
    :param W: (MxM) similarity matrix
    :param K: Number of neighbors to use per row
    :returns S: (MxM) S matrix
    """
    N = W.shape[0]
    J = np.argpartition(-W, K, 1)[:, 0:K]
    I = np.tile(np.arange(N)[:, None], (1, K))
    V = W[I.flatten(), J.flatten()]
    #Now figure out L1 norm of each row
    V = np.reshape(V, J.shape)
    SNorm = np.sum(V, 1)
    SNorm[SNorm == 0] = 1
    V = V/SNorm[:, None]
    [I, J, V] = [I.flatten(), J.flatten(), V.flatten()]
    S = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    return S

def fused_score(Pts):
    """
    Perform an average of a bunch of affinity matrices
    """
    FusedScores = np.zeros(Pts[0].shape)
    for Pt in Pts:
        FusedScores += Pt
    return FusedScores/len(Pts)

def avg_halfthresh(Pts):
    """
    Perform an average of a bunch of affinity matrices, and
    only keep the value if more than half of them are nonzero
    """
    FusedScores = np.zeros(Pts[0].shape)
    Counts = np.zeros(Pts[0].shape)
    for Pt in Pts:
        FusedScores += Pt
        Counts += (Pt > 0)
    res = FusedScores/len(Pts)
    res[Counts < len(Pts)/2] = 0
    return res, Counts

def snf_ws(Ws, K = 5, niters = 20, reg_diag = True, \
        do_animation = False, verbose_times = True):
    """
    Perform similarity fusion between a set of exponentially
    weighted similarity matrices
    :param Ws: An array of NxN affinity matrices for N songs
    :param K: Number of nearest neighbors
    :param niters: Number of iterations
    :param reg_diag: Identity matrix regularization parameter for
        self-similarity promotion
    :param reg_neighbs: Neighbor regularization parameter for promoting
        adjacencies in time
    :param do_animation: Save an animation of the cross-diffusion process
    :return D: A fused NxN similarity matrix
    """
    tic = time.time()
    #Full probability matrices
    Ps = [get_P(W, reg_diag) for W in Ws]
    #Nearest neighbor truncated matrices
    Ss = [get_S(W, K) for W in Ws]

    #Now do cross-diffusion iterations
    Pts = [np.array(P) for P in Ps]
    nextPts = [np.zeros(P.shape) for P in Pts]
    if verbose_times:
        print("Time getting Ss and Ps: %g"%(time.time() - tic))

    N = len(Pts)
    AllTimes = []
    if do_animation:
        plt.figure(figsize=(12, 6))
    for it in range(niters):
        ticiter = time.time()
        if do_animation:
            Im =fused_score(Pts)
            Im_Disp = np.array(Im)
            np.fill_diagonal(Im_Disp, 0)
            plt.subplot(121)
            plt.imshow(Im_Disp, interpolation = 'none', cmap = 'magma_r')
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(csm_binary_mutual(-Im, K))
            plt.savefig("SSMFusion%i.png"%it, dpi=300, bbox_inches='tight')
            plt.clf()
        for i in range(N):
            nextPts[i] *= 0
            tic = time.time()
            for k in range(N):
                if i == k:
                    continue
                nextPts[i] += Pts[k]
            nextPts[i] /= float(N-1)

            #Need S*P*S^T, but have to multiply sparse matrix on the left
            tic = time.time()
            A = Ss[i].dot(nextPts[i].T)
            nextPts[i] = Ss[i].dot(A.T)
            
            if reg_diag:
                P = 0.5*np.eye(nextPts[i].shape[0])
                PNoDiag = np.array(nextPts[i])
                np.fill_diagonal(PNoDiag, 0)
                RowSum = np.sum(PNoDiag, 1)
                RowSum[RowSum == 0] = 1
                P = P + 0.5*PNoDiag/RowSum[:, None]
                nextPts[i] = P

            toc = time.time()
            AllTimes.append(toc - tic)

        Pts = nextPts
        if verbose_times:
            print("Elapsed Time Iter %i of %i: %g"%(it+1, niters, time.time()-ticiter))
    if verbose_times:
        print("Total Time multiplying: %g"%np.sum(np.array(AllTimes)))
    return fused_score(Pts)

def snf(Scores, K = 5, niters = 20, reg_diag = True, \
        reg_neighbs = 0.5, do_animation = False):
    """
    Do similarity fusion on a set of NxN distance matrices.
    Parameters the same as snf_ws
    :returns (An array of similarity matrices for each feature, Fused Similarity Matrix)
    """
    #Affinity matrices
    Ws = [get_W(D, K) for D in Scores]
    return (Ws, snf_ws(Ws=Ws, K=K, niters=niters, reg_diag=reg_diag, do_animation=do_animation))


def doSNFSyntheticTest():
    np.random.seed(100)
    N = 200
    D = np.ones((N, N)) + 0.1*np.random.randn(N, N)
    D[D < 0] = 0
    I = np.arange(100)
    D[I, I] = 0

    I = np.zeros(40, dtype=np.int64)
    I[0:20] = 15 + np.arange(20)
    I[20::] = 50 + np.arange(20)
    J = I + 100
    D1 = 1.0*D
    D1[I, J] = 0

    I2 = np.arange(30, dtype=np.int64) + 20
    J2 = I2 + 60
    D2 = 1.0*D
    D2[I2, J2] = 0

    K = 5
    plt.subplot(121)
    plt.imshow(0.5*(D1+D2))
    plt.subplot(122)
    plt.imshow(csm_binary_mutual(0.5*(D1+D2), K))
    plt.show()

    snf([D1, D2], K = K, niters = 20, do_animation=True, reg_diag=True)

if __name__ == '__main__':
    doSNFSyntheticTest()