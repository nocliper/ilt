# Term Project: Inverse Laplace Transform of Real-valued Relaxation Data
# @icaizk, Fall 2014

# Main routine for Laplace transform inversion
# call support routine: ldp()

from __future__ import division
import numpy as np
import ldp


def Contin(t, F, bound, Nz, alpha):
    """
    DISCRIPTION:
    -----------
    This code performs Inverse Laplace Transform by constructing Regularized
    NonNegative Least Squares (RNNLS) Problem [1], which is further converted
    and solved by Least Distance Programming [2].
    Note: This is a simplified and user-friendly version of CONTIN [1]
    to address Inverse Laplace Transform only.
    FORMULA OF LAPLACE TRANSFORM:
    ----------------------------
    F(t) = \int_lb^ub dz f(z) \exp(-z*t)
    INPUT:
    -----
    t: array of t
    F: array of F(t)
    bound: [lowerbound, upperbound] of above integral
    Nz: number of points z to compute, must be smaller than length(F)
    alpha: regularization parameter
    OUTPUT:
    ------
    z: array of z, equally spaced on LOG scale
    f: array of f(z), inverse Laplace transform of F(t)
    ----------------------------
    @Zhikun Cai, NPRE, UIUC
    ----------------------------
    REFERENCE:
    ---------
    [1] Provencher, S. (1982), A constrained regularization method for
        inverting data represented by linear algebraic or integral equations,
        Computer Physics Communications, 27, 213?227.
    [2] Lawson, C., & Hanson, R. (1974), Solving Least Squares Problems, SIAM
    """

    F = F - F[-1]
    F = np.abs(F)

    # pre-processing
    #if len(t) != len(F):
        #print('Error in ilt(): array t has different dimension from array F!')
    #if len(F) < Nz:
        #print('Error in ilt(): Nz is expected smaller than the dimension of F!')

    # set up grid points (# = Nz)
    h = np.log(bound[1]/bound[0])/(Nz - 1)      # equally spaced on logscale
    z = bound[0]*np.exp(np.arange(Nz)*h)        # z (Nz by 1)

    # construct coefficients matrix C by integral discretization (trapzoidal)
    # ||F - C*f||^2 = || F - \int_lb^ub [f(z)*z]*exp(-z*t) d(lnz) ||^2
    z_mesh, t_mesh = np.meshgrid(z, t)
    C = np.exp(-t_mesh*z_mesh)                   # specify integral kernel
    C[:, 0] /= 2.
    C[:, -1] /= 2.
    C *= h

    # construct regularizor matrix R to impose smoothness
    # || r - R*f ||^2 = || R*f ||^2 = \int_lb^ub [[z*f(z)]'']^2 d(lnz)
    Nreg = Nz + 2
    R = np.zeros([Nreg, Nz])
    R[0, 0] = 1.
    R[-1, -1] = 1.
    R[1:-1, :] = -2*np.diag(np.ones(Nz)) + np.diag(np.ones(Nz-1), 1) \
        + np.diag(np.ones(Nz-1), -1)

    # 1st SVD of R, R = U*H*Z^T
    U, H, Z = np.linalg.svd(R, full_matrices=False)     # H diagonal
    Z = Z.T
    H = np.diag(H)
    #print('\n-------------------------------')
    #print('1st SVD: rank(H) = %d' % np.linalg.matrix_rank(H))

    # 2nd SVD of C*Z*inv(H) = Q*S*W^T
    Hinv = np.diag(1.0/np.diag(H))
    Q, S, W = np.linalg.svd(C.dot(Z).dot(Hinv), full_matrices=False)  # S diag
    W = W.T
    S = np.diag(S)
    #print('2nd SVD: rank(S) = %d' % np.linalg.matrix_rank(S))

    # construct GammaTilde & Stilde
    # ||GammaTilde - Stilde*f5||^2 = ||Xi||^2
    Gamma = np.dot(Q.T, F)
    Sdiag = np.diag(S)
    Salpha = np.sqrt(Sdiag**2 + alpha**2)
    GammaTilde = Gamma*Sdiag/Salpha
    Stilde = np.diag(Salpha)
    #print('regularized: rank(Stilde) = %d' % np.linalg.matrix_rank(Stilde))
    #print('-------------------------------')

    # construct LDP matrices G = Z*inv(H)*W*inv(Stilde), B = -G*GammaTilde
    # LDP: ||Xi||^2 = min, with constraint G*Xi >= B
    Stilde_inv = np.diag(1.0/np.diag(Stilde))
    G = Z.dot(Hinv).dot(W).dot(Stilde_inv)
    B = -np.dot(G, GammaTilde)

    # call LDP solver
    Xi = ldp.ldp(G, B)

    # final solution
    zf = np.dot(G, Xi + GammaTilde)
    f = zf/z

    # residuals
    res_lsq = F - np.dot(C, zf)
    res_reg = np.dot(R, zf)

    F = C@zf

    return z, f, F
