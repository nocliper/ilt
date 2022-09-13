"""Module to implement routines of numerical inverse 
Laplace tranform using Contin  algorithm [1]

[1] Provencher, S. (1982)
"""

from __future__ import division
import numpy as np

def Contin(t, F, bound, Nz, alpha):
    """
    Function returns processed data f(z) from the equation
    
    F(t) = ∫f(z)*exp(-z*t)

    Parameters:
    --------------
    t : array of t (time domain data)
    F : array of F(t) (transient data)
    bound : [lowerbound, upperbound] of z domain points
    Nz : number of points z to compute, must be smaller than length(F)
    alpha : regularization parameter

    Returns:
    --------------
    z : array of (evenly spaced on log scale)
    f : array of f(z), inverse Laplace transform of F(t)
    F : array of C@f(z), reconstructed transient from f(z)
    """


    # set up grid points (# = Nz)
    h = np.log(bound[1]/bound[0])/(Nz - 1)      # equally spaced on logscale
    z = bound[0]*np.exp(np.arange(Nz)*h)        # z (Nz by 1)

    # construct C matrix from [1]
    z_mesh, t_mesh = np.meshgrid(z, t)
    C = np.exp(-t_mesh*z_mesh)       
    C[:, 0] /= 2.
    C[:, -1] /= 2.
    C *= h

    # construct regularization matrix R to impose gaussian-like peaks in f(z)
    # R - tridiagonal matrix (1,-2,1)
    Nreg = Nz + 2
    R = np.zeros([Nreg, Nz])
    R[0, 0] = 1.
    R[-1, -1] = 1.
    R[1:-1, :] = -2*np.diag(np.ones(Nz)) + np.diag(np.ones(Nz-1), 1) \
        + np.diag(np.ones(Nz-1), -1)

    #R = U*H*Z^T 
    U, H, Z = np.linalg.svd(R, full_matrices=False)     # H diagonal
    Z = Z.T
    H = np.diag(H)
    
    #C*Z*inv(H) = Q*S*W^T
    Hinv = np.diag(1.0/np.diag(H))
    Q, S, W = np.linalg.svd(C.dot(Z).dot(Hinv), full_matrices=False)  # S diag
    W = W.T
    S = np.diag(S)

    # construct GammaTilde & Stilde
    # ||GammaTilde - Stilde*f5||^2 = ||Xi||^2
    Gamma = np.dot(Q.T, F)
    Sdiag = np.diag(S)
    Salpha = np.sqrt(Sdiag**2 + alpha**2)
    GammaTilde = Gamma*Sdiag/Salpha
    Stilde = np.diag(Salpha)

    # construct LDP matrices G = Z*inv(H)*W*inv(Stilde), B = -G*GammaTilde
    # LDP: ||Xi||^2 = min, with constraint G*Xi >= B
    Stilde_inv = np.diag(1.0/np.diag(Stilde))
    G = Z @ Hinv @ W @ Stilde_inv
    B = -G @ GammaTilde

    # call LDP solver
    Xi = ldp(G, B)

    # final solution
    zf = np.dot(G, Xi + GammaTilde)
    f = zf/z

    F_restored = C@zf

    return z, f, F_restored


def ldp(G, h):
    """
    Helper for Contin() for solving NNLS [1]
    
    [1] - Lawson and Hanson’s (1974)

    Parameters:
    -------------
    G : Z*inv(H)*W*inv(Stilde)
    h : -G*GammaTilde

    Returns:
    -------------
    x : Solution of argmin_x || Ax - b ||_2 
    """

    from scipy.optimize import nnls

    m, n = G.shape
    A = np.concatenate((G.T, h.reshape(1, m)))
    b = np.zeros(n+1)
    b[n] = 1.

    # Solving for argmin_x || Ax - b ||_2 
    x, resnorm = nnls(A, b)

    r = A@x - b

    if np.linalg.norm(r) == 0:
        print('\n No solution found, try different input!')
    else:
        x = -r[0:-1]/r[-1]
    return x
