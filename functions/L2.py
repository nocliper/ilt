"""Module to implement routines of numerical inverse 
Laplace tranform using L2 regularization algorithm
"""

import numpy as np
from scipy.sparse import diags

def L2(t, F, bound, Nz, alpha):
    """
    Returns solution for problem imposing L2 regularization

    F(t) = ∫f(s)*exp(-s*t)ds

    or

    min = ||C*f - F||2 + alpha*||I*f||2


    Parameters:
    ------------
    t : array of t (time domain data)
    F : array of F(t) (transient data)
    bound : [lowerbound, upperbound] of s domain points
    Nz : number of points s to compute, must be smaller than length(Y)
    alpha : egularization parameter for L2 regularization


    Returns:
    ------------
    s : s-domain points
    f : solution f(s)
    F : Reconstructed transient F(t) = C@f(s)
    """

    # set up grid points (# = Nz)
    h = np.log(bound[1]/bound[0])/(Nz - 1)      # equally spaced on logscale
    s = bound[0]*np.exp(np.arange(Nz)*h)        # z (Nz by 1)

    # construct C matrix from [1]
    s_mesh, t_mesh = np.meshgrid(s, t)
    C = np.exp(-t_mesh*s_mesh)       
    C[:, 0] /= 2.
    C[:, -1] /= 2.
    C *= h

    l2 = alpha

    data = [-2*np.ones(Nz), 1*np.ones(Nz), 1*np.ones(Nz)]
    positions = [-1, -2, 0]
    I = diags(data, positions, (Nz+2, Nz)).toarray()
    #I      = np.identity(Nz)

    f   = np.linalg.solve(l2*np.dot(I.T,I) + np.dot(C.T,C), np.dot(C.T,F))

    F_restored = C@f

    return s, f, F_restored#, res_norm, sol_norm
