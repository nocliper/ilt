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
    t : array 
        Time domain data from experiment
    F : array,
        Transient data from experiment F(t)
    bound : list
        [lowerbound, upperbound] of s domain points
    Nz : int
        Number of points z to compute, must be smaller than len(F)
    alpha : float
        Regularization parameters for L2 regularizers
    iterations : int 
        Maximum number of iterations. Optional


    Returns:
    ------------
    s : array
        Emission rates domain points (evenly spaced on log scale)
    f : array
        Inverse Laplace transform f(s)
    F_restored : array 
        Reconstructed transient from C@f + intercept
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
