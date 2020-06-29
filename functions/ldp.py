# Term Project: Inverse Laplace Transform of Real-valued Relaxation Data
# @icaizk, Fall 2014

# subroutine routine for ilt()


from __future__ import division
import numpy as np
import scipy.optimize


def ldp(G, h):
    """
    DISCRIPTION:
    -----------
        Solver for Least Distance Programming (LDP) with constraint.
    FORMULA:
    -------
        ||x|| = min, with constraint G*x >= h
    INPUT & OUTPUT:
    --------------
        Input matrix G and vector h describe constraint,
        Output is solution vector x.
    ----------------------------
    @Zhikun Cai, NPRE, UIUC
    ----------------------------
    REFERENCE:
    ---------
    Lawson, C., & Hanson, R. (1974), Solving Least Squares Problems, SIAM
    """

    # pre-processing
    m, n = G.shape
    if m != h.shape[0]:
        print('\nError in ldp(): input G and h have different dimensions!')

    # define matrix E = [G^T, h^T], vector f = [n zeros, 1]^T
    E = np.concatenate((G.T, h.reshape(1, m)))
    f = np.zeros(n+1)
    f[n] = 1.

    # solve for ||E*u -f|| = min, with constraint u >= 0
    u, resnorm = scipy.optimize.nnls(E, f)

    # compute residual vector r = E*u - f
    r = np.dot(E, u) - f

    # test criteria: if ||r|| = 0, the solution is incompatible with inequality;
    # otherwise, the computed solution is x_j = - r_j / r_n, j = 0, 1, ..., n-1.
    if np.linalg.norm(r) == 0:
        print('\nError in ldp(): solution is incompatible with inequality!')
    else:
        x = -r[0:-1]/r[-1]
    return x