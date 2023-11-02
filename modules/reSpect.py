import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz, quad
from scipy.optimize import nnls, minimize, least_squares

def Initialize_f(F, s, kernMat, *argv):
    """
    Computes initial guess for f and then call get_f()
    to return regularized solution:

    argmin_f = ||F - kernel(f)||2 +  lambda * ||L f||2

    Parameters:
    ------------
    F : array
        Transient experimental data
    s : array
        Tau-domain points
    kernMat : matrix (len(s), len(F))
        Matrix of inverse Laplace transform
    F_b : float
        Baseline value, optional

    Returns:
    -----------
    flam : array
        Regularized inverse of Laplace transform
    F_b : float
        Baseline value
    """

    f    = -5.0 * np.ones(len(s)) + np.sin(np.pi * s)  # initial guess for f
    lam  = 1e0

    if len(argv) > 0:
        F_b       = argv[0]
        flam, F_b = get_f(lam, F, f, kernMat, F_b)
        return flam, F_b
    else:
        flam     = get_f(lam, F, f, kernMat)
        return flam

def get_f(lam, F, f, kernMat, *argv):
    """
    Solves following equation for f. Uses jacobianLM() with 
    scipy.optimize.least_squares() solver:

    argmin_f = ||F - kernel(f)||2 +  lambda * ||L f||2

    Parameters:
    -------------
    lambda : float
        Regularization parameter
    F : array
        Transient experimental data
    f : array
        Guessed f(s) for emission rates
    kernMat : matrix (len(f), len(F))
        Matrix of inverse Laplace transform
    F_b : float
        Baseline value,optional

    Returns:
    -------------
    flam : array
        Regularized solution
    F_b : float
        Baseline for solution
    """

    # send fplus = [f, F_b], on return unpack f and F_b
    if len(argv) > 0:
        fplus= np.append(f, argv[0])
        res_lsq = least_squares(residualLM, fplus, jac=jacobianLM, args=(lam, F, kernMat))
        return res_lsq.x[:-1], res_lsq.x[-1]

    # send normal f, and collect optimized f back
    else:
        res_lsq = least_squares(residualLM, f, jac=jacobianLM, args=(lam, F, kernMat))
        return res_lsq.x

def residualLM(f, lam, F, kernMat):
    """
    Computes residuals for below equation 
    and used with scipy.optimize.least_squares():

    argmin_f = ||F - kernel(f)||2 +  lambda * ||L f||2

    Parameters:
    -------------
    f : array
        Solution for above equation f(s)
    lambda : float
        Regularization parameter
    F : array
        Experimental transient data
    kernMat : matrix (len(f), len(F))
        Matrix of inverse Laplace transform
    F_b : float
        Plateau value for F experimental data

    Returns:
    -----------
    r : array 
        Residuals (||F - kernel(f)||2)''
    """

    n   = kernMat.shape[0];
    ns  = kernMat.shape[1];
    nl  = ns - 2;

    r   = np.zeros(n + nl);

    # if plateau then unfurl F_b
    if len(f) > ns:
        F_b    = f[-1]
        f      = f[:-1]
        r[0:n] = (1. - kernel_prestore(f, kernMat, F_b)/F)  # same as |F - kernel(f)|
    else:
        r[0:n] = (1. - kernel_prestore(f, kernMat)/F) # same as |F - kernel(f)| w/o F_b

    r[n:n+nl] = np.sqrt(lam) * np.diff(f, n=2)  # second derivative

    return r

def jacobianLM(f, lam, F, kernMat):
    """
    Computes jacobian for scipy.optimize.least_squares()

    argmin_f = ||F - kernel(f)||2 +  lambda * ||L f||2

    Parameters:
    -------------
    f : array
        Solution for above equation
    lam : float
        Regularization parameter
    F : array
        Experimental transient data
    kernMat : matrix (len(f), len(F))
        Matrix of inverse Laplace transform

    Returns:
    ------------
    Jr : matrix (len(f)*2 - 2, len(F) + 1)
        Contains Jr(i, j) = dr_i/df_j
    """

    n   = kernMat.shape[0];
    ns  = kernMat.shape[1];
    nl  = ns - 2;

    # L is a ns*ns tridiagonal matrix with 1 -2 and 1 on its diagonal;
    L  = np.diag(np.ones(ns-1), 1) + np.diag(np.ones(ns-1),-1) + np.diag(-2. * np.ones(ns))
    L  = L[1:nl+1,:]

    # Furnish the Jacobian Jr (n+ns)*ns matrix
    Kmatrix         = np.dot((1./F).reshape(n,1), np.ones((1,ns)));

    if len(f) > ns:

        F_b    = f[-1]
        f      = f[:-1]

        Jr  = np.zeros((n + nl, ns+1))

        Jr[0:n, 0:ns]   = -kernelD(f, kernMat) * Kmatrix;
        Jr[0:n, ns]     = -1./F                          # column for dr_i/dF_b

        Jr[n:n+nl,0:ns] = np.sqrt(lam) * L;
        Jr[n:n+nl, ns]  = np.zeros(nl)                      # column for dr_i/dF_b = 0

    else:

        Jr  = np.zeros((n + nl, ns))

        Jr[0:n, 0:ns]   = -kernelD(f, kernMat) * Kmatrix;
        Jr[n:n+nl,0:ns] = np.sqrt(lam) * L;

    return Jr

def kernelD(f, kernMat):
    """
    Helper for jacobianLM() approximates dK_i/df_j = K * e(f_j)

    argmin_f = ||F - kernel(f)||2 +  lambda * ||L f||2

    Parameters:
    -------------
    f : array
        Solution for above equation
    kernMat : matrix (len(f), len(F))
        Matrix of inverse Laplace transform

    Returns:
    -------------
    DK : matrix (len(f), len(F)) 
        Jacobian
    """

    n   = kernMat.shape[0];
    ns  = kernMat.shape[1];

    # A n*ns matrix with all the rows = f'
    fsuper  = np.dot(np.ones((n,1)), np.exp(f).reshape(1, ns))
    DK      = kernMat  * fsuper

    return DK

def getKernMat(s, t):
    """
    Mesh grid for s and t domain to construct 
    kernel matrix

    Parameters:
    -------------
    s: array
        Tau domain points
    t: array
        Time domain points from experiment

    Returns:
    -------------
    np.exp(-T/S) * hsv : matrix (len(s), len(t))
        Matrix of inverse Laplace transform, where hsv 
        trapezoidal coefficients
    """
    ns          = len(s)
    hsv         = np.zeros(ns);
    hsv[0]      = 0.5 * np.log(s[1]/s[0])
    hsv[ns-1]   = 0.5 * np.log(s[ns-1]/s[ns-2])
    hsv[1:ns-1] = 0.5 * (np.log(s[2:ns]) - np.log(s[0:ns-2]))
    S, T        = np.meshgrid(s, t);

    return np.exp(-T/S) * hsv;

def kernel_prestore(f, kernMat, *argv):
    """
    Function for prestoring kernel

    argmin_f = ||F - kernel(f)||2 +  lambda * ||L f||2

    Parameters:
    -------------
    f : array
        Solution of above equation
    kernMat : matrix (len(f), len(F))
        Matrix of inverse Laplace transform

    Returns:
    ------------
    np.dot(kernMat, np.exp(f)) + F_b : 
        Stores kernMat*(f)+ F_b 
    """

    if len(argv) > 0:
        F_b = argv[0]
    else:
        F_b = 0.

    return np.dot(kernMat, np.exp(f)) + F_b


def reSpect(t, F, bound, Nz, alpha):
    """
    Main routine to implement reSpect algorithm from [1].

    [1] Shanbhag, S. (2019)

    Parameters:
    --------------
    t : array
        Time domain points from experiment
    F : array
        Experimental transient F(t)
    bound : list 
        [lowerbound, upperbound] of bounds for tau-domain points
    Nz : int
        Length of tau-domain array
    alpha : float
        Regularization parameter

    Returns:
    --------------
    1/s[::-1] : array
        Tau-domain points
    np.exp(f)[::-1] : array
        Inverse Laplace transform of F(t)
    kernMat@np.exp(f)[::] : array
        Restored transient F_restored(t)
    """

    n    = len(t)
    ns   = Nz    # discretization of 'tau'

    tmin = t[0];
    tmax = t[n-1];

    smin, smax = 1/bound[1], 1/bound[0]  # s is tau domain points!

    hs   = (smax/smin)**(1./(ns-1))
    s    = smin * hs**np.arange(ns)  # s here is tau domain points

    kernMat = getKernMat(s, t)

    fgs, F_b  = Initialize_f(F, s, kernMat, np.min(F))

    alpha = alpha

    f, F_b  = get_f(alpha, F, fgs, kernMat, F_b);

    K   = kernel_prestore(f, kernMat, F_b);

    return 1/s[::-1], np.exp(f)[::-1], kernMat@np.exp(f)[::]
