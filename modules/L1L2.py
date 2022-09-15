""" Module to implement L1 and/or L2 regularization with
SciPy linear_models module
"""

def L1L2(t, F, bound, Nz, alpha1, alpha2, iterations = 10000):
    """
    Returns solution using mixed L1 and/or L2 regularization
    with simple gradient descent

    F(t) = ∫f(s)*exp(-s*t)ds

    or

    min = ||C*f - F||2 + alpha1*||I*f||1 + alpha2*||I*f||2

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
    alpha1, alpha 2 : floats
        Regularization parameters for L1 and L2 regularizers
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

    import numpy as np
    from scipy.sparse import diags
    from sklearn.linear_model import ElasticNet
    
    # set up grid points (# = Nz)
    h = np.log(bound[1]/bound[0])/(Nz - 1)      # equally spaced on logscale
    s = bound[0]*np.exp(np.arange(Nz)*h)        # z (Nz by 1)

    # construct C matrix from [1]
    s_mesh, t_mesh = np.meshgrid(s, t)
    C = np.exp(-t_mesh*s_mesh)       
    C[:, 0] /= 2.
    C[:, -1] /= 2.
    C *= h
    
    alpha = alpha1 + alpha2
    l1_ratio = alpha1/alpha

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol = 1e-12,
                       fit_intercept = True, max_iter = iterations)
    model.fit(C, F)
    
    f = model.coef_

    F_restored = C@f + model.intercept_
    return s, f, F_restored