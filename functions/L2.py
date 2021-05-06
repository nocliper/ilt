def L2(s, Y, bound, Nz, alpha, iterations = 50000):
    """Returns solution vector, t-domain points and reconstructed transient
    using L1 regression and gradient descent.

    s - s-domain points, equally spased at log scale
    Y - given transient function
    bound – list of left and right bounds of s-domain points
    Nz – int value which is lenght of calculated vector
    alpha - reg. parameter for L2 regularisation
    iterations - number of iterations for gradient descent

    X  – Laplace transform Matrix

    returns:
    t – t-domain points
    beta - solution
    F – Reconstructed transient
    """
    import numpy as np
    from scipy.sparse import diags

    Y = Y/np.average(Y[0])
    Y = Y - np.average(Y[-1])
    Y = np.abs(Y)
    Y = Y + np.average(Y)*2
    Y = Y - Y[-1]




    tmin = bound[0]
    tlim = bound[1]
    NF   = len(s)
    Nf   = Nz #
    t    = tmin*10**(np.linspace(0, 40*np.log10(tlim/tmin), Nf)*0.025) #t domain with exp density points
    dt   = np.gradient(t)

    X    = np.zeros([NF, Nf], dtype = float)
    for i in range(NF-1):
            for j in range(Nf-1):
                x1     = -s[i]*(t[j] - dt[j])
                x2     = -s[i]*(t[j] + dt[j])
                X[i,j] = (np.exp(x1) + np.exp(x2))*dt[j]
    np.shape(X)

    l2     = alpha

    data = [-2*np.ones(Nz), 1*np.ones(Nz), 1*np.ones(Nz)]
    positions = [-1, -2, 0]

    I = diags(data, positions, (Nz+2, Nz)).toarray()
    #I      = np.identity(Nz)

    beta   = np.linalg.solve(l2*np.dot(I.T,I) + np.dot(X.T,X), np.dot(X.T,Y))

    F = X@beta

    return t, beta, F#, res_norm, sol_norm
