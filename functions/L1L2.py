def L1L2(s, Y, bound, Nz, alpha1, alpha2, iterations = 50000):

    """Returns solution vector, t-domain points and reconstructed transient
    using L1 regression and gradient descent.

    s - s-domain points, equally spased at log scale
    Y - given transient function
    bound – list of left and right bounds of s-domain points
    Nz – int value which is lenght of calculated vector
    alpha1, alpha2 - reg. parameter for L1 and L2 regularisation
    iterations - number of iterations for gradient descent

    X  – Laplace transform Matrix

    returns:
    t – t-domain points
    beta - solution
    F – Reconstructed transient
    """

    import numpy as np

    Y = Y/np.average(Y[0])
    Y = Y - np.average(Y[-1])
    Y = np.abs(Y)
    Y = Y + np.average(Y)*2

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

    beta   = np.random.randn(Nf)/np.sqrt(Nf) ## initiating weights
    learning_rate = 0.09
    l1     = alpha1
    l2     = alpha2
    #costs = []
    for k in range(iterations):
        Yhat  = X@beta
        delta = Yhat - Y
        beta  = beta - learning_rate*(X.T@delta + l1*np.sign(beta) + l2*2*beta)
        mse   = delta.dot(delta)/NF
        #costs.append(mse)
    F = X@beta
    #res_norm = np.linalg.norm(Y - X@beta)
    #sol_norm = np.linalg.norm(beta)

    return t, beta, F#, res_norm, sol_norm
