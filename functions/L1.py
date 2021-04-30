def L1(s, Y, bound, Nz, alpha, iterations = 50000):

    """Returns solution vector, t-domain points and reconstructed transient
    using L1 regression and gradient descent.

    s - s-domain points (time points)
    Y - given transient function
    bound – list of left and right bounds of s-domain points
    Nz – int value which is lenght of calculated vector
    alpha - reg. parameter for L1 regularisation
    iterations - number of iterations for gradient descent

    X  – Laplace transform Matrix

    returns:
    t – t-domain points (emission rates)
    beta - solution
    F – Reconstructed transient
    """

    import numpy as np

    Y = Y/np.average(Y[0])
    Y = Y - np.average(Y[-1])
    Y = np.abs(Y)
    Y = Y - Y[-1]

    tmin = np.log10(bound[0])
    tlim = np.log10(bound[1])
    NF   = len(s)
    Nf   = Nz #
    t    = np.logspace(tmin, tlim, Nf) ##  evenly spaced on log grid
    #t    = tmin*10**(np.linspace(0, 40*np.log10(tlim/tmin), Nf)*0.025) #t domain with exp density points
    dt   = np.gradient(t)

    X    = np.zeros([NF, Nf], dtype = float)
    for i in range(NF-1):
            for j in range(Nf-1):
                x1     = -s[i]*(t[j] - dt[j])
                x2     = -s[i]*(t[j] + dt[j])
                X[i,j] = (np.exp(x1) + np.exp(x2))*dt[j]
    np.shape(X)

    beta = np.array(0, dtype = np.float64)
    beta   = np.random.randn(Nf)/np.sqrt(Nf) ## initiating weights
    learning_rate = 0.09
    l1     = alpha
    #costs = []
    Yhat  = np.array(0, dtype = np.float64)
    for k in range(iterations):
        Yhat  = np.dot(X,beta)
        delta = Yhat - Y
        beta  = beta - learning_rate*(X.T@delta + l1*np.sign(beta))
        #mse   = delta.dot(delta)/NF
        #costs.append(mse)
    F = X@beta
    #res_norm = np.linalg.norm(Y - X@beta)
    #sol_norm = np.linalg.norm(beta)

    return t, beta, F#, res_norm, sol_norm
