def L1(s, Y, bound, Nz, alpha, iterations = 50000):

    """Returns Inverse Laplase Transform of F(s) as Nz lenght vector
     using L1 regularization method  dQ/db = """

    import numpy as np

    tmin = bound[0]
    tlim = bound[1]
    NF   = len(s)
    Nf   = Nz #
    t    = tmin*10**(np.linspace(0, 40*np.log10(tlim/tmin), Nf)*0.025) #t domain with exp density points
    dt   = np.diff(t)

    X    = np.zeros([NF, Nf], dtype = float)
    for i in range(NF-1):
            for j in range(Nf-1):
                x1     = -s[i]*(t[j] - dt[j])
                x2     = -s[i]*(t[j] + dt[j])
                X[i,j] = (np.exp(x1) + np.exp(x2))*dt[j]
    np.shape(X)

    beta   = np.random.randn(Nf)/np.sqrt(Nf) ## initiating weights
    learning_rate = 0.09
    l1     = alpha
    #costs = []
    for k in range(iterations):
        Yhat  = X@beta
        delta = Yhat - Y
        beta  = beta - learning_rate*(X.T@delta + l1*np.sign(beta))
        #mse   = delta.dot(delta)/NF
        #costs.append(mse)
    F = X@beta
    #res_norm = np.linalg.norm(Y - X@beta)
    #sol_norm = np.linalg.norm(beta)

    return t, beta, F#, res_norm, sol_norm
