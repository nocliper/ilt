def laplace(s, F, Nz, Reg_L1, Reg_L2, Reg_SVD, Bounds, Methods):

    ''' Initiates routines for choosed method

    s - s-domain points, equally spased at log scale
    F - given transient function
    Nz        – int value which is lenght of calculated vector
    Reg_L1, Reg_L2 - reg. parameter for L1 and L2 regularisation
    Bounds – list of left and right bounds of s-domain points
    Methods   – list with methods to process data

    returns processed data

    '''
    import numpy as np
    from L1 import L1
    from L2 import L2
    from L1L2 import L1L2
    from ilt import SVD

    data = []

    for i in Methods:
        if i == 'L1':
            t, f, F_hat = L1(s, F, Bounds, Nz, Reg_L1)
            data.append([t, f, F_hat, 'L1'])

        elif i == 'L2':
            t, f, F_hat = L2(s, F, Bounds, Nz, Reg_L2)
            data.append([t, f, F_hat, 'L2'])

        elif i == 'L1+L2':
            t, f, F_hat = L1L2(s, F, Bounds, Nz, Reg_L1, Reg_L2)
            data.append([t, f, F_hat, 'L1+L2'])

        elif i == 'SVD':
            t, f, F_hat = SVD(s, F, Bounds, Nz, Reg_SVD)
            data.append([t, f, F_hat, 'SVD'])

    data = np.asarray(data)
    return data
