def laplace(s, F, Nz, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, Methods):

    ''' Initiates routines for choosed method

    s - s-domain points (time of transient)
    F - given transient function
    Nz – int value which is lenght of calculated vector
    Reg_L1, Reg_L2 - reg. parameter for L1 and L2 regularisation
    Bounds – list of left and right bounds of s-domain points
    Methods – list with methods to process data

    returns processed data

    '''
    import numpy as np
    from L1 import L1
    from L2 import L2
    from L1L2 import L1L2
    from ilt import Contin
    from reSpect import reSpect, InitializeH, getAmatrix, getBmatrix, oldLamC, getH, jacobianLM, kernelD, guiFurnishGlobals

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

        elif i == 'Contin':
            t, f, F_hat = Contin(s, F, Bounds, Nz, Reg_C)
            data.append([t, f, F_hat, 'Contin'])

        elif i == 'reSpect':
            t, f, F_hat = reSpect(s, F, Bounds, Nz, Reg_S)
            data.append([t, f, F_hat, 'reSpect'])

    data = np.asarray(data)
    return data
