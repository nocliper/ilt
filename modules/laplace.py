def laplace(t, F, Nz, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, Methods):

    """ Initiates routines for chosen method

    Parameters:
    -------------
    t : array 
        Time domain data from experiment
    F : array,
        Transient data from experiment F(t)
    Nz : int
        Number of points z to compute, must be smaller than len(F)
    Reg_L1, Reg_L2 : floats
        Reg. parameters for FISTA(L1) and L2 regularization
    Reg_C, Reg_S : floats
        Reg. parameters for CONTIN and reSpect algorithms
    Bounds : list
        [lowerbound, upperbound] of s domain points
    Methods : list 
        Names of processing methods

    Returns:
    -------------
    data : list of [[s, f, F_restored, Method1], ...]
        Data list for processed data

    """
    import numpy as np
    from L1_FISTA import l1_fista
    from L2 import L2
    from L1L2 import L1L2
    from contin import Contin
    from reSpect import reSpect

    data = []

    for i in Methods:
        if i == 'FISTA':
            s, f, F_hat = l1_fista(t, F, Bounds, Nz, Reg_L1)
            data.append([s, f, F_hat, 'FISTA'])

        elif i == 'L2':
            s, f, F_hat = L2(t, F, Bounds, Nz, Reg_L2)
            data.append([s, f, F_hat, 'L2'])

        elif i == 'L1+L2':
            s, f, F_hat = L1L2(t, F, Bounds, Nz, Reg_L1, Reg_L2)
            data.append([s, f, F_hat, 'L1+L2'])

        elif i == 'Contin':
            s, f, F_hat = Contin(t, F, Bounds, Nz, Reg_C)
            data.append([s, f, F_hat, 'Contin'])

        elif i == 'reSpect':
            s, f, F_hat = reSpect(t, F, Bounds, Nz, Reg_S)
            data.append([s, f, F_hat, 'reSpect'])

    data = np.asarray(data, dtype="object")
    return data
