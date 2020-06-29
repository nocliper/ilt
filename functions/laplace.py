def laplace(s, F, Nz, Reg_L1, Reg_L2, Bounds, Methods):

    ''' awd '''
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
            t, f, F_hat = SVD(s, F, Bounds, Nz, Reg_L1)
            data.append([t, f, F_hat, 'SVD'])

    data = np.asarray(data)
    return data
