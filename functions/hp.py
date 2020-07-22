def hp(s, C, T, Methods, Index, Reg_L1, Reg_L2, Bounds, Nz):
    """Returns heatmap

    s – s-domain points(time)
    C - transient F(s) = C
    T - Tempetarures
    Methods – name of methods to process dataset
    Index – index to plot specific slise of heatplot
    Reg_L1, Reg_L2 – reg. parameters for L1 and L2 routines
    Bounds – list of left and right bounds of s-domain points
    Nz – int value which is lenght of calculated vector
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm
    from L1 import L1
    from L2 import L2
    from L1L2 import L1L2
    from ilt import SVD


    cut = len(T)
    cus = Nz

    if len(Methods) > 1:
        print('Choose only one Method')
        Methods = Methods[0]


    XZ = []
    YZ = []
    ZZ = []

    for M in Methods:
        if M == 'L1':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                TEMPE, TEMPX, a = L1(s, C[i] - C[i][-1], Bounds, Nz, Reg_L1)
                XZ.append(TEMPE)
                ZZ.append(TEMPX)
        elif M == 'L2':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                TEMPE, TEMPX, a = L2(s, C[i] - C[i][-1], Bounds, Nz, Reg_L2)
                XZ.append(TEMPE)
                ZZ.append(TEMPX)
        elif M == 'L1+L2':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                TEMPE, TEMPX, a = L1L2(s, C[i] - C[i][-1], Bounds, Nz, Reg_L1, Reg_L2)
                XZ.append(TEMPE)
                ZZ.append(TEMPX)
        elif M == 'SVD':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                TEMPE, TEMPX, a = SVD(s, C[i], Bounds, Nz, Reg_L1)
                XZ.append(TEMPE)
                ZZ.append(TEMPX)


    XZ = np.asarray(XZ)
    YZ = np.asarray(YZ)
    ZZ = np.asarray(ZZ)

    fig = plt.figure(figsize = (10,4.5))
    a2d = fig.add_subplot(121)

    cmap = cm.bwr
    v = np.amax(np.abs(ZZ))/50
    normalize = plt.Normalize(vmin = -v, vmax = v)

    extent = [np.log10(Bounds[0]), np.log10(Bounds[1]), (T[-1]), (T[0])]
    a2d.set_xlabel(r'Emission $\log_{10}{(e)}$')
    a2d.set_title(Methods)
    a2d.set_ylabel('Temperature T, K')
    a2d.axvline(x = np.log10(XZ[0][Index]), c = 'k')
    #a2d.grid(True)



    pos = a2d.imshow(ZZ[::1,::1], cmap = cmap,
               norm = normalize, interpolation = 'none',
               aspect = 'auto', extent = extent)
    plt.colorbar(pos)

    from matplotlib.ticker import FormatStrFormatter
    a2d.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.xticks(np.arange(np.log10(TEMPE[0]),np.log10(TEMPE[-1]), 0.9999))
    plt.yticks(np.arange(T[0], T[-1], 20.0))
    #plt.yticks(np.arange(T[0], T[-1], 20.0))

    ad = fig.add_subplot(122)
    ad.plot(T, ZZ[:,Index], c = 'k')
    #ad.legend()

    plt.show()
    plt.tight_layout()

    ##save file
    Table = []
    Table.append([0] + TEMPE.tolist())
    for i in range(cut):
        Table.append([T[i]] + ZZ[i,:].tolist())

    #print(Table)
    #Table = np.asarray(Table)

    #print(Table)
    print("shape XZ:",   XZ.shape, '; XZ[0][0] - XZ[-1][-1]: %.2f – %.2f'%(XZ[0][0], XZ[-1][-1]))
    print("shape YZ:",   YZ.shape, '; YZ[0][0] - YZ[-1][-1]: %.2f – %.2f'%(YZ[0][0], YZ[-1][-1]))
    print("shape ZZ:",   ZZ.shape, '; ZZ[0][0] - ZZ[-1][-1]: %.2f – %.2f'%(ZZ[0][0], ZZ[-1][-1]))
    print("shape e:", TEMPE.shape, '; e[0] - e[-1]: %.2f – %.2f'%(TEMPE[0], TEMPE[-1]))
    print("shape T:",     T.shape, '; T[0] – T[-1]: %.2f – %.2f'%(T[0], T[-1]))


    np.savetxt('DDMMYY.csv', Table, delimiter=',', fmt = '%4E')
