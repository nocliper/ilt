def hp(s, C, T, Methods, Index, Reg_L1, Reg_L2, Reg_C, Bounds, Nz, LCurve = False):
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
    from matplotlib import gridspec
    from L1 import L1
    from L2 import L2
    from L1L2 import L1L2
    from ilt import Contin
    from reSpect import reSpect, InitializeH, getAmatrix, getBmatrix, oldLamC, getH, jacobianLM, kernelD, guiFurnishGlobals
    from residuals import residuals


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
                TEMPE, TEMPX, a = L1(s, C[i], Bounds, Nz, Reg_L1)
                XZ.append(TEMPE)
                ZZ.append(TEMPX*TEMPE)
        elif M == 'L2':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                TEMPE, TEMPX, a = L2(s, C[i], Bounds, Nz, Reg_L2)
                XZ.append(TEMPE)
                ZZ.append(TEMPX*TEMPE)
        elif M == 'L1+L2':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                TEMPE, TEMPX, a = L1L2(s, C[i], Bounds, Nz, Reg_L1, Reg_L2)
                XZ.append(TEMPE)
                ZZ.append(TEMPX*TEMPE)
        elif M == 'Contin':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                if LCurve:
                    ay = 0
                    Reg_C = residuals(s, C[i], ay, Methods, Reg_L1, Reg_L2, Reg_C, Bounds, Nz, LCurve)
                TEMPE, TEMPX, a = Contin(s, C[i], Bounds, Nz, Reg_C)
                #print(YZ[-1][0], 'K; a = ', Reg_C)
                print(TEMPE[0], '->', TEMPE[-1])
                XZ.append(TEMPE)
                ZZ.append(TEMPX*TEMPE)
        elif M == 'reSpect':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                if LCurve:
                    ay = 0
                    Reg_C = residuals(s, C[i], ay, Methods, Reg_L1, Reg_L2, Reg_C, Bounds, Nz, LCurve)
                TEMPE, TEMPX, a = reSpect(s, C[i], Bounds, Nz, Reg_C)
                #print(YZ[-1][0], 'K; a = ', Reg_C)
                print(TEMPE[0], '->', TEMPE[-1])
                XZ.append(TEMPE)
                ZZ.append(TEMPX)


    XZ = np.asarray(XZ)
    YZ = np.asarray(YZ)
    ZZ = np.asarray(ZZ)

    fig = plt.figure(figsize = (12,4.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    a2d = fig.add_subplot(121)

    cmap = cm.jet
    #v = np.amax(np.abs(ZZ))/20/1000
    v = 0.02
    normalize = plt.Normalize(vmin = -v, vmax = v)

    extent = [np.log10(Bounds[0]), np.log10(Bounds[1]), (T[-1]), (T[0])]
    a2d.set_xlabel(r'Emission $\log_{10}{(e)}$')
    a2d.set_title(Methods[0])
    a2d.set_ylabel('Temperature T, K')
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
    ad.set_xlabel('Temperature, K')
    ad.set_ylabel('LDLTS signal, arb. units')
    for i in range(int(len(TEMPE)*0.1), int(len(TEMPE)*0.8), 20):
        ad.plot(T, ZZ[:, i]/np.amax(ZZ[:,i]), label=r'$\tau = %.3f s$'%(1/TEMPE[i]))
    ad.set_yscale('log')
    ad.set_ylim(1E-4, 10)
    ad.grid()
    ad.legend()

    plt.show()
    plt.tight_layout()

    ##save file
    #Table = []
    #Table.append([0] + (1/TEMPE).tolist())
    #for i in range(cut):
    #    Table.append([T[i]] + (ZZ[i,:]).tolist())

    Table = []
    e = 1/TEMPE
    Table.append([0] + e.tolist())
    for i in range(cut):
        Table.append([T[i]] + (ZZ[i,:]).tolist())


    #print(Table)
    #Table = np.asarray(Table)

    np.savetxt('SAMPLE_NAME'%((s[1]-s[0])*1000) +'_1'+'.LDLTS', Table, delimiter='\t', fmt = '%4E')
