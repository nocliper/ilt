def hp(s, C, T, Methods, Index, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, Nz, LCurve = False):
    """Returns heatmap

    s – s-domain points(time)
    C - transient F(s)x
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
    from L1_FISTA import l1_fista
    from L2 import L2
    from L1L2 import L1L2
    from ilt import Contin
    from reSpect import reSpect, InitializeH, getAmatrix, getBmatrix, oldLamC, getH, jacobianLM, kernelD, guiFurnishGlobals
    from residuals import residuals

    import sys

    def progressbar(i, iterations):
        i = i + 1
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("[%-20s] %d%%  Building Heatmap" % ('#'*np.ceil(i*100/iterations*0.2).astype('int'), np.ceil(i*100/iterations)))
        sys.stdout.flush()

    cut = len(T)
    cus = Nz

    if len(Methods) > 1:
        print('Choose only one Method')
        Methods = Methods[0]

    XZ = []
    YZ = []
    ZZ = []

    for M in Methods:
        if M == 'FISTA':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                TEMPE, TEMPX, a = l1_fista(s, C[i], Bounds, Nz, Reg_L1)
                XZ.append(TEMPE)
                ZZ.append(TEMPX*TEMPE)

                progressbar(i, cut)

        elif M == 'L2':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                TEMPE, TEMPX, a = L2(s, C[i], Bounds, Nz, Reg_L2)
                XZ.append(TEMPE)
                ZZ.append(TEMPX*TEMPE)

                progressbar(i, cut)

        elif M == 'L1+L2':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                TEMPE, TEMPX, a = L1L2(s, C[i], Bounds, Nz, Reg_L1, Reg_L2)
                XZ.append(TEMPE)
                ZZ.append(TEMPX*TEMPE)

                progressbar(i, cut)

        elif M == 'Contin':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                if LCurve:
                    ay = 0
                    Reg_C = residuals(s, C[i], ay, Methods, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, Nz, LCurve)
                TEMPE, TEMPX, a = Contin(s, C[i], Bounds, Nz, Reg_C)
                #print(YZ[-1][0], 'K; a = ', Reg_C)
                XZ.append(TEMPE)
                ZZ.append(TEMPX*TEMPE)

                progressbar(i, cut)

        elif M == 'reSpect':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                if LCurve:
                    ay = 0
                    Reg_S = residuals(s, C[i], ay, Methods, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, Nz, LCurve)
                TEMPE, TEMPX, a = reSpect(s, C[i], Bounds, Nz, Reg_S)
                #print(YZ[-1][0], 'K; a = ', Reg_C)
                XZ.append(TEMPE)
                ZZ.append(TEMPX)

                progressbar(i, cut)



    XZ = np.asarray(XZ)
    YZ = np.asarray(YZ)
    ZZ = np.asarray(ZZ)

    fig = plt.figure(figsize = (12,4.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    a2d = fig.add_subplot(121)


    if Methods[0] == 'reSpect':
        v = np.abs(np.average(ZZ[10:-10,5:-5]))*5
        vmin, vmax = 0, v
        cmap = cm.gnuplot2

    elif Methods[0] == 'Contin':
        v = np.abs(np.average(ZZ[10:-10,5:-5]))*10
        vmin, vmax = 0, v
        cmap = cm.gnuplot2

    elif Methods[0] == 'FISTA' or Methods[0] == 'L2' or Methods[0] == 'L1+L2':
        v = np.abs(np.average(ZZ))*5
        vmin, vmax = -v, v
        cmap = cm.bwr

    #extent = [np.log10(Bounds[0]), np.log10(Bounds[1]), (T[-1]), (T[0])]

    x, y = np.meshgrid(TEMPE, T)

    a2d.set_xlabel(r'Emission rate $e_{n,p}$, s')
    a2d.set_title(Methods[0])
    a2d.set_ylabel('Temperature T, K')
    a2d.grid(True)
    #normalize = plt.Normalize(vmin = -v, vmax = v)
    print('v ->', v)
    heatmap = a2d.contourf(x, y, ZZ, 200, cmap=cmap, vmin = vmin, vmax = vmax, extend = 'both')
    plt.colorbar(heatmap)
    a2d.set_xscale('log')

    ad = fig.add_subplot(122)
    ad.set_xlabel('Temperature, K')
    ad.set_ylabel('LDLTS signal, arb. units')
    for i in range(int(len(TEMPE)*0.1), int(len(TEMPE)*0.8), 20):
    #    ad.plot(T, ZZ[:, i], label=r'$\tau = %.3f s$'%(1/TEMPE[i]))
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
