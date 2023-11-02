def hp(t, C, T, ahp, Methods, Index, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, Nz, LCurve = False, Arrhenius = False):
    """Plots heatmap in ahp = [ahp1, ahp2] axes
    ahp1 for T vs Emission rates and
    ahp2 for Arrhenius or DLTS plots

    Parameters:
    -------------
    t : array 
        Time domain data from experiment
    C : 2D array (len(t), len(T))
        Contains transients for temperatures 1, 2, ...
        [F1(t), F2(t),...] from experiment
    T : array 
        Temperature from experiment
    ahp : list of matplotlib axes [ahp1, ahp2] 
        Axes to plot heatplot and Arrhenius
    Methods : list
        Method names used for plotting
    Index : int 
        Index to plot specific slice of heatplot
    Reg_L1, Reg_L2 : floats
        Reg. parameters for FISTA(L1) and L2 regularization
    Reg_C, Reg_S : floats
        Reg. parameters for CONTIN and reSpect algorithms
    Bounds : list
        [lowerbound, upperbound] of s domain points
    Nz : int
        Value which is length of calculated vector f(s)
    LCurve : boolean 
        Plot using L-curve criteria if True
    """

    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib import cm
    from matplotlib import gridspec
    
    from L1_FISTA import l1_fista
    from L2 import L2
    from L1L2 import L1L2
    from contin import Contin
    from reSpect import reSpect
    from regopt import regopt

    import sys

    def progressbar(i, iterations):
        """Prints simple progress bar"""
        i = i + 1
        sys.stdout.write("[%-20s] %d%%  Building Heatmap" % ('#'*np.ceil(i*100/iterations*0.2).astype('int'), 
            np.ceil(i*100/iterations))+'\r')
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
                TEMPE, TEMPX, a = l1_fista(t, C[i], Bounds, Nz, Reg_L1)
                XZ.append(TEMPE)
                ZZ.append(TEMPX)

                progressbar(i, cut)

        elif M == 'L2':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                TEMPE, TEMPX, a = L2(t, C[i], Bounds, Nz, Reg_L2)
                XZ.append(TEMPE)
                ZZ.append(TEMPX)

                progressbar(i, cut)

        elif M == 'L1+L2':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                TEMPE, TEMPX, a = L1L2(t, C[i], Bounds, Nz, Reg_L1, Reg_L2)
                XZ.append(TEMPE)
                ZZ.append(TEMPX)

                progressbar(i, cut)

        elif M == 'Contin':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                if LCurve:
                    ay = 0
                    Reg_C = regopt(t, C[i], ay, Methods, Reg_L1, Reg_L2, 
                                    Reg_C, Reg_S, Bounds, Nz, LCurve)
                TEMPE, TEMPX, a = Contin(t, C[i], Bounds, Nz, Reg_C)
                #print(YZ[-1][0], 'K; a = ', Reg_C)
                XZ.append(TEMPE)
                ZZ.append(TEMPX*TEMPE)

                progressbar(i, cut)

        elif M == 'reSpect':
            for i in range(0, cut):
                YZ.append(np.ones(cus)*T[i])
                if LCurve:
                    ay = 0
                    Reg_S = regopt(t, C[i], ay, Methods, Reg_L1, Reg_L2, 
                                    Reg_C, Reg_S, Bounds, Nz, LCurve)
                TEMPE, TEMPX, a = reSpect(t, C[i], Bounds, Nz, Reg_S)
                #print(YZ[-1][0], 'K; a = ', Reg_C)
                XZ.append(TEMPE)
                ZZ.append(TEMPX)

                progressbar(i, cut)



    XZ = np.asarray(XZ)
    YZ = np.asarray(YZ)
    ZZ = np.asarray(ZZ)

    ahp1, ahp2 = ahp[0], ahp[1]

    if Methods[0] == 'reSpect':
        v = np.abs(np.average(ZZ[10:-10,20:-20]))*20
        vmin, vmax = 0, v/2
        cmap = cm.jet
        levels = np.linspace(vmin, vmax, 20)
        #print(v)

    elif Methods[0] == 'Contin':
        v = np.abs(np.average(ZZ[10:-10,20:-20]))*10
        vmin, vmax = 0, v
        cmap = cm.jet
        levels = 20

    elif Methods[0] == 'FISTA' or Methods[0] == 'L2' or Methods[0] == 'L1+L2':
        v = np.abs(np.average(ZZ))*50
        vmin, vmax = -v, v
        cmap = cm.bwr
        levels = np.linspace(vmin, vmax, 20)

    #extent = [np.log10(Bounds[0]), np.log10(Bounds[1]), (T[-1]), (T[0])]

    x, y = np.meshgrid(TEMPE, T)

    ahp1.set_xlabel(r'Emission rate $e_{n,p}$, s')
    ahp1.set_title(Methods[0])
    ahp1.set_ylabel('Temperature T, K')
    ahp1.grid(True)
    #normalize = plt.Normalize(vmin = -v, vmax = v)

    heatmap = ahp1.contourf(x, y, ZZ, levels = levels,   cmap=cmap, corner_mask = False,
                            vmin = vmin, vmax = vmax, extend = 'both')
    plt.colorbar(heatmap)
    ahp1.set_xscale('log')

    if Arrhenius:

        ahp2.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useOffset = False)

        arrh = ahp2.contourf(1/y, np.log(x*y**-2), ZZ, levels = levels, cmap=cmap,
                             vmin = vmin, vmax = vmax, extend = 'both')
        #ahp2.set_xscale('log')
        ahp2.set_xlabel('Temperature $1/T$, $K^-1$')
        ahp2.set_ylabel('$\ln(e\cdot T^-2)$')

    else:
        ahp2.set_xlabel('Temperature, K')
        ahp2.set_ylabel('LDLTS signal, arb. units')
        for i in range(int(len(TEMPE)*0.1), int(len(TEMPE)*0.8), 20):
        #    ad.plot(T, ZZ[:, i], label=r'$\tau = %.3f s$'%(1/TEMPE[i]))
            ahp2.plot(T, ZZ[:, i]/np.amax(ZZ[:,i]), label=r'$\tau = %.3f s$'%(1/TEMPE[i]))
        #ahp2.set_yscale('log')
        #ahp2.set_ylim(1E-4, 10)
        ahp2.grid()
        ahp2.legend()

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

    np.savetxt('processed/NEW-FILE'%((t[1]-t[0])*1000) +'_1'+'.LDLTS', 
                Table, delimiter='\t', fmt = '%4E')
