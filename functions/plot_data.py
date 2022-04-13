def plot_data(s, F, data, T, Index):
    """Plots data:

    s - s-domain points, equally spased at log scale
    F - given transient function of data with
    data – processed data
    T – Tempetarures
    Index – int value contains an index of transient in dataset

    """

    import matplotlib.pyplot as plt
    import numpy as np

    ## plotting main plot
    fig = plt.figure(constrained_layout=True, figsize = (9.5,11))
    widths  = [0.5, 0.5]
    heights = [0.3, 0.3, 0.4]
    spec = fig.add_gridspec(ncols=2, nrows=3, width_ratios=widths,
                              height_ratios=heights)


    ax  = fig.add_subplot(spec[0,:])
    ax.set_title(r'Temperature %.2f K'%T[Index])
    ax.set_ylabel(r'Amplitude, arb. units')
    ax.set_xlabel(r'Emission rate, $s^{-1}$')
    ax.set_xscale('log')
    ax.grid(True, which = "both", ls = "-")
    #print(data[:,2])
    for i, e in enumerate(data[:,-1]):
        if e == 'FISTA':
            ax.plot(data[i][0], data[i][1], 'r-', label = e)
            #np.savetxt('FISTA %.2fK.csv'%T[Index], [data[i][0], data[i][1]], delimiter = ',')
        elif e == 'L2':
            ax.plot(data[i][0], data[i][1], 'b-', label = e)
        elif e == 'L1+L2':
            ax.plot(data[i][0], data[i][1], 'm-', label = e)
        elif e == 'Contin':
            ax.plot(data[i][0],  data[i][1]*data[i][0], 'c-', label = e)
        elif e == 'reSpect':
            ax.plot(data[i][0],  data[i][1], 'y-', label = e)
            #np.savetxt('reSpect %.2fK.csv'%T[Index], [data[i][0], data[i][1]], delimiter = ',')
    ax.legend()

    ## plotting residuals
    ay = fig.add_subplot(spec[1, 0])
    ## plotting transients
    az = fig.add_subplot(spec[1, 1])
    az.set_ylabel(r'Transient , arb. units')
    az.set_xlabel(r'Time $t$, $s$')
    az.grid(True, which = "both", ls = "-")
    az.plot(s, F, 'ks-', label = 'Original')
    az.set_xscale('log')
    for i, e in enumerate(data[:,-1]):
        if e == 'FISTA':
            d = data[i][2]
            #d = np.abs(d)
            #d = d - min(d)
            #d = d/max(d)
            az.plot(s, d, 'ro-', label = e)
            #np.savetxt('Transients_'+e+' %.2fK.csv'%T[Index], [s, F, d], delimiter = ',')
        elif e == 'L2':
            d = data[i][2][:-1] # last point sucks
            #d = np.abs(d)
            #d = d - min(d)
            #d = d/max(d)
            az.plot(s[:-1], d, 'b>-', label = e)
        elif e == 'L1+L2':
            d = data[i][2]
            #d = np.abs(d)
            #d = d - min(d)
            #d = d/max(d)
            az.plot(s, d, 'm*-', label = e)
        elif e == 'Contin':
            d = data[i][2]
            #d = np.abs(d)
            #d = d - min(d)
            #d = d/max(d)
            az.plot(s, d, 'cx-', label = e)
        elif e == 'reSpect':
            d = data[i][2]
            #d = np.abs(d)
            #d = d - min(d)
            #d = d/max(d)
            az.plot(s, d - d[-1] + F[-1], 'y+-', label = e)
    az.legend()


    plt.tight_layout()

    ahp1, ahp2 = fig.add_subplot(spec[2, 0]), fig.add_subplot(spec[2, 1])

    return ay, [ahp1, ahp2]
