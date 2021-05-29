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

    F = F/np.average(F[0])
    if F[0] > F[-1]:
        F = F - min(F)
    else:
        F = F - max(F)
    F = np.abs(F)
    F = F + np.average(F)*1


    ## plotting main plot
    fig = plt.figure(figsize = (9.5, 6))
    ax  = fig.add_subplot(211)
    ax.set_title(r'Temperature %.2f K'%T[Index])
    ax.set_ylabel(r'Amplitude, arb. units')
    ax.set_xlabel(r'Emission rate, $s^{-1}$')
    ax.set_xscale('log')
    ax.grid(True, which = "both", ls = "-")
    #print(data[:,2])
    for i, e in enumerate(data[:,-1]):
        if e == 'L1':
            ax.plot(data[i][0], data[i][1], 'r-', label = e)
        elif e == 'L2':
            ax.plot(data[i][0], data[i][1], 'b-', label = e)
        elif e == 'L1+L2':
            ax.plot(data[i][0], data[i][1], 'm-', label = e)
        elif e == 'Contin':
            ax.plot(data[i][0],  data[i][1]*data[i][0], 'c-', label = e)
        elif e == 'reSpect':
            ax.plot(data[i][0],  data[i][1], 'y-', label = e)
    ax.legend()

    ## plotting residuals
    ay = fig.add_subplot(223)
    ## plotting transients
    az = fig.add_subplot(224)
    az.set_ylabel(r'Transient , arb. units')
    az.set_xlabel(r'Time $t$, $s$')
    az.grid(True, which = "both", ls = "-")
    az.plot(s, F - F[-1], 'ks-', label = 'Original')
    az.set_xscale('log')
    for i, e in enumerate(data[:,-1]):
        if e == 'L1':
            d = data[i][2]
            #d = np.abs(d)
            #d = d - min(d)
            #d = d/max(d)
            az.plot(s, d - d[-1], 'ro-', label = e)
        elif e == 'L2':
            d = data[i][2][:-1] # last point sucks
            #d = np.abs(d)
            #d = d - min(d)
            #d = d/max(d)
            az.plot(s[:-1], d - d[-1], 'b>-', label = e)
        elif e == 'L1+L2':
            d = data[i][2]
            #d = np.abs(d)
            #d = d - min(d)
            #d = d/max(d)
            az.plot(s, d - d[-1], 'm*-', label = e)
        elif e == 'Contin':
            d = data[i][2]
            #d = np.abs(d)
            #d = d - min(d)
            #d = d/max(d)
            az.plot(s, d - d[-1], 'cx-', label = e)
        elif e == 'reSpect':
            d = data[i][2]
            #d = np.abs(d)
            #d = d - min(d)
            #d = d/max(d)
            az.plot(s, d - d[-1], 'y+-', label = e)
    az.legend()


    plt.tight_layout()
    return ay
