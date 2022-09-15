def plot_data(t, F, data, T, Index):
    """Gets data from demo() and plots it:

    Parameters:
    -------------
    t : array 
        Time domain data from experiment
    F : array,
        Transient data from experiment F(t)
    data : list of [[s, f, F_restored, Method1], ...]    
        Data list for processed data
    T : float 
        Temperature value of certain transient
    Index : int 
        Index of transient in initial dataset (not data)

    Returns:
    -------------
    ay : matplotlib axes 
        Axes for L-Curve plotting 
    [ahp1, ahp2] : list of matplotlib axes 
        Axes for its Arrhenuis or DLTS plots in hp()
    """

    import matplotlib.pyplot as plt
    import numpy as np

    ## Plotting main plot f(s)
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
        elif e == 'L2':
            ax.plot(data[i][0], data[i][1], 'b-', label = e)
        elif e == 'L1+L2':
            ax.plot(data[i][0], data[i][1], 'm-', label = e)
        elif e == 'Contin':
            ax.plot(data[i][0],  data[i][1]*data[i][0], 'c-', label = e)
        elif e == 'reSpect':
            ax.plot(data[i][0],  data[i][1], 'y-', label = e)
    ax.legend()

    # Axes for L-Curve
    ay = fig.add_subplot(spec[1, 0])

    # Plotting transients F(t)
    az = fig.add_subplot(spec[1, 1])
    az.set_ylabel(r'Transient , arb. units')
    az.set_xlabel(r'Time $t$, $s$')
    az.grid(True, which = "both", ls = "-")
    az.plot(t, F, 'ks-', label = 'Original')
    az.set_xscale('log')
    for i, e in enumerate(data[:,-1]):
        if e == 'FISTA':
            d = data[i][2]
            az.plot(t, d, 'ro-', label = e)
        elif e == 'L2':
            d = data[i][2][:-1] # last point sucks
            az.plot(t[:-1], d, 'b>-', label = e)
        elif e == 'L1+L2':
            d = data[i][2]
            az.plot(t, d, 'm*-', label = e)
        elif e == 'Contin':
            d = data[i][2]
            az.plot(t, d, 'cx-', label = e)
        elif e == 'reSpect':
            d = data[i][2]
            az.plot(t, d - d[-1] + F[-1], 'y+-', label = e)
    az.legend()


    plt.tight_layout()

    ahp1, ahp2 = fig.add_subplot(spec[2, 0]), fig.add_subplot(spec[2, 1])

    return ay, [ahp1, ahp2]
