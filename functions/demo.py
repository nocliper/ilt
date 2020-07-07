def demo(Index, Nz, Reg_L1, Reg_L2, Bounds, Methods, Plot, Residuals, Heatplot):
    """Gets data from interface() and display processed data

    Index     – int value contains an index of transient in dataset
    Nz        – int value which is lenght of calculated vector
    Reg_L1, Reg_L2 – reg. parameters for L1 and L2 regularisation
    Bounds    – list with left and right bound of t-domain
    Methods   – list with methods to process data
    Plot      – boolean which calls plot_data() if true
    Residuals – (not working yet)
    Hetplot   – plots heatplot for all dataset

    """
    import numpy as np
    from read_file import read_file
    from laplace import laplace
    from plot_data import plot_data
    from hp import hp
    from read_file import read_file

    Bounds = 10.0**np.asarray(Bounds)

    s, C, T = read_file('/Users/antonvasilev/GitHub/ilt/data/beta/EUNB29b_1-16-2_15_2.DLTS')
    cut = len(T)
    cus = len(C[0])

    data = laplace(s, C[Index] - C[Index][-1], Nz, Reg_L1, Reg_L2, Bounds, Methods)
    if Plot:
        plot_data(s, C[Index] - C[Index][-1], data, T, Index)
    if Residuals:
        print('Plotting L-curve...')
        print(Residuals)
    if Heatplot:
        print('Plotting Heatplot...')
        hp(s, C, T, Methods, Index, Reg_L1, Reg_L2, Bounds, Nz)
