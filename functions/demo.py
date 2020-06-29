def demo(Index, Nz, Reg_L1, Reg_L2, Bounds, Methods, Plot, Residuals, Heatplot):

    """ad"""
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
