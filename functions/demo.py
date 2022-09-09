def demo(Index, Nz, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, dt, Methods, Plot, Residuals, LCurve, Arrhenius, Heatplot):
    """Gets data from wigets initialized with interface(), 
    calls laplace() to process data and calls plot_data(), hp()
    to plot results

    Parameters:
    -------------
    Index : int value contains an index of transient in dataset
    Nz : int value which is lenght of calculated vector
    Reg_L1, Reg_L2 : reg. parameters for FISTA(L1) and L2 regularisation
    Reg_C, Reg_S : reg. parameters for CONTIN and reSpect algorithms
    Bounds : list with left and right bound of t-domain(emmision rates domain)
    dt : time step of transient data points
    Methods : list with methods to process data
    Plot : boolean which calls plot_data() if true
    Residuals : calls residuals() and plots L-curve to control regularization
    LCurve : boolean, picks optimal reg. parameter (used for automation)
    Hetplot : plots heatplot for all dataset and saves data in .LDLTS 
    """

    import numpy as np
    from read_file import read_file
    from laplace import laplace
    from plot_data import plot_data
    from hp import hp
    from read_file import read_file
    from regopt import regopt
    from interface import interface

    Bounds = 10.0**np.asarray(Bounds)

    t, C, T = read_file(interface.path, dt, proc = True)# time, transients, temperatures 
    
    data = laplace(t, C[Index], Nz, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, Methods)
    #print(data)
    
    if Plot:
        ay, aph = plot_data(t, C[Index], data, T, Index)
        
        if Heatplot:
            hp(t, C, T, aph, Methods, Index, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, Nz, LCurve, Arrhenius)
        
    if Residuals:
        regopt(t, C[Index], ay, Methods, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, Nz)
    