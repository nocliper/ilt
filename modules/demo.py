def demo(Index, Nz, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, dt, Methods, Plot, Residuals, LCurve, Arrhenius, Heatplot):
    """Gets data from widgets initialized with interface(), 
    calls laplace() to process data and calls plot_data(), hp()
    to plot results

    Parameters:
    -------------
    Index : int
        Index of transient in dataset
    Nz : int
        Value which is length of calculated vector
    Reg_L1, Reg_L2 : floats
        Reg. parameters for FISTA(L1) and L2 regularization
    Reg_C, Reg_S : floats
        Reg. parameters for CONTIN and reSpect algorithms
    Bounds : list
        [lowerbound, upperbound ] bounds of emission rates domain points
    dt : int
        Time step of transient data points in ms
    Methods : list 
        Methods to process data
    Plot : boolean
        Calls plot_data() if True
    Residuals : boolean
        Calls regopt() and plots L-curve to control 
        regularization if True
    LCurve : boolean, 
        Automatically picks optimal reg. parameter from 
        L-curve if True
    Heatplot : boolean
        Plots heatplot for all dataset and saves data 
        in .LDLTS if True
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
    