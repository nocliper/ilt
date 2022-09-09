
def residuals(t, F, ay, Methods, Reg_L1, Reg_L2, Reg_C, Reg_S, Bounds, Nz, LCurve = False):
    """ 
    Computes L-curve from residual and solution norm 
    and derives optimal regularization paramener from 
    its curvature plot. (max(k) -> a_opt)

    Parameters:
    -------------
    t : array of t (time domain points)
    C : 2D array lof Fi(t) = [F1(t), F2(t),...]
    ay : mpl axes to draw a plot
    Methods : list of processing methods
    Reg_L1, Reg_L2 : reg. parameters for FISTA(L1) and L2 regularisation
    Reg_C,  Reg_S  : reg. parameters for CONTIN and reSpect algorithms
    Bounds : list of left and right bounds of s-domain points
    Nz : number of points z to compute, must be smaller than length(F)
    L-curve : boolean if True residuals() returns optimal reg. parameter 
        for choosen method

    Returns:
    -------------
    alpha : float optimal reg. parameter for choosen method
        from L-curve curvature
    """
    from laplace import laplace
    #from matplotlib.cm import jet
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import savgol_filter

    import sys

    def progressbar(i, iterations):
        """Prints simple progress bar"""
        i = i + 1
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%  Building L-curve" % ('#'*np.ceil(i*100/iterations*0.2).astype('int'), np.ceil(i*100/iterations)))
        sys.stdout.flush()

    def curvature(x, y, a):
        """Returns curvature of line defined by:

        k = (x'*y''-x''*y')/((x')^2+(y')^2)^(3/2)

        Parameters:
        -------------
        x, y : arrays of x and y respactively
        a : array of reg parameters 

        Returns:
        -------------
        k : array of curvature values
        """
        x = savgol_filter(x, 13, 1)
        y = savgol_filter(y, 13, 1)
        da = np.gradient(a)
        f_x  = np.gradient(x)/da
        f_y  = np.gradient(y)/da
        f_xx = np.gradient(f_x)/da
        f_yy = np.gradient(f_y)/da

        k = (f_x*f_yy - f_xx*f_y)/(f_x**2 + f_y**2)**(3/2)
        return savgol_filter(k, 5, 1)
        #return k

    res = []  # residuals norm ||Cf - F||2
    sol = []  # solution norm ||f||2

    alpha_L2  = 10**np.linspace(np.log10(Reg_L2)  - 3, np.log10(Reg_L2)  + 3, 40)
    alpha_C   = 10**np.linspace(np.log10(Reg_C) - 3, np.log10(Reg_C) + 3, 40)
    alpha_S   = 10**np.linspace(np.log10(Reg_S) - 3, np.log10(Reg_S) + 3, 40)

    if LCurve:
        alpha_C = 10**np.linspace(np.log10(Reg_C) - 3, np.log10(Reg_C) + 3, 40)
    alpha = alpha_C

    data = []

    Fx = F

    for i in Methods:

        if len(Methods) > 1:
            print('!!!Choose only one Method!!!')
            break

        if i == 'FISTA':
            break

        elif i == 'L2':
            for j, v in enumerate(alpha_L2):
                data = laplace(t, F, Nz, Reg_L1, v, Reg_C, Reg_S, Bounds, Methods)
                e, f, F_restored = data[0][0], data[0][1], data[0][2]

                res.append(np.linalg.norm(np.abs(Fx) - np.abs(F_restored), ord = 2)**2)
                sol.append(np.linalg.norm(f, ord = 2)**2)
                progressbar(j, len(alpha_L2))
            alpha = alpha_L2
            break

        elif i == 'L1+L2':
            break

        elif i == 'Contin':
            for j, v in enumerate(alpha_C):
                data = laplace(t, F, Nz, Reg_L1, Reg_L2, v, Reg_S, Bounds, Methods)
                e, f, F_restored = data[0][0], data[0][1], data[0][2]

                res.append(np.linalg.norm(np.abs(Fx) - np.abs(F_restored), ord = 2)**2)
                #sol.append(np.linalg.norm(f, ord = 2)**2)
                sol.append(np.linalg.norm(f*e, ord = 2)**2)
                progressbar(j, len(alpha_C))
            alpha = alpha_C
            break

        elif i == 'reSpect':
            for j, v in enumerate(alpha_S):
                data = laplace(t, F, Nz, Reg_L1, Reg_L2, Reg_C, v, Bounds, Methods)
                e, f, F_restored = data[0][0], data[0][1], data[0][2]

                res.append(np.linalg.norm(np.abs(Fx - Fx[-1]) - np.abs(F_restored - F_restored[-1]), ord = 2)**2)
                sol.append(np.linalg.norm(f, ord = 2)**2)
                progressbar(j, len(alpha_S))
            alpha = alpha_S
            break

    # Plotting L-curve and its normalized curvature

    if len(data) == 0:
        ay.annotate(text = 'Choose only one method \n Contin, reSpect or L2', xy = (0.5,0.5), ha="center", size = 16)
        plt.tight_layout()
    elif LCurve:
        k = curvature(np.log10(res), np.log10(sol), alpha)
        k_max = np.amax(k)
        if Methods[0] == 'reSpect':
            i = np.where(k == np.amax(k[1:-1]))
        else:
            i = np.where(k == np.amax(k[1:-1]))
        i = np.squeeze(i)
        return alpha[i]
    else:
        k = curvature(np.log10(res), np.log10(sol), alpha)
        k_max = np.amax(k)
        i = np.where(k == np.amax(k))
        if Methods[0] != 'reSpect':
            i = np.where(k == np.amax(k[1:-1]))
        i = np.squeeze(i)

        ay.plot(np.log10(res),    np.log10(sol),    'k-', )

        ay.set_ylabel(r'Solution norm $\lg||x||^2_2$', c='k')
        ay.set_xlabel(r'Residual norm $\lg||\eta-Cx||^2_2$', c='k')

        ay_k = ay.twinx()
        ay_k_t = ay_k.twiny()
        ay_k_t.set_xscale('log')
        ay_k_t.plot(alpha,    k/k[i],    'r-')
        ay_k.set_ylabel(r'Curvature, arb. units', c='r')
        ay_k.set_ylim(-0.1, 1.1)
        #ay_k.set_yscale('log')
        #ay_k.set_ylim(1e-3, 2.0)
        ay_k_t.set_xlabel(r'Reg. parameter $\lambda_{%.s}$'%(Methods[0]), c='r')

        ay_k_t.spines['top'].set_color('red')
        ay_k_t.spines['right'].set_color('red')
        ay_k_t.xaxis.label.set_color('red')
        ay_k_t.tick_params(axis='x', colors='red', which='both')
        ay_k.yaxis.label.set_color('red')
        ay_k.tick_params(axis='y', colors='red', which='both')

        # Draw maximal curvature point of L-curve
        ay_k_t.plot(alpha[i], k[i]/k[i], 'r*')
        ay.plot(np.log10(res[i]), np.log10(sol[i]), 'r*') #highlight optimal lambda
        ay.annotate(r"$\lambda_\mathrm{opt} =$ %.1e"%alpha[i], c = 'k',
            xy = (np.log10(res[i])*0.98, np.log10(sol[i])*0.98), ha = 'left')

        plt.tight_layout()
