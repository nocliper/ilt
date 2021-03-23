def residuals(s, C, ay, Methods, Reg_L1, Reg_L2, Reg_SVD, Bounds, Nz):

    from laplace import laplace
    #from matplotlib.cm import jet
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import savgol_filter

    def curvature(x, y, a):
        '''Returns curvature of line

        k = (x'*y''-x''*y')/((x')^2+(y')^2)^(3/2)

        '''
        x = savgol_filter(x, 7, 1)
        y = savgol_filter(y, 7, 1)
        da = np.gradient(a)
        f_x  = np.gradient(x)/da
        f_y  = np.gradient(y)/da
        f_xx = np.gradient(f_x)/da
        f_yy = np.gradient(f_y)/da

        k = (f_x*f_yy - f_xx*f_y)/(f_x**2 + f_y**2)**(3/2)
        return k

    res = []
    sol = []

    alpha_L2  = 10**np.linspace(np.log10(Reg_L2)  - 3, np.log10(Reg_L2)  + 3, 200)
    alpha_SVD = 10**np.linspace(np.log10(Reg_SVD) - 3, np.log10(Reg_SVD) + 3, 200)
    alpha = alpha_SVD

    data = []

    for i in Methods:

        if len(Methods) > 1:
            print('!!!Choose only one Method!!!')
            break

        if i == 'L1':
            break

        elif i == 'L2':
            for j, v in enumerate(alpha_L2):
                data = laplace(s, C - C[-1], Nz, Reg_L1, v, Reg_SVD, Bounds, Methods)
                e, f, C_restored = data[0][0], data[0][1], data[0][2]

                res.append(np.linalg.norm(np.abs(C - C[-1]) - np.abs(C_restored), ord = 2)**2)
                sol.append(np.linalg.norm(f, ord = 2)**2)
            alpha = alpha_L2
            break

        elif i == 'L1+L2':
            break

        elif i == 'SVD':
            for j, v in enumerate(alpha_SVD):
                data = laplace(s, C - C[-1], Nz, Reg_L1, Reg_L2, v, Bounds, Methods)
                e, f, C_restored = data[0][0], data[0][1], data[0][2]

                res.append(np.linalg.norm(np.abs(C - C[-1]) - np.abs(C_restored), ord = 2)**2)
                sol.append(np.linalg.norm(f, ord = 2)**2)
            alpha = alpha_SVD
            break



    if len(data) == 0:
        ay.annotate(text = 'Choose L2 or SVD option', xy = (0.5,0.5), ha="center", size = 16)

    else:
        k = curvature(np.log10(res), np.log10(sol), alpha)
        k_max = np.amax(k)
        i = np.where(k == np.amax(k))
        i = np.squeeze(i)

        ay.plot(np.log10(res),    np.log10(sol),    'k-', )
        ay.plot(np.log10(res[i]), np.log10(sol[i]), 'r*') #highlight optimal lambda
        ay.set_ylabel(r'Solution norm $lg||x||^2_2$', c='k')
        ay.set_xlabel(r'Residual norm $lg||\eta-Cx||^2_2$', c='k')

        ay_k = ay.twinx()
        ay_k_t = ay_k.twiny()
        ay_k_t.set_xscale('log')
        ay_k_t.plot(alpha,    k,    'r-')
        ay_k_t.plot(alpha[i], k[i], 'r*')
        ay_k.set_ylabel(r'Curvature, arb. units', c='r')
        ay_k.set_ylim(-0.1, k_max*1.1)
        ay_k_t.set_xlabel(r'Reg. parameter $\lambda_{%.s}$'%(Methods[0]), c='r')

        ay_k_t.spines['top'].set_color('red')
        ay_k_t.spines['right'].set_color('red')
        ay_k_t.xaxis.label.set_color('red')
        ay_k_t.tick_params(axis='x', colors='red')
        ay_k.yaxis.label.set_color('red')
        ay_k.tick_params(axis='y', colors='red')

    plt.tight_layout()
