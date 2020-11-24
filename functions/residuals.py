def residuals(s, C, ay, Methods, Reg_L1, Reg_L2, Reg_SVD, Bounds, Nz):

    from laplace import laplace
    #from matplotlib.cm import jet
    import matplotlib.pyplot as plt
    import numpy as np

    def curvature(x, y):
        x1st = np.gradient(x)
        f1st = np.gradient(y)/x1st
        f2nd = np.gradient(y)/x1st

        k = np.abs(f2nd)/(np.sqrt(1+f1st**2))**3
        return k

    res = []
    sol = []

    alpha_L1  = 10**np.linspace(np.log10(Reg_L1)  - 2, np.log10(Reg_L1)  + 2, 150)
    alpha_L2  = 10**np.linspace(np.log10(Reg_L2)  - 2, np.log10(Reg_L2)  + 2, 150)
    alpha_SVD = 10**np.linspace(np.log10(Reg_SVD) - 2, np.log10(Reg_SVD) + 2, 150)

    for i in Methods:

        if len(Methods) > 1:
            print('!!!Choose only one Method!!!')
            break

        if i == 'L1':
            print('!!!Not ready!!!')

        elif i == 'L2':
            print('!!!not ready!!!')

        elif i == 'L1+L2':
            print('!!!not ready!!!')

        elif i == 'SVD':
            for j, v in enumerate(alpha_SVD):
                data = laplace(s, C - C[-1], Nz, Reg_L1, Reg_L2, v, Bounds, Methods)
                e, f, C_restored = data[0][0], data[0][1], data[0][2]

                res.append(np.linalg.norm(np.abs(C - C[-1]) - np.abs(C_restored), ord = 2)**2)
                sol.append(np.linalg.norm(f, ord = 2)**2)
            break


    k = curvature(np.log10(res), np.log10(sol))

    ay.plot(np.log10(res), np.log10(sol), 'k*-', )
    ay.set_ylabel(r'Solution norm $lg|f_{\alpha}|_2$', c='k')
    ay.set_xlabel(r'Residual norm $lg|C_0 - C_{\alpha}|_2$', c='k')



    ay_k = ay.twinx()
    ay_k_t = ay_k.twiny()
    ay_k_t.set_xscale('log')
    ay_k_t.plot(alpha_SVD, k, 'r-')
    ay_k_t.set_ylabel(r'Curvature', c='r')
    ay_k_t.set_xlabel(r'Reg. parameter $\lambda_{%.s}$'%(Methods[0]), c='r')

    ay_k_t.spines['top'].set_color('red')
    ay_k_t.spines['right'].set_color('red')
    ay_k_t.xaxis.label.set_color('red')
    ay_k_t.tick_params(axis='x', colors='red')
    ay_k.yaxis.label.set_color('red')
    ay_k.tick_params(axis='y', colors='red')

    plt.tight_layout()
