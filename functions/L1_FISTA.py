from fista import Fista
import numpy as np

def l1_fista(t, Y, bound, Nz, alpha, iterations = 50000, penalty = 'l11'):

    h = np.log(bound[1]/bound[0])/(Nz - 1)      # equally spaced on logscale
    z = bound[0]*np.exp(np.arange(Nz)*h)        # z (Nz by 1)

    # construct coefficients matrix C by integral discretization (trapzoidal)
    # ||F - K*f||^2 = || F - \int_lb^ub [f(z)*z]*exp(-z*t) d(lnz) ||^2
    z_mesh, t_mesh = np.meshgrid(z, t)
    K = np.exp(-t_mesh*z_mesh)                   # specify integral kernel
    K[:, 0] /= 2.
    K[:, -1] /= 2.
    K *= h

    fista = Fista(loss='least-square', penalty = penalty, lambda_=alpha*1e-4, n_iter = iterations)
    fista.fit(K, Y)

    return z, fista.coefs_, fista.predict(K)
