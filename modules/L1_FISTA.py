from fista import Fista
import numpy as np

def l1_fista(t, F, bound, Nz, alpha, iterations = 50000, penalty = 'l11'):

    "Function to implement FISTA algorithm"

    h = np.log(bound[1]/bound[0])/(Nz - 1)      # equally spaced on logscale
    z = bound[0]*np.exp(np.arange(Nz)*h)        # z (Nz by 1)

    z_mesh, t_mesh = np.meshgrid(z, t)
    C = np.exp(-t_mesh*z_mesh)                   # specify integral kernel
    C[:, 0] /= 2.
    C[:, -1] /= 2.
    C *= h

    fista = Fista(loss='least-square', penalty = penalty, lambda_=alpha, n_iter = iterations)
    fista.fit(C, F)


    return z, fista.coefs_, fista.predict(C)
