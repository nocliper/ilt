#
# Help to find continuous spectrum
# March 2019 major update:
# (i)   added plateau modulus G0 (also in pyReSpect-time) calculation
# (ii)  following Hansen Bayesian interpretation of Tikhonov to extract p(lambda)
# (iii) simplifying lcurve (starting from high lambda to low)
# (iv)  changing definition of rho2 and eta2 (no longer dividing by 1/n and 1/nl)

import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz, quad
from scipy.optimize import nnls, minimize, least_squares

# HELPER FUNCTIONS

def InitializeH(Gexp, s, kernMat, *argv):
	"""
	Function: InitializeH(input)

	Input:  Gexp       = n*1 vector [Gt],
	           s       = relaxation modes,
			   kernMat = matrix for faster kernel evaluation
			   G0      = optional; if plateau is nonzero

	 Output:   H = guessed H
			  G0 = optional guess if *argv is nonempty
	"""
	#
	# To guess spectrum, pick a negative Hgs and a large value of lambda to get a
	# solution that is most determined by the regularization
	# March 2019; a single guess is good enough now, because going from large lambda to small
	#             lambda in lcurve.

	H    = -5.0 * np.ones(len(s)) + np.sin(np.pi * s)
	lam  = 1e0

	if len(argv) > 0:
		G0       = argv[0]
		Hlam, G0 = getH(lam, Gexp, H, kernMat, G0)
		return Hlam, G0
	else:
		Hlam     = getH(lam, Gexp, H, kernMat)
		return Hlam

def getH(lam, Gexp, H, kernMat, *argv):

	"""Purpose: Given a lambda, this function finds the H_lambda(s) that minimizes V(lambda)

	          V(lambda) := ||Gexp - kernel(H)||^2 +  lambda * ||L H||^2

	 Input  : lambda  = regularization parameter ,
	          Gexp    = experimental data,
	          H       = guessed H,
  		      kernMat = matrix for faster kernel evaluation
  		      G0      = optional

	 Output : H_lam, [G0]
	          Default uses Trust-Region Method with Jacobian supplied by jacobianLM
	"""

	# send Hplus = [H, G0], on return unpack H and G0
	if len(argv) > 0:
		Hplus= np.append(H, argv[0])
		res_lsq = least_squares(residualLM, Hplus, jac=jacobianLM, args=(lam, Gexp, kernMat))
		return res_lsq.x[:-1], res_lsq.x[-1]

	# send normal H, and collect optimized H back
	else:
		res_lsq = least_squares(residualLM, H, jac=jacobianLM, args=(lam, Gexp, kernMat))
		return res_lsq.x

def residualLM(H, lam, Gexp, kernMat):
	"""
	%
	% HELPER FUNCTION: Gets Residuals r
	 Input  : H       = guessed H,
			  lambda  = regularization parameter ,
	          Gexp    = experimental data,
  		      kernMat = matrix for faster kernel evaluation
  		      G0      = plateau

	 Output : a set of n+nl residuals,
	          the first n correspond to the kernel
	          the last  nl correspond to the smoothness criterion
	%"""


	n   = kernMat.shape[0];
	ns  = kernMat.shape[1];
	nl  = ns - 2;

	r   = np.zeros(n + nl);

	# if plateau then unfurl G0
	if len(H) > ns:
		G0     = H[-1]
		H      = H[:-1]
		r[0:n] = (1. - kernel_prestore(H, kernMat, G0)/Gexp)  # the Gt and
	else:
		r[0:n] = (1. - kernel_prestore(H, kernMat)/Gexp)  # the Gt and

	# the curvature constraint is not affected by G0
	r[n:n+nl] = np.sqrt(lam) * np.diff(H, n=2)  # second derivative


	return r

def jacobianLM(H, lam, Gexp, kernMat):
	"""
	HELPER FUNCTION for optimization: Get Jacobian J

	returns a (n+nl * ns) matrix Jr; (ns + 1) if G0 is also supplied.

	Jr_(i, j) = dr_i/dH_j

	It uses kernelD, which approximates dK_i/dH_j, where K is the kernel

	"""
	n   = kernMat.shape[0];
	ns  = kernMat.shape[1];
	nl  = ns - 2;

	# L is a ns*ns tridiagonal matrix with 1 -2 and 1 on its diagonal;
	L  = np.diag(np.ones(ns-1), 1) + np.diag(np.ones(ns-1),-1) + np.diag(-2. * np.ones(ns))
	L  = L[1:nl+1,:]

	# Furnish the Jacobian Jr (n+ns)*ns matrix
	Kmatrix         = np.dot((1./Gexp).reshape(n,1), np.ones((1,ns)));

	if len(H) > ns:

		G0     = H[-1]
		H      = H[:-1]

		Jr  = np.zeros((n + nl, ns+1))

		Jr[0:n, 0:ns]   = -kernelD(H, kernMat) * Kmatrix;
		Jr[0:n, ns]     = -1./Gexp							# column for dr_i/dG0

		Jr[n:n+nl,0:ns] = np.sqrt(lam) * L;
		Jr[n:n+nl, ns]  = np.zeros(nl)						# column for dr_i/dG0 = 0

	else:

		Jr  = np.zeros((n + nl, ns))

		Jr[0:n, 0:ns]   = -kernelD(H, kernMat) * Kmatrix;
		Jr[n:n+nl,0:ns] = np.sqrt(lam) * L;

	return	Jr

def kernelD(H, kernMat):
	"""
	 Function: kernelD(input)

	 outputs the (n*ns) dimensional matrix DK(H)(t)
	 It approximates dK_i/dH_j = K * e(H_j):

	 Input: H       = substituted CRS,
		    kernMat = matrix for faster kernel evaluation

	 Output: DK = Jacobian of H
	"""

	n   = kernMat.shape[0];
	ns  = kernMat.shape[1];


	# A n*ns matrix with all the rows = H'
	Hsuper  = np.dot(np.ones((n,1)), np.exp(H).reshape(1, ns))
	DK      = kernMat  * Hsuper

	return DK

def getKernMat(s, t):
	"""furnish kerMat() which helps faster kernel evaluation
	   given s, t generates hs * exp(-T/S) [n * ns matrix], where hs = wi = weights
	   for trapezoidal rule integration.

	   This matrix (K) times h = exp(H), Kh, is comparable with Gexp"""
	ns          = len(s)
	hsv         = np.zeros(ns);
	hsv[0]      = 0.5 * np.log(s[1]/s[0])
	hsv[ns-1]   = 0.5 * np.log(s[ns-1]/s[ns-2])
	hsv[1:ns-1] = 0.5 * (np.log(s[2:ns]) - np.log(s[0:ns-2]))
	S, T        = np.meshgrid(s, t);

	return np.exp(-T/S) * hsv;

def kernel_prestore(H, kernMat, *argv):
	"""
	     turbocharging kernel function evaluation by prestoring kernel matrix
		 Function: kernel_prestore(input) returns K*h, where h = exp(H)

		 Same as kernel, except prestoring hs, S, and T to improve speed 3x.

		 outputs the n*1 dimensional vector K(H)(t) which is comparable to Gexp = Gt

		 3/11/2019: returning Kh + G0

		 Input: H = substituted CRS,
		        kernMat = n*ns matrix [w * exp(-T/S)]

	"""

	if len(argv) > 0:
		G0 = argv[0]
	else:
		G0 = 0.

	return np.dot(kernMat, np.exp(H)) + G0


def reSpect(t, Gexp, bound, Nz, lamC):
	"""
	This is the main driver routine for computing the continuous spectrum

	(*)   input  : "par" dictionary from "inp.dat" which specifies GexpFile (often 'Gt.dat')
	(*)   return : H and lambdaC; the latter can be used to microscpecify lambdaC as desired
	                without having to do the entire lcurve calculation again
	"""

	n    = len(t)
	ns   = Nz    # discretization of 'tau'

	tmin = t[0];
	tmax = t[n-1];

	smin, smax = 1/bound[1], 1/bound[0] # s is tau domain points!!!!

	hs   = (smax/smin)**(1./(ns-1))
	s    = smin * hs**np.arange(ns)

	kernMat = getKernMat(s, t)

	Hgs, G0  = InitializeH(Gexp, s, kernMat, np.min(Gexp))

	lamC = lamC

	H, G0  = getH(lamC, Gexp, Hgs, kernMat, G0);

	K   = kernel_prestore(H, kernMat, G0);

	return 1/s[::-1], np.exp(H)[::-1], kernMat@np.exp(H)[::]
