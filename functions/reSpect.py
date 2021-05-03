#
# Help to find continuous spectrum
# March 2019 major update:
# (i)   added plateau modulus G0 (also in pyReSpect-time) calculation
# (ii)  following Hansen Bayesian interpretation of Tikhonov to extract p(lambda)
# (iii) simplifying lcurve (starting from high lambda to low)
# (iv)  changing definition of rho2 and eta2 (no longer dividing by 1/n and 1/nl)

from common import *

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

def getAmatrix(ns):
	"""Generate symmetric matrix A = L' * L required for error analysis:
	   helper function for lcurve in error determination"""
	# L is a ns*ns tridiagonal matrix with 1 -2 and 1 on its diagonal;
	nl = ns - 2
	L  = np.diag(np.ones(ns-1), 1) + np.diag(np.ones(ns-1),-1) + np.diag(-2. * np.ones(ns))
	L  = L[1:nl+1,:]

	return np.dot(L.T, L)

def getBmatrix(H, kernMat, Gexp, *argv):
	"""get the Bmatrix required for error analysis; helper for lcurve()
	   not explicitly accounting for G0 in Jr because otherwise I get underflow problems"""
	n   = kernMat.shape[0];
	ns  = kernMat.shape[1];
	nl  = ns - 2;
	r   = np.zeros(n);   	  # vector of size (n);

	# furnish relevant portion of Jacobian and residual

	Kmatrix = np.dot((1./Gexp).reshape(n,1), np.ones((1,ns)));
	Jr      = -kernelD(H, kernMat) * Kmatrix;

	# if plateau then unfurl G0
	if len(argv) > 0:
		G0 = argv[0]
		r  = (1. - kernel_prestore(H, kernMat, G0)/Gexp)
	else:
		r = (1. - kernel_prestore(H, kernMat)/Gexp)

	B = np.dot(Jr.T, Jr) + np.diag(np.dot(r.T, Jr))

	return B


def oldLamC(par, lam, rho, eta):

	#
	# 8/1/2018: Making newer strategy more accurate and robust: dividing by minimum rho/eta
	# which is not as sensitive to lam_min, lam_max. This makes lamC robust to range of lam explored
	#
	#er = rho/np.amin(rho) + eta/np.amin(eta);
	er    = rho/np.amin(rho) + eta/(np.sqrt(np.amax(eta)*np.amin(eta)));

	#
	# Since rho v/s lambda is smooth, we can interpolate the coarse mesh to find minimum
	#
	# change 3/20/2019: Scipy 0.17 has a bug with extrapolation: so making lami tad smaller
	lami = np.logspace(np.log10(min(lam)+1e-15), np.log10(max(lam)-1e-15), 1000)
	erri = np.exp(interp1d(np.log(lam), np.log(er), kind='cubic', bounds_error=False,
	                   fill_value=(np.log(er[0]), np.log(er[-1])))(np.log(lami)))


	ermin = np.amin(erri)
	eridx = np.argmin(erri)
	lamC  = lami[eridx]

	#
	# 2/2: Copying 12/18 edit from pyReSpect-time;
	#      for rough data have cutoff at rho = rho_cutoff?
	#
	rhoF  = interp1d(lam, rho)

	if  rhoF(lamC) <= par['rho_cutoff']:
		try:
			eridx = (np.abs(rhoF(lami) - par['rho_cutoff'])).argmin()
			if lami[eridx] > lamC:
				lamC = lami[eridx]
		except:
			pass

	return lamC


def lcurve(Gexp, Hgs, kernMat, par, *argv):

	"""
	 Function: lcurve(input)

	 Input: Gexp    = n*1 vector [Gt],
	        Hgs     = guessed H,
		    kernMat = matrix for faster kernel evaluation
	        par     = parameter dictionary
	        G0      = optionally

	 Output: lamC and 3 vectors of size npoints*1 contains a range of lambda, rho
	         and eta. "Elbow"  = lamC is estimated using a *NEW* heuristic AND by Hansen method


	March 2019: starting from large lambda to small cuts calculation time by a lot
				also gives an error estimate

	"""
	if par['plateau']:
		G0 = argv[0]


	npoints = int(par['lamDensity'] * (np.log10(par['lam_max']) - np.log10(par['lam_min'])))
	hlam    = (par['lam_max']/par['lam_min'])**(1./(npoints-1.))
	lam     = par['lam_min'] * hlam**np.arange(npoints)
	eta     = np.zeros(npoints)
	rho     = np.zeros(npoints)
	logP    = np.zeros(npoints)
	H       = Hgs.copy()
	n       = len(Gexp)
	ns      = len(H)
	nl      = ns - 2
	logPmax = -np.inf					# so nothing surprises me!
	Hlambda = np.zeros((ns, npoints))

	# Error Analysis: Furnish A_matrix
	Amat       = getAmatrix(len(H))
	_, LogDetN = np.linalg.slogdet(Amat)

	#
	# This is the costliest step
	#
	for i in reversed(range(len(lam))):

		lamb    = lam[i]

		if par['plateau']:
			H, G0   = getH(lamb, Gexp, H, kernMat, G0)
			rho[i]  = np.linalg.norm((1. - kernel_prestore(H, kernMat, G0)/Gexp))
			Bmat    = getBmatrix(H, kernMat, Gexp, G0)
		else:
			H       = getH(lamb, Gexp, H, kernMat)
			rho[i]  = np.linalg.norm((1. - kernel_prestore(H,kernMat)/Gexp))
			Bmat    = getBmatrix(H, kernMat, Gexp)

		eta[i]       = np.linalg.norm(np.diff(H, n=2))
		Hlambda[:,i] = H


		_, LogDetC = np.linalg.slogdet(lamb*Amat + Bmat)
		V          =  rho[i]**2 + lamb * eta[i]**2

		# this assumes a prior exp(-lam)
		logP[i]    = -V + 0.5 * (LogDetN + ns*np.log(lamb) - LogDetC) - lamb

		if(logP[i] > logPmax):
			logPmax = logP[i]
		elif(logP[i] < logPmax - 18):
			break

	# truncate all to significant lambda
	lam  = lam[i:]
	logP = logP[i:]
	eta  = eta[i:]
	rho  = rho[i:]
	logP = logP - max(logP)

	Hlambda = Hlambda[:,i:]

	#
	# currently using both schemes to get optimal lamC
	# new lamM works better with actual experimental data
	#
	lamC = oldLamC(par, lam, rho, eta)
	plam = np.exp(logP); plam = plam/np.sum(plam)
	lamM = np.exp(np.sum(plam*np.log(lam)))

	#
	# Dialling in the Smoothness Factor
	#
	if par['SmFacLam'] > 0:
		lamM = np.exp(np.log(lamM) + par['SmFacLam']*(max(np.log(lam)) - np.log(lamM)));
	elif par['SmFacLam'] < 0:
		lamM = np.exp(np.log(lamM) + par['SmFacLam']*(np.log(lamM) - min(np.log(lam))));

	#
	# printing this here for now because storing lamC for sometime only
	#

	return lamM, lam, rho, eta, logP, Hlambda


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

def guiFurnishGlobals(par):
	"""Furnish Globals to accelerate interactive plot in jupyter notebooks"""

	# plot settings
	from matplotlib import rcParams
	rcParams['axes.labelsize'] = 14
	rcParams['xtick.labelsize'] = 12
	rcParams['ytick.labelsize'] = 12
	rcParams['legend.fontsize'] = 12
	rcParams['lines.linewidth'] = 2

	# experimental data
	t, Gexp = GetExpData(par['GexpFile'])
	n    = len(t)
	ns   = par['ns']    # discretization of 'tau'

	tmin = t[0];
	tmax = t[n-1];

	# determine frequency window
	if par['FreqEnd'] == 1:
		smin = np.exp(-np.pi/2) * tmin; smax = np.exp(np.pi/2) * tmax
	elif par['FreqEnd'] == 2:
		smin = tmin; smax = tmax
	elif par['FreqEnd'] == 3:
		smin = np.exp(+np.pi/2) * tmin; smax = np.exp(-np.pi/2) * tmax

	hs   = (smax/smin)**(1./(ns-1))
	s    = smin * hs**np.arange(ns)
	kernMat = getKernMat(s,t)

	# toggle flags to prevent printing

	par['verbose'] = False
	par['plotting'] = False

	# load lamda, rho, eta
	lam, rho, eta = np.loadtxt('output/rho-eta.dat', unpack=True)


	return s, t, kernMat, Gexp, par, lam, rho, eta

def reSpect(t, Gexp, bound, Nz, lamC):
	"""
	This is the main driver routine for computing the continuous spectrum

	(*)   input  : "par" dictionary from "inp.dat" which specifies GexpFile (often 'Gt.dat')
	(*)   return : H and lambdaC; the latter can be used to microscpecify lambdaC as desired
	                without having to do the entire lcurve calculation again
	"""

	Gexp = Gexp/np.average(Gexp[0])
	Gexp = Gexp - np.average(Gexp[-1])
	Gexp = np.abs(Gexp)
	Gexp = Gexp + np.average(Gexp)*2

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

	return 1/s, np.exp(H), kernMat@np.exp(H)
