#
# Global Imports and Plot Settings
#

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz, quad
from scipy.optimize import nnls, minimize, least_squares

import time
import os

#
# plotting preferences: change this block to suit your taste
#

#plt.style.use('ggplot')

#~ try:
	#~ import seaborn as sns
#~ except ImportError:
	#~ plt.style.use('ggplot')
#~ else:
	#~ plt.style.use('seaborn-ticks')
	#~ sns.set_color_codes()
	#~ sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

#~ from matplotlib import rcParams
#~ rcParams['axes.labelsize'] = 28
#~ rcParams['xtick.labelsize'] = 20
#~ rcParams['ytick.labelsize'] = 20
#~ rcParams['legend.fontsize'] = 20
#~ rcParams['lines.linewidth'] = 2

#
#
# Functions common to both discrete and continuous spectra
#
# def readInput(InpDataFileName)  : read input data to determine program settings
# def GetExpData(GtDataFileName)  : read experimental data
# def getKernMat(s, t) 			  : prestore Kernel Matrix, ... together with ...
# def kernel_prestore(H, kernMat) : ... greatly speeds evaluation of the kernel and overall speed
#

def readInput(fname='inp.dat'):
	"""Reads data from the input file (default = 'inp.dat')
	   and populates the parameter dictionary par"""

	par  = {}

	# read the input file
	for line in open(fname):

		li=line.strip()

		# if not empty or comment line; currently list or float
		if len(li) > 0 and not li.startswith("#" or " "):

			li = line.rstrip('\n').split(':')
			key = li[0].strip()
			tmp = li[1].strip()

			val = eval(tmp)

			par[key] = val

	# create output directory if none exists
	if not os.path.exists("output"):
		os.makedirs("output")

	return par

def GetExpData(fname):

	"""Function: GetExpData(input)
	   Reads in the experimental data from the input file
	   Input:  fname = name of file that contains G(t) in 2 columns [t Gt]
	   Output: A n*1 vector "t", and a n*1 vector Gt"""

	try:
		to, Gto = np.loadtxt(fname, unpack=True)  # 2 columns, t - Gt
	except OSError:
		print('*Error*: G(t) data file is either not in the correct path, or incorrectly formatted')
		quit()
	#
	# any repeated "time" values
	#

	to, indices = np.unique(to, return_index = True)
	Gto         = Gto[indices]

	#
	# Sanitize the input by spacing it out. Using linear interpolation
	#

	f  =  interp1d(to, Gto, fill_value="extrapolate")
	t  =  np.logspace(np.log10(np.min(to)), np.log10(np.max(to)), max(len(to),100))
	Gt =  f(t)


	return t, Gt

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
