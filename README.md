# Inverse Laplce Transform for Deep-Level Transient Spectroscopy

This repository contains code for analysing DLTS and Laplace DLTS data using different regularisation approaches to produce reliable results. Arrhenius plot option available to plot temperature swept data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nocliper/ilt/blob/master/colab-ilt.ipynb)

## Classic DLTS
The time-window concept proposed by D. Lang is the reason why classic DLTS is very sensitive to the small concentration of deep traps even with a low SNR(signal-to-noise ratio). But the main drawback of this approach is the inability to deconvolute the signal of two overlapped traps. 

## Laplace DLTS
So instead of setting the time windows and searching for the peaks all over the plot for correct emission rates determination, regularisation is imposed. This gives an increase in trap separation ability but higher SNR is needed. Higher SNR can be achieved by averaging many transients (SNR ~ sqrt(N), where is N – number of averaged transients).

### Results
This notebook can be used to perform Laplace DLTS for `.DLTS` files in the data folder. Contin and pyReSpect will work in 99% of cases. L1, L1 + L2 and especially FISTA work well in the other 1% with different sign exponential decays [(example of FISTA accessing mobile ions in perovskites)](https://doi.org/10.1103/PhysRevApplied.13.034018). 

![](screenshot.png)
