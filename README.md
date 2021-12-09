# Inverse Laplce Transform for Deep-Level Relaxation Spectroscopy

## Classic DLTS
The time-window concept proposed by D. Lang is the reason why classic DLTS is very sensitive to the small concentration of deep traps even with a low SNR(signal-to-noise ratio). But the main drawback of this approach is the inability to deconvolute the signal of two overlapped traps. 

## Laplace DLTS
So instead of setting the time windows and finding peaks of the plot for correct emission rates determination, regularisation is imposed. This gives an increase in trap separation ability but higher SNR is needed. Achieving higher SNR can be achieved by averaging many transients than SNR ~ sqrt(N), where is N – number of averaged transients.

### Used algorithms
Multiple algorithms are used to increase the reliability of results:
* python version of Contin – Fast and reliable algorithm. [Original code](https://github.com/caizkun/pyilt) was written by [caizkun](https://github.com/caizkun). 
* [FISTA](https://github.com/JeanKossaifi/FISTA) – Used to obtain sparse solutions. Works good when donor and acceptor traps present in transient. 
* pyReSpect – adapted algorithm from [shane5ul/pyReSpect-time](https://github.com/shane5ul/pyReSpect-time).
* regular L2 and L1+L2 regression.

(L-cure is used with Contin, L2, reSpect for regularisation parameter optimisation)

### Result

![](screenshot.png)

