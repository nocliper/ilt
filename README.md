# Inverse the Laplace Transfom using L1 regression methdod

## The point

The point is to determine the decay rate <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\tau" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\tau" title="\tau" /></a> of the exponential function <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;e^{-\tau&space;s}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;e^{-\tau&space;s}" title="e^{-\tau s}" /></a> using numerical routines for Inverting the Laplace Transform. For exponential continuous function <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;F(s)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;F(s)" title="F(s)" /></a>, for <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\text{Re}(s)&space;>&space;0" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\text{Re}(s)&space;>&space;0" title="\text{Re}(s) > 0" /></a>:
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;F(s)&space;=&space;\sum_{i=0}^{n}&space;\exp{\left(-\tau_i&space;s\right)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;F(s)&space;=&space;\sum_{i=0}^{n}&space;\exp{\left(-\tau_i&space;s\right)}" title="F(s) = \sum_{i=0}^{n} \exp{\left(-\tau_i s\right)}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;f(t)&space;=&space;\mathcal{L}^{-1}\left\{F(s)\right&space;\}&space;=&space;\sum_{i=0}^{n}&space;\delta(t&space;-&space;\tau_i)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;f(t)&space;=&space;\mathcal{L}^{-1}\left\{F(s)\right&space;\}&space;=&space;\sum_{i=0}^{n}&space;\delta(t&space;-&space;\tau_i)" title="f(t) = \mathcal{L}^{-1}\left\{F(s)\right \} = \sum_{i=0}^{n} \delta(t - \tau_i)" /></a>

## Numerical Laplace Transform

For this, we replace (1) with finite-difference approximation (2) <a href="https://sci-hub.st/https://doi.org/10.1137/0730038">[1]</a>:

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;F(s)&space;=&space;\int_0^{\infty}&space;\exp{(-st)}f(t)dt\;(1);" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;F(s)&space;=&space;\int_0^{\infty}&space;\exp{(-st)}f(t)dt\;(1);" title="F(s) = \int_0^{\infty} \exp{(-st)}f(t)dt\;(1);" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathbf{Y}&space;=&space;\mathbf{X}\vec{\beta}&plus;\vec\epsilon\;\;\;\;\;(2)," target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{Y}&space;=&space;\mathbf{X}\vec{\beta}&plus;\vec\epsilon\;\;\;\;\;(2)," title="\mathbf{Y} = \mathbf{X}\vec{\beta}+\vec\epsilon\;\;\;\;\;(2)," /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathbf{X}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{X}" title="\mathbf{X}" /></a> - discrete approximation of Laplace transform;
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;$\vec\beta$" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;$\vec\beta$" title="$\vec\beta$" /></a> – elements of spectral function, with length <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;N_F" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;N_F" title="N_F" /></a>;
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathbf{Y}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{Y}" title="\mathbf{Y}" /></a> – given transient vector lenght <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;N_f" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;N_f" title="N_f" /></a>;
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\vec\epsilon" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\vec\epsilon" title="\vec\epsilon" /></a> – normal noise.

$$\begin{equation*}
\mathbf{X} = \left(
\begin{array}{cccc}
x_{11} & x_{12} & \ldots & x_{1(N_f-1)}\\
x_{21} & x_{22} & \ldots & x_{2(N_f-1)}\\
\vdots & \vdots & \ddots & \vdots\\
x_{(N_F-1)1} & x_{(N_F-1)2} & \ldots & x_{(N_F-1)(N_f-1)}
\end{array}
\right)
\end{equation*}$$

$$x_{ij} = \int_{s_j}^{s_{j+1}}\exp{\left(-s\cdot t_i\right)}ds$$

## The problem

Considering that noise is normal distributed we can maximize the posterior probability and hence minimize the residual sum of squares  

$$\mathbf{Y} = \mathbf{X}\vec{\beta}+\vec\epsilon$$

Since we will fitting the exponential decay with shifted delta function ( $\mathcal{L}\left\{\delta(t-\tau)\right\} = e^{-\tau s}$, for $\text{Re}(s)>0$ ), and in this model weights $\beta_i$ may take zero values. $L_1$ regression can gives us a sparse solution.

$$Q = \left(\mathbf{Y} - \mathbf{X}\vec{\beta}\right)^{T}\cdot\left(\mathbf{Y} - \mathbf{X}\vec{\beta}\right) + \lambda \vec\beta$$
$$\frac{\partial Q}{\partial \beta_i} = -2\mathbf{X}^T\mathbf{Y}+2\mathbf{X}^T\mathbf{X}\vec\beta +\lambda \text{sign}(\beta)$$

### Papers

1 <a href="https://sci-hub.st/https://doi.org/10.1137/0730038"> A REGULARIZATION METHOD FOR THE NUMERICAL INVERSION OF THE LAPLACE TRANSFORM. CHEN WEI DONG</a>

2 <a href="https://sci-hub.st/https://doi.org/10.1002/cmr.a.21263"> Laplace Inversion of Low-Resolution NMR Relaxometry Data Using Sparse Representation Methods. PAULA BERMAN ET AL.</a>
