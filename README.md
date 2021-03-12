# Inverse the Laplace Transfom using L1 and L2 regression methods

## The point

The point is to determine the decay rate <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\tau" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\tau" title="\tau" /></a> of the exponential function <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;e^{-\tau&space;s}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;e^{-\tau&space;s}" title="e^{-\tau s}" /></a> using numerical routines for Inverting the Laplace Transform. For exponential continuous function <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;F(s)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;F(s)" title="F(s)" /></a>, for <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\text{Re}(s)&space;>&space;0" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\text{Re}(s)&space;>&space;0" title="\text{Re}(s) > 0" /></a>:
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;F(s)&space;=&space;\sum_{i=0}^{n}&space;\exp{\left(-\tau_i&space;s\right)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;F(s)&space;=&space;\sum_{i=0}^{n}&space;\exp{\left(-\tau_i&space;s\right)}" title="F(s) = \sum_{i=0}^{n} \exp{\left(-\tau_i s\right)}" /></a>

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;f(t)&space;=&space;\mathcal{L}^{-1}\left\{F(s)\right&space;\}&space;=&space;\sum_{i=0}^{n}&space;\delta(t&space;-&space;\tau_i)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;f(t)&space;=&space;\mathcal{L}^{-1}\left\{F(s)\right&space;\}&space;=&space;\sum_{i=0}^{n}&space;\delta(t&space;-&space;\tau_i)" title="f(t) = \mathcal{L}^{-1}\left\{F(s)\right \} = \sum_{i=0}^{n} \delta(t - \tau_i)" /></a>
</p>

## Numerical Laplace Transform

For this, we replace (1) with finite-difference approximation (2) <a href="https://sci-hub.st/https://doi.org/10.1137/0730038">[1]</a>:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;F(s)&space;=&space;\int_0^{\infty}&space;\exp{(-st)}f(t)dt\;(1);" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;F(s)&space;=&space;\int_0^{\infty}&space;\exp{(-st)}f(t)dt\;(1);" title="F(s) = \int_0^{\infty} \exp{(-st)}f(t)dt\;(1);" /></a>
</p>

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathbf{Y}&space;=&space;\mathbf{X}\vec{\beta}&plus;\vec\epsilon\;\;\;\;\;(2)," target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{Y}&space;=&space;\mathbf{X}\vec{\beta}&plus;\vec\epsilon\;\;\;\;\;(2)," title="\mathbf{Y} = \mathbf{X}\vec{\beta}+\vec\epsilon\;\;\;\;\;(2)," /></a>
</p>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathbf{X}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{X}" title="\mathbf{X}" /></a> - discrete approximation of Laplace transform;
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;$\vec\beta$" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;$\vec\beta$" title="$\vec\beta$" /></a> – elements of spectral function, with length <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;N_F" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;N_F" title="N_F" /></a>;
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathbf{Y}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{Y}" title="\mathbf{Y}" /></a> – given transient vector lenght <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;N_f" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;N_f" title="N_f" /></a>;
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\vec\epsilon" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\vec\epsilon" title="\vec\epsilon" /></a> – normal noise.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathbf{X}&space;=&space;\left(&space;\begin{array}{cccc}&space;x_{11}&space;&&space;x_{12}&space;&&space;\ldots&space;&&space;x_{1(N_f-1)}\\&space;x_{21}&space;&&space;x_{22}&space;&&space;\ldots&space;&&space;x_{2(N_f-1)}\\&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots\\&space;x_{(N_F-1)1}&space;&&space;x_{(N_F-1)2}&space;&&space;\ldots&space;&&space;x_{(N_F-1)(N_f-1)}&space;\end{array}&space;\right)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{X}&space;=&space;\left(&space;\begin{array}{cccc}&space;x_{11}&space;&&space;x_{12}&space;&&space;\ldots&space;&&space;x_{1(N_f-1)}\\&space;x_{21}&space;&&space;x_{22}&space;&&space;\ldots&space;&&space;x_{2(N_f-1)}\\&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots\\&space;x_{(N_F-1)1}&space;&&space;x_{(N_F-1)2}&space;&&space;\ldots&space;&&space;x_{(N_F-1)(N_f-1)}&space;\end{array}&space;\right)" title="\mathbf{X} = \left( \begin{array}{cccc} x_{11} & x_{12} & \ldots & x_{1(N_f-1)}\\ x_{21} & x_{22} & \ldots & x_{2(N_f-1)}\\ \vdots & \vdots & \ddots & \vdots\\ x_{(N_F-1)1} & x_{(N_F-1)2} & \ldots & x_{(N_F-1)(N_f-1)} \end{array} \right)" /></a>
</p>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x_{ij}&space;=&space;\int_{s_j}^{s_{j&plus;1}}\exp{\left(-s\cdot&space;t_i\right)}ds" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;x_{ij}&space;=&space;\int_{s_j}^{s_{j&plus;1}}\exp{\left(-s\cdot&space;t_i\right)}ds" title="x_{ij} = \int_{s_j}^{s_{j+1}}\exp{\left(-s\cdot t_i\right)}ds" /></a>

## The problem

Considering that noise is normal distributed we can maximise the posterior probability and hence minimise the residual sum of squares.  

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathbf{Y}&space;=&space;\mathbf{X}\vec{\beta}&plus;\vec\epsilon" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathbf{Y}&space;=&space;\mathbf{X}\vec{\beta}&plus;\vec\epsilon" title="\mathbf{Y} = \mathbf{X}\vec{\beta}+\vec\epsilon" /></a>
</p>

Since we will fitting the exponential decay with shifted delta function (<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathcal{L}\left\{\delta(t-\tau)\right\}&space;=&space;e^{-\tau&space;s}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\mathcal{L}\left\{\delta(t-\tau)\right\}&space;=&space;e^{-\tau&space;s}" title="\mathcal{L}\left\{\delta(t-\tau)\right\} = e^{-\tau s}" /></a>, for <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\text{Re}(s)>0" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\text{Re}(s)>0" title="\text{Re}(s)>0" /></a>), and in this model weights <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\beta_i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\beta_i" title="\beta_i" /></a> may take zero values. <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;L_1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;L_1" title="L_1" /></a> regression can gives us a sparse solution.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Q&space;=&space;\left(\mathbf{Y}&space;-&space;\mathbf{X}\vec{\beta}\right)^{T}\cdot\left(\mathbf{Y}&space;-&space;\mathbf{X}\vec{\beta}\right)&space;&plus;&space;\lambda&space;\vec\beta" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;Q&space;=&space;\left(\mathbf{Y}&space;-&space;\mathbf{X}\vec{\beta}\right)^{T}\cdot\left(\mathbf{Y}&space;-&space;\mathbf{X}\vec{\beta}\right)&space;&plus;&space;\lambda&space;\vec\beta" title="Q = \left(\mathbf{Y} - \mathbf{X}\vec{\beta}\right)^{T}\cdot\left(\mathbf{Y} - \mathbf{X}\vec{\beta}\right) + \lambda \vec\beta" /></a>
</p>

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\frac{\partial&space;Q}{\partial&space;\beta_i}&space;=&space;-2\mathbf{X}^T\mathbf{Y}&plus;2\mathbf{X}^T\mathbf{X}\vec\beta&space;&plus;\lambda&space;\text{sign}(\beta)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\frac{\partial&space;Q}{\partial&space;\beta_i}&space;=&space;-2\mathbf{X}^T\mathbf{Y}&plus;2\mathbf{X}^T\mathbf{X}\vec\beta&space;&plus;\lambda&space;\text{sign}(\beta)" title="\frac{\partial Q}{\partial \beta_i} = -2\mathbf{X}^T\mathbf{Y}+2\mathbf{X}^T\mathbf{X}\vec\beta +\lambda \text{sign}(\beta)" /></a>
</p>

### Result

![](sc.png)

### Papers

1 <a href="https://sci-hub.do/https://doi.org/10.1137/0730038"> A REGULARIZATION METHOD FOR THE NUMERICAL INVERSION OF THE LAPLACE TRANSFORM. CHEN WEI DONG</a>

2 <a href="https://sci-hub.do/https://doi.org/10.1002/cmr.a.21263"> Laplace Inversion of Low-Resolution NMR Relaxometry Data Using Sparse Representation Methods. PAULA BERMAN ET AL.</a>
