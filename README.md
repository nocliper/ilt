# Inverse the Laplace Transfom using L1 regression methdod

## The point

The point is to determine the decay rate $\tau$ of the exponential function $e^{-\tau s}$ using numerical routines for Inverting the Laplace Transform. For exponential continous function $F(s)$, for $\text{Re}(s) > 0$:
$$F(s) = \sum_{i=0}^{n} \exp{\left(-\tau_i s\right)}$$
$$f(t) = \mathcal{L}^{-1}\left\{F(s)\right \} = \sum_{i=0}^{n} \delta(t - \tau_i)$$

## Numerical Laplace Transform

For this, we replace (1) with finite-difference approximation (2) from <a href="https://sci-hub.st/https://doi.org/10.1137/0730038">[1]</a>:

$$F(s) = \int_0^{\infty} \exp{(-st)}f(t)dt\text{   (1);}$$

$$ \mathbf{Y} = \mathbf{X}\vec{\beta}+\vec\epsilon\text{    (2),}$$

where $\mathbf{X}$ - discrete approximation of Laplace transform;
$\vec\beta$ – elements of spectral function, with length $N_F$;
$\mathbf{Y}$ – given transient vector lenght $N_f$; 
$\vec\epsilon$ – normal noise.

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
