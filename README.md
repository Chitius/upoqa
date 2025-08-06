# UPOQA: Unconstraint Partially-separable Optimization by Quadratic Approximation

UPOQA is a derivative-free model-based optimizer designed for unconstrained optimization problems with **partially-separable** structures. This solver leverages quadratic interpolation models within a trust-region framework to efficiently solve complex optimization problems without requiring gradient information.

For more details, please refer to the [documentation](https://upoqa.readthedocs.io/en/latest/) or [our paper](https://arxiv.org/abs/2506.21948).

## Installation

UPOQA requires Python 3.8 or higher to be installed, and the following python packages should be installed (these will be installed automatically if using *pip*):

- NumPy ([http://www.numpy.org/](http://www.numpy.org/))
- SciPy ([http://www.scipy.org/](http://www.scipy.org/))
- tqdm ([https://github.com/tqdm/tqdm](https://github.com/tqdm/tqdm))

You can install `upoqa` using *pip:*

```bash
pip install upoqa           # minimal install
pip install upoqa[profile]  # + all benchmarking dependencies
```

## Basic Usage

### API in a nutshell

```python
upoqa.minimize(fun, x0, coords={}, maxiter=None, maxfev={}, weights={}, xforms={}, xform_bounds={}, 
               extra_fun=None, npt=None, radius_init=1.0, radius_final=1e-06, noise_level=0, 
               seek_global_minimum=False, f_target=None, tr_shape='structured', callback=None, 
               disp=True, verbose=False, debug=False, return_internals=False, options={}, **kwargs)
```

Returned object (`upoqa.utils.OptimizeResult`) contains `x`, `fun`, element values `funs`, evaluation counts `nfev`, and more—see the full docstring and documentation.

### A Simple Example

Here is a simple example of using the solver to minimize a function with two elements:

$$
\min_{x,y,z\in \mathbb{R}} \quad x^2 + 2y^2 + z^2 + 2xy - (y + 1)z
$$

let's replace $[x,y,z]$ with $\mathbf{x} = [x_1,x_2,x_3]$, and rewrite this problem into

$$
\min_{\mathbf{x}\in\mathbb{R}^3} \quad f_1(x_1, x_2) + f_2(x_2, x_3)
$$

where

$$
f_1(x_1, x_2) = x_1^2 + x_2^2 + 2x_1 x_2, \quad f_2(x_2, x_3) = x_2^2 + x_3^2 - (x_2 + 1)x_3,
$$

then we can optimize it by the following code:

```python
from upoqa import minimize

def f1(x):    # f1(x,y)
    return x[0] ** 2 + x[1] ** 2 + 2 * x[0] * x[1]     # x^2 + y^2 + 2xy

def f2(x):    # f2(y,z)
    return x[0] ** 2 + x[1] ** 2 - (x[0] + 1) * x[1]   # y^2 + z^2 - (y+1)z

fun =    {'xy': f1,     'yz': f2    }
coords = {'xy': [0, 1], 'yz': [1, 2]}
x0 = [0, 0, 0]

result = minimize(fun, x0, coords = coords, disp = False)
print(result)
```

The output will be

```
   message: Success: The resolution has reached its minimum. 
   success: True
       fun: -0.33333333333333215
      funs: xy: 3.3306690738754696e-16
            yz: -0.3333333333333325
 extra_fun: 0.0
         x: [-3.333e-01  3.333e-01  6.667e-01]
       jac: [-3.137e-08 -9.089e-08  2.260e-09]
      hess: [[ 1.962e+00  1.980e+00  0.000e+00]
             [ 1.980e+00  4.042e+00 -1.002e+00]
             [ 0.000e+00 -1.002e+00  1.997e+00]]
       nit: 39
      nfev: xy: 39
            yz: 38
  max_nfev: 39
  avg_nfev: 38.5
      nrun: 1
```

## Mathematical Background

UPOQA solves optimization problems with partially-separable structures:

$$
\min_{x\in\mathbb{R}^n} \quad \sum_{i=1}^q f_i(U_i x),
$$

Where:

- $f_i:\mathbb{R}^{|\mathcal{I}_i|} \to \mathbb{R}$ are black-box element functions whose gradients and hessians are unavailable
- $U_i$ are projection operators selecting relevant variables
- $|\mathcal{I}_i| < n$ (element functions depend on small subsets of variables)

The solver also supports a more general objective form:

$$
\min_{x\in\mathbb{R}^n} \quad f_0(x) + \sum_{i=1}^q w_i h_i\left(f_i(U_i x)\right),
$$

where:

- $f_0$ is a white-box component with known derivatives
- $w_i$ are element weights
- $h_i$ are smooth transformations of element outputs

## Contributing

Contributions are welcome! Please submit pull requests to our repository.

## License

This project is licensed under the GPLv3 License - see the `LICENSE` file for details.
