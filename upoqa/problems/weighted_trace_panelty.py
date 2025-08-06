# Copyright (c) 2025, Yichuan Liu and Yingzhou Li
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from .problem_base import PSProblem
from .utils import *


class WeightedTracePenaltyProb(PSProblem):
    r"""Weighted trace-penalty optimization problem for eigenvalue computation.

    Solves the optimization problem:
    .. math::
        \min_{X \in \mathbb{R}^{n \times r}}
        \operatorname{tr}(X^T A X) + \frac{\mu}{2} \|X^T X\|_F^2

    where:
    - :math:`w_i` are weights
    - :math:`\mu` is the regularization parameter
    - The constraints enforce orthonormality of eigenvectors when :math:`\mu`
      is large enough.

    Parameters
    ----------
    n : int
        Dimension of the problem (size of matrix A)
    r : int
        Number of eigenvectors to compute
    A : ndarray
        Symmetric input matrix (n x n).
    reg : float, optional
        Regularization coefficient. Default: 1.0
    cons : float, optional
        Constant offset in objective. Default: 0.0
    lumped_reg : bool, optional
        If True, wrap the regularization coefficient into the element
        function callables. Otherwise, treat the regularization coefficient
        as part of ``weights``.
    eval_diag : bool, optional
        Evaluate diagonal terms. Default: True
    noise_wrapper : callable, optional
        Noise injection function. Default: None
    weights : list or dict, optional
        Weight factors for eigenvalues. Default: [r, r-1, ..., 1]
    normalize : bool, optional
        Normalize input vectors everytime function evaluation is called.
        Default: True
    """

    def __init__(
        self,
        n,
        r,
        A,
        reg=1.0,
        cons=0.0,
        lumped_reg=True,
        eval_diag=True,
        noise_wrapper: callable = None,
        weights=None,
        normalize=True,
    ) -> None:
        self.n = n
        self.r = r
        self.weights = weights if weights is not None else [r - i for i in range(r)]
        # assert weights must be non-increasing
        if isinstance(self.weights, list):
            self.weights = {i: float(w) for i, w in enumerate(self.weights)}

        super().__init__(
            elements=dict(),
            coords=dict(),
            xforms=dict(),
            dim=n * r,
            weights=self.weights,
            noise_wrapper=noise_wrapper,
            nfev_mode="wst",
        )

        self.A = A.copy()
        self.cons = cons
        self.reg = reg
        self.lumped_reg = lumped_reg
        self.normalize = normalize
        self.x0 = np.ones(self.dim, dtype=np.float64)

        def sub_fun_gen(i, j=None):
            if j is None:

                def _fun(x):
                    if self.normalize:
                        x = normalize_func(x, self.n, 1)
                    return x.dot(x @ A)

            elif j != i:
                if lumped_reg:

                    def _fun(x):
                        if self.normalize:
                            x = normalize_func(x, self.n, 2)
                        x, y = x[0 : x.size // 2], x[x.size // 2 :]
                        return self.reg * np.power(x.dot(y), 2)

                else:

                    def _fun(x):
                        if self.normalize:
                            x = normalize_func(x, self.n, 2)
                        x, y = x[0 : x.size // 2], x[x.size // 2 :]
                        return np.power(x.dot(y), 2)

            else:
                if lumped_reg:

                    def _fun(x):
                        if self.normalize:
                            x = normalize_func(x, self.n, 1)
                        return self.reg * 0.5 * np.power(x.dot(x) - 1, 2)

                else:

                    def _fun(x):
                        if self.normalize:
                            x = normalize_func(x, self.n, 1)
                        return 0.5 * np.power(x.dot(x) - 1, 2)

            return _fun

        for i in range(r):
            coord = [x for x in range(n * i, n * i + n)]
            ele_idx = i
            self.weights[ele_idx] = self.weights.get(ele_idx, 1.0)
            self.append(ele_idx, sub_fun_gen(i, None), coord)

        for j in range(r):
            self.weights[ele_idx] = self.weights.get(ele_idx, 1.0)
            if eval_diag:
                coord = [x for x in range(n * j, n * j + n)]
                ele_idx = (j, j)
                self.append(ele_idx, sub_fun_gen(j, j), coord)
            for i in range(j):
                coord = [x for x in range(n * i, n * i + n)] + [
                    x for x in range(n * j, n * j + n)
                ]
                ele_idx = (i, j)
                if not lumped_reg:
                    self.weights[ele_idx] = self.reg
                else:
                    self.weights[ele_idx] = 1.0
                self.append(ele_idx, sub_fun_gen(i, j), coord)

        if self.cons != 0.0:
            self.append("cons", lambda _: self.cons, [0])

        eigenvalues, eigenvectors = smallest_eigenvectors(A, r)
        if not normalize:
            pass
            # TODO
        else:
            xopt = eigenvectors.flatten()
            fopt = sum(
                self.weights[ele_idx] * eigenvalues[ele_idx] for ele_idx in range(r)
            )
        self.sol_info.update(xopt=xopt, fopt=fopt)
        self.y_shift = -fopt


def normalize_func(X, dim, r, ord=2):
    r"""
    Normalize vectors to unit norm.

    Reshapes input into r vectors of dimension dim and normalizes each to unit norm.

    Supports three input formats:
    1. 1D array (dim*r elements)
    2. 2D array (r x dim)
    3. 2D array (N x dim*r) - batch mode

    Parameters
    ----------
    X : ndarray
        Input vector(s) to normalize
    dim : int
        Dimension of each vector
    r : int
        Number of vectors
    ord : int, optional
        Norm order (default: 2, Euclidean)

    Returns
    -------
    ndarray
        Normalized vectors in same shape as input

    Raises
    ------
    ValueError
        If NaN values encountered during normalization
    Exception
        For unsupported input formats
    """
    if X.ndim == 1 and X.size == dim * r:
        X_sample = X.reshape(r, dim)
        X_vecnorm = np.linalg.norm(X_sample, axis=1, ord=ord).reshape(-1, 1)
        if np.isnan(X_vecnorm).any():
            raise ValueError("NaN values encountered!")
            return X
        return (X_sample / X_vecnorm).flatten()
    elif X.shape[0] == r and X.shape[1] == dim:
        return X / np.linalg.norm(X, axis=1, ord=ord).reshape(-1, 1)
    elif X.shape[1] == dim * r:
        N = X.shape[0]
        value = np.zeros_like(X)
        for i in range(N):
            X_sample = X[i, :].reshape(r, dim)
            value[i] = (
                X_sample / np.linalg.norm(X_sample, axis=1, ord=ord).reshape(-1, 1)
            ).flatten()
        return value.reshape(N, dim * r)
    else:
        raise Exception("Wrong input format for normalize_func(X, dim, r, ord = 2).")
