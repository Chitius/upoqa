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


class BlockDiagBiquadForm(PSProblem):
    r"""Block diagonal biquadratic form optimization problem.

    The objective function consists of ``r`` local functions, each defined on
    an ``n``-dimensional block with ``overlap`` overlapping dimensions between
    consecutive blocks. Each local function has the form:

    .. math::

        f_i(x_i) = \frac{1}{2} (x_i - c_i)^T H_i (x_i - c_i) + \gamma_i^T (x_i - \gamma_i)^4

    where:
    - :math:`x_i` : Variables for the i-th block
    - :math:`c_i` : Center point for the quadratic term
    - :math:`H_i` : Positive definite matrix for the quadratic term
    - :math:`\gamma_i` : Coefficient vector for the biquadratic penalty term,
      randomly generated.

    Parameters
    ----------
    n : int
        Dimension of each local block
    r : int
        Number of local functions/blocks
    overlap : int, optional
        Overlapping dimension between consecutive blocks (default 0)
    center : ndarray, optional
        Global center point vector of shape (dim,). If None, randomly generated.
    x0 : ndarray, optional
        Initial guess for the optimization variables. If None, initialized to zeros.
    mat_gen : callable, optional
        Matrix generator function with signature (k, idx) -> ndarray of shape (k, k).
        If None, uses random positive definite matrices.
    noise_wrapper : callable, optional
        Noise injection function with signature (x, idx) -> noisy_x.
        If None, no noise is added.
    reg : float, optional
        Regularization coefficient for the biquadratic term (default 1.0)
    """

    def __init__(
        self,
        n,
        r,
        overlap=0,
        center=None,
        x0=None,
        mat_gen=None,
        noise_wrapper=None,
        reg=1.0,
    ) -> None:
        if mat_gen is None:
            min_and_max_eig = (0.1, 10)
            mat_gen = lambda k, _: rand_matrix_gen(
                k, min_and_max_eig[0], min_and_max_eig[1]
            )
        if noise_wrapper is None:
            noise_wrapper = lambda x, _: x
        self.dim = (n - overlap) * (r - 1) + n
        super().__init__(dim=self.dim, noise_wrapper=noise_wrapper)
        self.n = n
        self.overlap = overlap
        self.r = r
        self.mats = []
        self.center = (
            np.atleast_1d(np.asarray(center).squeeze())
            if center is not None
            else np.random.randn(self.dim)
        )
        self.gammas = [np.random.randn(n) / np.sqrt(n) for _ in range(r)]
        self.x0 = (
            np.zeros(self.dim)
            if x0 is None
            else np.atleast_1d(np.asarray(x0).squeeze())
        )
        self.reg = reg

        def sub_fun_gen(i, center):

            assert isinstance(center, np.ndarray)

            def _fun(x):
                tmp = x - center
                return tmp.dot(tmp @ self.mats[i])

            return _fun

        def sub_penalty_gen(i, gamma):

            def _fun(x):
                tmp = x - gamma
                return gamma.dot(np.power(tmp, 4))

            return _fun

        for i in range(r):
            coord = [x for x in range((n - overlap) * i, (n - overlap) * i + n)]
            mat = mat_gen(n, i)
            ele_idx = f"block_{i+1}"
            biquad_idx = f"biquad_{i+1}"
            center_ele = self.center[coord]
            self.mats.append(mat)
            self.append(ele_idx, sub_fun_gen(i, center_ele), coord)
            self.append(biquad_idx, sub_penalty_gen(i, self.gammas[i]), coord, self.reg)
