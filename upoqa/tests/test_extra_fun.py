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

import sys
import numpy as np
import upoqa
from scipy.optimize import minimize as sp_minimize
import numpy.linalg as LA
from upoqa.problems import PSProblem


def f1(x):  # f1(x,y)
    return x[0] ** 2 + x[1] ** 2 + 2 * x[0] * x[1]  # x^2 + y^2 + 2xy


def f2(x):  # f2(y,z)
    return x[1] ** 2 + x[2] ** 2 - (x[1] + 1) * x[2]  # y^2 + z^2 - (y+1)z


def f2_grad(x):
    return np.array([0, 2 * x[1] - x[2], 2 * x[2] - x[1] - 1])


def f2_hess(x):
    return np.array([[0, 0, 0], [0, 2, -1], [0, -1, 2]])


fun = {"xy_part": f1}
coords = {"xy_part": [0, 1]}
x0 = [0, 0, 0]
dim = 3
prob = PSProblem(fun, coords=coords, dim=dim)
extra_fun = [f2, f2_grad, f2_hess]

xopt = np.array([-1 / 3, 1 / 3, 2 / 3])
fopt = -1 / 3

standard_res = upoqa.minimize(prob.fun, x0=x0, coords=prob.coords, extra_fun=extra_fun, disp=0)


def test_prob1_correct():
    assert standard_res.success
    assert abs(fopt - standard_res.fun) <= np.finfo(float).eps ** 0.5 * max(1.0, abs(fopt))
    assert LA.norm(xopt - standard_res.x) <= 1e-6 * LA.norm(xopt)
