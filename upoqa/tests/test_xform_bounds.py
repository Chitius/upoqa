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

sys.path.append("..")
import numpy as np
import upoqa
from scipy.optimize import minimize as sp_minimize

np.random.seed(588)
noise_level = 0.0

######################## Test Problem 1 ########################
"""
obj(x, y) = (x - y) ** 2 + sqrt(exp((x + y) ** 2) - 1) = f1(x, y) + sqrt(g2(x, y))
             
Sadly, we can only get noisy evaluation of g2, 
meaning that we may get negative function value from g2.
If that's the case, sqrt(.) will get an invalid input.
"""


def f1(x):
    return (x[0] - x[1]) ** 2


def g2(x):
    return np.exp((x[0] + x[1]) ** 2) - 1 + np.random.randn() * noise_level


def xform_func(g):
    return np.sqrt(g)


def xform_grad(g):
    return 1 / (2 * np.sqrt(g))


def xform_hess(g):
    return -1 / (4 * g**1.5)


def obj(x):
    return f1(x) + xform_func(g2(x))


x0 = [2.0, 1.0]
xopt = [0.0, 0.0]
fopt = 0.0

fun = {
    "f1": f1,
    "g2": g2,
}
xforms = {
    "g2": [xform_func, xform_grad, xform_hess],
}
xform_bounds = {"g2": (1e-16, np.inf)}

noise_level = 0.0
exact_res = sp_minimize(obj, x0, method="L-BFGS-B")

noise_level = 1e-6

res_wrong_res = upoqa.minimize(
    fun, x0=x0, xforms=xforms, noise_level=1, tr_shape="spherical", maxfev=600, disp=1
)

res_with_bound_check = upoqa.minimize(
    fun,
    x0=x0,
    xforms=xforms,
    xform_bounds=xform_bounds,
    noise_level=1,
    tr_shape="spherical",
    maxfev=600,
    disp=1,
)


def test_fail_run():
    assert not res_wrong_res.success


def test_success_with_bound_check():
    assert res_with_bound_check.success
    assert res_with_bound_check.fun <= fopt + (noise_level + np.finfo(float).eps ** 0.5) * max(
        1.0, abs(fopt)
    )
