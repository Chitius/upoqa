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
import numpy.linalg as LA

######################## Test Problem 1 ########################
"""
obj(x, y, z) = (x + y - 1) ** 2 + exp((y + z) ^ 2)

There are two ways to call upoqa:

1. separate directly

obj(x, y, z) = f1(x, y) + f2(y, z)

2. use xform (slightly faster, yet not obvious on such a small problem)

obj(x, y, z) = f1(x, y) + exp(g2(y, z))
"""


def f1(x):
    return (x[0] + x[1] - 1) ** 2


def g2(x):
    return (x[0] + x[1]) ** 2


def f2(x):
    return np.exp(g2(x))


def obj(x):
    return f1(x[0:2]) + f2(x[1:])


x0 = [0, 0, 0]
xopt1 = [1.0, 0.0, 0.0]
fopt1 = 1.0

fun1 = {
    "f1": f1,
    "f2": f2,
}
coords1 = {
    "f1": [0, 1],
    "f2": [1, 2],
}
upoqa_res1 = upoqa.minimize(fun1, x0=x0, coords=coords1, disp=0)

fun2 = {
    "f1": f1,
    "g2": g2,
}
coords2 = {
    "f1": [0, 1],
    "g2": [1, 2],
}
xforms2 = {
    "g2": [lambda x: np.exp(x), lambda x: np.exp(x), lambda x: np.exp(x)],
}
upoqa_res2 = upoqa.minimize(fun2, x0=x0, coords=coords2, xforms=xforms2, disp=0)


def test_prob1_res_correct():
    assert upoqa_res1.success
    assert upoqa_res2.success
    assert abs(fopt1 - upoqa_res1.fun) <= np.finfo(float).eps ** 0.5 * max(1.0, abs(fopt1))
    assert abs(fopt1 - upoqa_res2.fun) <= np.finfo(float).eps ** 0.5 * max(1.0, abs(fopt1))
    assert LA.norm(np.array(xopt1) - upoqa_res1.x) <= 1e-6 * LA.norm(xopt1)
    assert LA.norm(np.array(xopt1) - upoqa_res2.x) <= 1e-6 * LA.norm(xopt1)


def test_prob1_xform_superiority():
    print(f"not use xforms: {upoqa_res1.max_nfev} evals")
    print(f"use xforms:     {upoqa_res2.max_nfev} evals")
    assert upoqa_res2.max_nfev < upoqa_res1.max_nfev


if __name__ == "__main__":
    test_prob1_xform_superiority()
