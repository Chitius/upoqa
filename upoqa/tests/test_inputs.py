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
import numpy.linalg as LA
from problems import PSProblem


######################## Test Problem 1 ########################
def f1(x):  # f1(x,y)
    return x[0] ** 2 + x[1] ** 2 + 2 * x[0] * x[1]  # x^2 + y^2 + 2xy


def f2(x):  # f2(y,z)
    return x[0] ** 2 + x[1] ** 2 - (x[0] + 1) * x[1]  # y^2 + z^2 - (y+1)z


fun1 = {'xy_part': f1, 'yz_part': f2}
coords1 = {'xy_part': [0, 1], 'yz_part': [1, 2]}
x0_1 = [0, 0, 0]
dim = 3
prob1 = PSProblem(fun1, coords=coords1, dim=dim)
xopt1 = np.array([-1 / 3, 1 / 3, 2 / 3])
fopt1 = -1 / 3

standard_res1 = upoqa.minimize(prob1.fun, x0=x0_1, coords=prob1.coords, disp=0)


def test_prob1_correct():
    assert standard_res1.success
    assert abs(fopt1 - standard_res1.fun) <= np.finfo(float).eps ** 0.5 * max(1.0, abs(fopt1))
    assert LA.norm(xopt1 - standard_res1.x) <= 1e-6 * LA.norm(xopt1)


def test_all_list1():
    prob1.clear()
    res = upoqa.minimize([f1, f2], x0=x0_1, coords=list(coords1.values()), disp=0)
    assert res.success
    assert np.equal(res.x, standard_res1.x).all()
    assert np.equal(res.fun, standard_res1.fun).all()
    assert np.equal(res.nit, standard_res1.nit).all()
    assert np.equal(res.nrun, standard_res1.nrun).all()
    assert np.equal(res.max_nfev, standard_res1.max_nfev).all()


def test_partial_list1():
    prob1.clear()
    res = upoqa.minimize([f1, f2], x0=x0_1, coords=prob1.coords, disp=0)
    print(res.message)
    assert not res.success


def test_partial_list2():
    prob1.clear()
    res = upoqa.minimize(prob1.fun, x0=x0_1, coords=list(coords1.values()), disp=0)
    assert res.success
    assert np.equal(res.x, standard_res1.x).all()
    assert np.equal(res.fun, standard_res1.fun).all()
    assert np.equal(res.nit, standard_res1.nit).all()
    assert np.equal(res.nrun, standard_res1.nrun).all()
    assert np.equal(res.max_nfev, standard_res1.max_nfev).all()


######################## Test Problem 2 ########################
"""
obj(x, y, z) =
    [1 / (1 + exp(- f1(x, y))) - 0.5] ^ 2 
    + 
    [1 / (1 + exp(- f2(y, z))) + 0.3] ^ 2 
    + 
    0.01 * || x ||^2 
    + 
    0.01 * || y ||^2 
    +
    0.01 * || z ||^2
    
xopt = [ 0.23691907573189747, -1.0248519445077102,  3.3235559784457394]

opt = 0.6071670615869595

f1(x, y) = 0.6 * y - 0.2 * x

f2(y, z) = 0.6 * y - 0.2 * z
"""


def sigmoid(f):
    return 1 / (1 + np.exp(-f)) * (f > 0) + np.exp(f) / (1 + np.exp(f)) * (f <= 0)


def sigmoid_grad(f):
    return sigmoid(f) * (1 - sigmoid(f))


def sigmoid_hess(f):
    return sigmoid(f) * (1 - sigmoid(f)) * (1 - 2 * sigmoid(f))


reg = 0.01


def obj(x):
    return (
        xform1(g1(x[0:2]))
        + xform2(g2(x[1:]))
        + reg * LA.norm(x[0]) ** 2
        + reg * LA.norm(x[1]) ** 2
        + reg * LA.norm(x[2]) ** 2
    )


def xform1(f):
    return (sigmoid(f) - 0.5) ** 2


def xform2(f):
    return (sigmoid(f) + 0.3) ** 2


def xform1_grad(f):
    return 2 * sigmoid(f) * sigmoid_grad(f) - sigmoid_grad(f)


def xform2_grad(f):
    return 2 * sigmoid(f) * sigmoid_grad(f) + 0.6 * sigmoid_grad(f)


def xform1_hess(f):
    return 2 * (sigmoid_grad(f)) ** 2 + 2 * sigmoid(f) * sigmoid_hess(f) - sigmoid_hess(f)


def xform2_hess(f):
    return 2 * (sigmoid_grad(f)) ** 2 + 2 * sigmoid(f) * sigmoid_hess(f) + 0.6 * sigmoid_hess(f)


def g1(x):
    return 0.6 * sigmoid(0.4 * x[1]) - 0.2 * x[0]


def g2(x):
    return 0.6 * sigmoid(0.4 * x[0]) - 0.2 * x[1]


def norm2(x):
    return LA.norm(x) ** 2


# def f1(x):    # f1(x,y)
#     return x[0] ** 2 + x[1] ** 2 + 2 * x[0] * x[1]     # exp(x^2 + y^2 + 2xy)

# def f2(x):    # f2(y,z)
#     return x[0] ** 2 + x[1] ** 2 - (x[0] + 1) * x[1]   # y^2 + z^2 - (y+1)z

fun2 = {
    'xy_part': g1,
    'yz_part': g2,
    'x_penalty': norm2,
    'y_penalty': norm2,
    'z_penalty': norm2,
}
coords2 = {
    'xy_part': [0, 1],
    'yz_part': [1, 2],
    'x_penalty': [0],
    'y_penalty': [1],
    'z_penalty': [2],
}
xforms2 = {
    'xy_part': [xform1, xform1_grad, xform1_hess],
    'yz_part': [xform2, xform2_grad, xform2_hess],
}
weights2 = {
    'xy_part': 1.0,
    'yz_part': 1.0,
    'x_penalty': reg,
    'y_penalty': reg,
    'z_penalty': reg,
}
x0_2 = [0, 0, 0]
prob2 = PSProblem(fun2, coords=coords2, weights=weights2, xforms=xforms2, dim=3)

correct_res2 = sp_minimize(obj, x0_2, method='L-BFGS-B')
xopt2 = np.array([0.23691907573189747, -1.0248519445077102, 3.3235559784457394])
fopt2 = 0.6071670615869595

standard_res2 = upoqa.minimize(
    prob2.fun,
    x0=x0_2,
    coords=prob2.coords,
    weights=prob2.weights,
    xforms=prob2.xforms,
    disp=0,
)


def test_prob2_correct():
    assert standard_res2.success
    assert abs(standard_res2.fun - correct_res2.fun) <= np.finfo(float).eps ** 0.5 * max(
        1.0, abs(fopt2)
    )


def test_all_list2():
    prob2.clear()
    res = upoqa.minimize(
        list(fun2.values()),
        x0=x0_2,
        coords=list(coords2.values()),
        weights=list(weights2.values()),
        xforms=list(xforms2.values()),
        disp=0,
    )
    assert res.success
    assert np.equal(res.x, standard_res2.x).all()
    assert np.equal(res.fun, standard_res2.fun).all()
    assert np.equal(res.nit, standard_res2.nit).all()
    assert np.equal(res.nrun, standard_res2.nrun).all()
    assert np.equal(res.max_nfev, standard_res2.max_nfev).all()


def test_partial_list3():
    prob2.clear()
    res = upoqa.minimize(
        prob2.fun,
        x0=x0_2,
        coords=list(coords2.values()),
        weights=prob2.weights,
        xforms=list(xforms2.values()),
        disp=0,
    )
    assert res.success
    assert np.equal(res.x, standard_res2.x).all()
    assert np.equal(res.fun, standard_res2.fun).all()
    assert np.equal(res.nit, standard_res2.nit).all()
    assert np.equal(res.nrun, standard_res2.nrun).all()
    assert np.equal(res.max_nfev, standard_res2.max_nfev).all()


def test_missing_input1():
    prob2.clear()
    weights3 = {'x_penalty': reg, 'y_penalty': reg, 'z_penalty': reg}
    res = upoqa.minimize(
        prob2.fun,
        x0=x0_2,
        coords=list(coords2.values()),
        weights=weights3,
        xforms=list(xforms2.values()),
        disp=0,
    )
    assert res.success
    assert np.equal(res.x, standard_res2.x).all()
    assert np.equal(res.fun, standard_res2.fun).all()
    assert np.equal(res.nit, standard_res2.nit).all()
    assert np.equal(res.nrun, standard_res2.nrun).all()
    assert np.equal(res.max_nfev, standard_res2.max_nfev).all()
