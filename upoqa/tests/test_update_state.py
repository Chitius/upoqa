import sys

sys.path.append("..")
import numpy as np
import upoqa
import numpy.linalg as LA
from scipy.optimize import minimize as sp_minimize
from upoqa.problems import rand_matrix_gen

######################## Test Problem 1 ########################
"""
obj(x, y) = (x + y - 1) ** 2 + a * |xy| 

Direct optimization is difficult for the existence of L1 term, 
so we optimize it using a L2 smoothing on this term:

obj(x, y) = f1(x, y) + h2(f2(x, y), tau)

where, 

    f1(x, y) = (x + y - 1) ** 2

    f2(x, y) = (xy) ** 2
    
    h2(x, tau) = sqrt(x + tau), tau > 0
"""

a = 5.0
start_tau = 1e-2
tau = start_tau


def f1(x):
    return (x[0] + x[1] - 1) ** 2


def f2(x):
    return (x[0] * x[1]) ** 2


def h2(x, tau):
    return a * np.sqrt(x + tau)


def h2_grad(x, tau):
    # \nabla h2(x, tau) = 0.5 * (x + tau) ** (- 0.5)
    return 0.5 * a / np.sqrt(x + tau)


def h2_hess(x, tau):
    # \nabla^2 h2(x, tau) = - 0.25 * (x + tau) ** (- 1.5)
    return -0.25 * a / np.power(x + tau, 1.5)


def obj_1(x, tau):
    return f1(x) + h2(f2(x), tau)


x0_1 = [-5, -5]
fopt1 = 0.0

fun1 = {
    "loss": f1,
    "xy_L1": f2,
}

coords1 = {
    "loss": [0, 1],
    "xy_L1": [0, 1],
}

xforms1 = {
    "xy_L1": [lambda x: h2(x, tau), lambda x: h2_grad(x, tau), lambda x: h2_hess(x, tau)],
}

xform_bounds1 = {
    "xy_L1": (0, np.inf),
}


def callback_func(intermediate_result):
    tau = start_tau * 0.85**intermediate_result.nit
    intermediate_result.manager.update_xforms(
        {
            "xy_L1": [lambda x: h2(x, tau), lambda x: h2_grad(x, tau), lambda x: h2_hess(x, tau)],
        }
    )


upoqa_res1 = upoqa.minimize(
    fun1,
    x0=x0_1,
    coords=coords1,
    xforms=xforms1,
    callback=callback_func,
    xform_bounds=xform_bounds1,
    disp=0,
    debug=1,
)

# direct optimization fails
sp_res1 = sp_minimize(lambda x: obj_1(x, start_tau), x0_1)


def test_prob1_res_correct():
    assert upoqa_res1.success
    assert obj_1(upoqa_res1.x, 0.0) < 1e-6
    assert obj_1(sp_res1.x, 0.0) > 1e2 * obj_1(upoqa_res1.x, 0.0)
    assert obj_1(upoqa_res1.x, start_tau * 0.85**upoqa_res1.nit) == upoqa_res1.fun


def test_prob1_final_xform_correct():
    trial_x = 1e-2
    assert upoqa_res1.manager.xforms[1][0](trial_x) == h2(trial_x, start_tau * 0.85**upoqa_res1.nit)
    assert upoqa_res1.manager.xforms[1][1](trial_x) == h2_grad(
        trial_x, start_tau * 0.85**upoqa_res1.nit
    )
    assert upoqa_res1.manager.xforms[1][2](trial_x) == h2_hess(
        trial_x, start_tau * 0.85**upoqa_res1.nit
    )


######################## Test Problem 2 ########################
"""
obj(w, x, y, z; tau) = 
    (0.2 * w^2 + 2 * wx + 10 * x^2) + 
    (0.2 * y^2 + 2 * yz + 10 * z^2) + 
    tau * (
        |wy + xz|^2  + |w^2 + x^2 - 1|^2 + |y^2 + z^2 - 1|^2
    )

when tau approach infty,
fopt converges to 0.0990001 + 10.1009999 = 10.2000

obj(w, x, y, z; tau) = f(w, x) + f(y, z) + tau * p1(w, x, y, z) + tau * p2(w, x) + tau * p2(y, z)

where, 

    f(w, x) = (0.2 * w^2 + 2 * wx + 10 * x^2)
    
    p1(w, x, y, z) = |wy + xz|^2
    
    p2(w, x) = |w^2 + x^2 - 1|^2
    
let's optimize it using a varing `tau` -- small `tau` accelerates the convergence, 
and big `tau` ensure that the solution is accurate.
"""


def f_wx(wx):
    w, x = wx[0], wx[1]
    return 0.2 * w**2 + 2 * w * x + 10 * x**2


def f_yz(yz):
    y, z = yz[0], yz[1]
    return 0.2 * y**2 + 2 * y * z + 10 * z**2


def p1(wxyz):
    w, x, y, z = wxyz[0], wxyz[1], wxyz[2], wxyz[3]
    return (w * y + x * z) ** 2


def p2_wx(wx):
    w, x = wx[0], wx[1]
    return (w**2 + x**2 - 1) ** 2


def p2_yz(yz):
    y, z = yz[0], yz[1]
    return (y**2 + z**2 - 1) ** 2


start_tau2 = 1e-2
tau2 = start_tau2


def obj_2(x, tau):
    return f_wx(x[:2]) + f_yz(x[2:]) + tau * (p1(x) + p2_wx(x[:2]) + p2_yz(x[2:]))


fun2 = {
    "quad1": f_wx,
    "quad2": f_yz,
    "penalty1": p1,
    "penalty2_wx": p2_wx,
    "penalty2_yz": p2_yz,
}

coords2 = {
    "quad1": [0, 1],  # w, x
    "quad2": [2, 3],  # y, z
    "penalty1": [0, 1, 2, 3],  # w, x, y, z
    "penalty2_wx": [0, 1],  # w, x
    "penalty2_yz": [2, 3],  # y, z
}

weights = {
    "penalty1": tau2,
    "penalty2_wx": tau2,
    "penalty2_yz": tau2,
}


def callback_func2(intermediate_result):
    tau2 = start_tau2 * (1.1**intermediate_result.nit)
    intermediate_result.manager.update_weights(
        {
            "penalty1": tau2,
            "penalty2_wx": tau2,
            "penalty2_yz": tau2,
        }
    )


x0_2 = [1, 1, 1, 1]
fopt2 = 10.2000

upoqa_res2 = upoqa.minimize(
    fun2, x0=x0_2, coords=coords2, weights=weights, callback=callback_func2, disp=0, debug=1
)

# Direct optimization with fixed tau (large value) for comparison
# get fun: 9.819850000011742, the solution is not the expected one
sp_res2_small_tau = sp_minimize(lambda x: obj_2(x, 1e2), x0_2)
# get fun: 10.196198834623207, near success
sp_res2_medium_tau = sp_minimize(lambda x: obj_2(x, 1e4), x0_2)
# get fun: 666668.3710234876, totally fail
sp_res2_large_tau = sp_minimize(lambda x: obj_2(x, 1e6), x0_2)


def test_prob2_res_correct():
    assert abs(upoqa_res2.fun - fopt2) < 1e-5 * abs(fopt2)
    # Direct method may not converge to the correct minimum
    assert abs(sp_res2_small_tau.fun - fopt2) > 1e-2 * fopt2
    assert abs(sp_res2_small_tau.fun - fopt2) > 1e-2 * fopt2
    assert abs(sp_res2_large_tau.fun - fopt2) > 1e2 * fopt2


def test_prob2_final_tau_correct():
    final_tau = start_tau2 * (1.1**upoqa_res2.nit)
    assert upoqa_res2.manager.weights[0] == 1.0
    assert upoqa_res2.manager.weights[1] == 1.0
    assert upoqa_res2.manager.weights[2] == final_tau
    assert upoqa_res2.manager.weights[3] == final_tau
    assert upoqa_res2.manager.weights[4] == final_tau
