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

"""
Wrapped Deriative-Free Optimizers Ready for Testing
====
"""

from termcolor import colored
import numpy as np
import time
import traceback
from scipy.optimize import minimize as sp_minimize
from typing import Dict, Any, Literal, Tuple, Optional, Callable, Union, List
from upoqa.problems import *
from upoqa.utils import OptimizeResult
from scipy.optimize import OptimizeResult as ScipyOptimizeResult
from matplotlib.lines import Line2D
import warnings
from utils import _get_meta_info_template, _save_log_for_a_run, _save_or_print_figure_for_a_run

REGISTERED_SOLVERS = {}


def wrap_dfo_solver(default_label: str, register: bool = True) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(
            prob: PSProblem,
            *args,
            f_target: Optional[float] = None,
            label: str = default_label,
            params: Dict[Any, Any] = {},
            plot_fig: bool = True,  # plot but not necessarily show
            show_fig: bool = False,
            save_fig: Optional[str] = None,  # path to save the figure, None if not save
            save_log: Optional[str] = None,  # path to save the log file, None if not save
            prob_suffix: str = "",
            plot_label: str = default_label,
            identifier: Optional[str] = None,
            fmt: Optional[str] = None,
            extra_log_info: Optional[dict] = None,
            **kwargs,
        ) -> Tuple[Union[OptimizeResult, ScipyOptimizeResult], Optional[Line2D]]:
            identifier = identifier or default_label
            start_time = time.time()
            print(
                "# Run "
                + colored(f"{label}", "yellow", attrs=["bold"])
                + " on problem: "
                + colored(f"{prob.name}", "blue", attrs=["bold"])
                + f" (source: {prob.meta_info.get('source', 'unknown')}, index: {prob.meta_info.get('problem_idx', 'unknown')})"
            )

            prob_info = {
                "meta_info": {
                    "Solver Params": params,
                    "f_target": f_target,
                    "Identifier": identifier,
                    "Objective Function Value at x0": (
                        prob.fun_eval(prob.x0) if hasattr(prob, 'x0') else None
                    ),
                },
                "optimize": {},
            }

            prob_info["meta_info"].update(_get_meta_info_template(prob, label)["meta_info"])
            if extra_log_info and isinstance(extra_log_info, dict):
                prob_info['meta_info'].update(extra_log_info)

            try:
                prob.clear()
                res, solver_info = func(prob, *args, f_target=f_target, params=params, **kwargs)
            except Exception as e:
                print(
                    colored("  Failed: ", color="light_red") + f"{e}\n\n{traceback.format_exc()}\n"
                )
                raise e

            try:
                if f_target is not None:
                    prob_info["optimize"].update({"ftarget_satisfied": bool(res.fun <= f_target)})
                print(colored("  Finished", color="green"), end="")
                if "max_nfev" in res:
                    print(f" ({time.time() - start_time:.2f}s, {res.max_nfev} evals)", end="")
                elif "nfev" in res:
                    print(f" ({time.time() - start_time:.2f}s, {res.nfev} evals)", end="")
                else:
                    print(f" ({time.time() - start_time:.2f}s)", end="")

                print(f", Final Objective Value is {res.fun}.")

                if f_target is not None and res.fun > f_target:
                    print(
                        colored("  Warning: ", color="light_yellow")
                        + f"The final objective value is {res.fun}, not satisfying <= f_target = {f_target}."
                    )
                print("")
                if solver_info and isinstance(solver_info, dict):
                    prob_info['meta_info'].update(solver_info)
                _save_log_for_a_run(
                    prob,
                    res,
                    solver_name=default_label,
                    prob_suffix=prob_suffix,
                    save_log=save_log,
                    extra_info=prob_info,
                )
                line = _save_or_print_figure_for_a_run(
                    prob,
                    label=plot_label,
                    solver_name=default_label,
                    prob_suffix=prob_suffix,
                    plot_fig=plot_fig,
                    show_fig=show_fig,
                    save_fig=save_fig,
                    fmt=fmt,
                )
            except Exception as e:
                line = None
                print(
                    f"\nPost-process for the result of the {default_label} solver failed: {e}\n\n{traceback.format_exc()}\n"
                )

            return res, line

        if register and default_label not in REGISTERED_SOLVERS:
            REGISTERED_SOLVERS[default_label] = wrapper

        return wrapper

    return decorator


@wrap_dfo_solver("upoqa")
def run_upoqa_on_prob(
    prob: PSProblem,
    maxfev: Union[int, Dict[Any, int]],
    maxiter: Optional[int] = None,
    f_target: Optional[float] = None,
    callback: Optional[callable] = None,
    disp: Optional[bool] = None,
    params: Dict[Any, Any] = {},
    **kwargs,
) -> Tuple[OptimizeResult, Optional[Line2D]]:
    import upoqa

    params.update({"maxiter": maxiter})
    params.update({"maxfev": maxfev})
    if disp is not None:
        params.update({"disp": disp})

    def wrapped_callback(intermediate_result):
        with prob.debug_mode(incre_nfev=False, noisy=False):
            fval_nf = prob.fun_eval(intermediate_result.x)
        prob._update_history_fun(fval_nf)
        if callback:
            return callback(intermediate_result)
        else:
            return False

    res = upoqa.minimize(
        prob.fun,
        prob.x0,
        coords=prob.coords,
        weights=prob.weights,
        f_target=f_target,
        callback=wrapped_callback,
        **params,
    )
    solver_info = {
        "Solver Support Stopping Criteria": True,
        "Solver Version": (upoqa.__version__ if hasattr(upoqa, "__version__") else "unknown"),
    }
    return res, solver_info


@wrap_dfo_solver("upoqa (single)")
def run_upoqa_on_prob(
    prob: PSProblem,
    maxfev: Union[int, Dict[Any, int]],
    maxiter: Optional[int] = None,
    f_target: Optional[float] = None,
    callback: Optional[callable] = None,
    disp: Optional[bool] = None,
    params: Dict[Any, Any] = {},
    **kwargs,
) -> Tuple[OptimizeResult, Optional[Line2D]]:
    import upoqa

    params.update({"maxiter": maxiter})
    params.update({"maxfev": maxfev})
    if disp is not None:
        params.update({"disp": disp})

    res = upoqa.minimize(prob.fun_eval, prob.x0, f_target=f_target, callback=callback, **params)
    solver_info = {
        "Solver Support Stopping Criteria": True,
        "Solver Version": (upoqa.__version__ if hasattr(upoqa, "__version__") else "unknown"),
    }
    return res, solver_info


@wrap_dfo_solver("uobyqa", register=False)  # pdfo is not compatible with numpy>=2.0?
def run_uobyqa_on_prob(
    prob: PSProblem,
    maxfev: int,
    f_target: Optional[float] = None,
    disp: Optional[bool] = None,
    params: Dict[Any, Any] = {},
    **kwargs,
) -> Tuple[OptimizeResult, Optional[Line2D]]:
    import pdfo

    if "options" in params:
        params["options"].update({"maxfev": maxfev})
    else:
        params["options"] = {"maxfev": maxfev}
    if f_target:
        params["options"].update({"f_target": f_target})
    if disp is not None:
        params["options"].update({"quiet": not disp})
    res = pdfo.pdfo(prob.fun_eval, prob.x0, method='UOBYQA', bounds=None, **params)
    solver_info = {"Solver Support Stopping Criteria": True}
    return res, solver_info


@wrap_dfo_solver("pybobyqa")
def run_pybobyqa_on_prob(
    prob: PSProblem,
    maxfev: int,
    f_target: Optional[float] = None,
    disp: Optional[bool] = None,
    params: Dict[Any, Any] = {},
    **kwargs,
) -> Tuple[OptimizeResult, Optional[Line2D]]:
    import pybobyqa

    params.update({"maxfun": maxfev})
    if f_target:
        if "user_params" not in params:
            params["user_params"] = {}
        params["user_params"].update({"model.abs_tol": f_target})
    if disp is not None:
        params["print_progress"] = bool(disp)
    soln = pybobyqa.solve(prob.fun_eval, prob.x0, **params)
    res = OptimizeResult(
        x=soln.x,
        fun=soln.f,
        grad=soln.gradient,
        hess=soln.hessian,
        message=soln.msg,
        nfev=soln.nx,
        nrun=soln.nruns,
        nit=None,
        diagnostic_info=soln.diagnostic_info,
    )
    solver_info = {"Solver Support Stopping Criteria": True}
    return res, solver_info


@wrap_dfo_solver("l-bfgs-b (ffd)")
def run_ffd_l_bfgs_b_on_prob(
    prob: PSProblem,
    maxfev: int,
    f_target: Optional[float] = None,
    callback: Optional[callable] = None,
    disp: Optional[bool] = None,
    params: Dict[Any, Any] = {},
    **kwargs,
) -> Tuple[ScipyOptimizeResult, Optional[Line2D]]:
    # see also: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
    if "options" in params:
        params["options"].update({"maxfun": maxfev})
    else:
        params["options"] = {"maxfun": maxfev}
    if disp is not None:
        params["options"].update({"disp": bool(disp)})

    def callback_with_ftarget_check(intermediate_result):
        ret = None
        if callback and callable(callback):
            try:
                ret = callback(intermediate_result)
            except Exception as _:
                ret = callback(intermediate_result.x)
        if f_target is not None:
            if intermediate_result.fun <= f_target:
                raise StopIteration
        return ret

    res = sp_minimize(
        prob.fun_eval,
        prob.x0,
        method='L-BFGS-B',
        jac='2-point',
        callback=callback_with_ftarget_check,
        **params,
    )
    solver_info = {"Solver Support Stopping Criteria": True}
    return res, solver_info


@wrap_dfo_solver("l-bfgs-b (cfd)")
def run_cfd_l_bfgs_b_on_prob(
    prob: PSProblem,
    maxfev: int,
    f_target: Optional[float] = None,
    callback: Optional[callable] = None,
    disp: Optional[bool] = None,
    params: Dict[Any, Any] = {},
    **kwargs,
) -> Tuple[ScipyOptimizeResult, Optional[Line2D]]:
    # see also: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
    if "options" in params:
        params["options"].update({"maxfun": maxfev})
    else:
        params["options"] = {"maxfun": maxfev}
    if disp is not None:
        params["options"].update({"disp": bool(disp)})

    def callback_with_ftarget_check(intermediate_result):
        ret = None
        if callback and callable(callback):
            try:
                ret = callback(intermediate_result)
            except Exception as _:
                ret = callback(intermediate_result.x)
        if f_target is not None:
            if intermediate_result.fun <= f_target:
                raise StopIteration
        return ret

    res = sp_minimize(
        prob.fun_eval,
        prob.x0,
        method='L-BFGS-B',
        jac='3-point',
        callback=callback_with_ftarget_check,
        **params,
    )
    solver_info = {"Solver Support Stopping Criteria": True}
    return res, solver_info


@wrap_dfo_solver("imfil")
def run_imflt_on_prob(
    prob: PSProblem,
    maxfev: int,
    f_target: Optional[float] = None,
    params: Dict[Any, Any] = {},
    **kwargs,
) -> Tuple[OptimizeResult, Optional[Line2D]]:
    """
    see also:
    - https://scikit-quant.readthedocs.io/en/latest/starting.html
    - https://ctk.math.ncsu.edu/imfil.html
    """
    import skquant.opt

    params.update(
        {
            "budget": maxfev,
            "options": {
                "stencil": 0,  # stencil = 1 denotes FFD, stencil = 0 denotes CFD
            },
        }
    )
    if f_target:
        params["options"]["target"] = f_target
    # must specify bounds to use IMFLT solver, may cause problems when optimizing an unconstrained problem.
    if prob.sol_info and prob.sol_info.get("xopt", None) is not None:
        xopt_bound = 10 * max(1.0, np.max(np.abs(prob.sol_info["xopt"])))
    else:
        xopt_bound = 10
    bounds = (-xopt_bound, xopt_bound)
    bounds_pair = np.hstack(
        (np.ones((prob.dim, 1)) * bounds[0], np.ones((prob.dim, 1)) * bounds[1])
    )
    result, histout = skquant.opt.minimize(
        prob.fun_eval, prob.x0, bounds_pair, method='imfil', **params
    )
    res = OptimizeResult(
        x=result.optpar,
        fun=result.optval,
        nit=histout.shape[0],
        nfev=prob.eval_count,
    )
    solver_info = {"Solver Support Stopping Criteria": True}
    return res, solver_info


@wrap_dfo_solver("spsa")
def run_spsa_on_prob(
    prob: PSProblem, maxfev: int, params: Dict[Any, Any] = {}, **kwargs
) -> Tuple[OptimizeResult, Optional[Line2D]]:
    import spsa

    warnings.warn(
        "spsa does not support assigning the maximum number of function evaluations, "
        "the value of \"int(maxfev / 6)\" will be treated as the maximum number of iterations."
    )
    params.update({"iterations": int(maxfev / 8)})
    xopt = spsa.minimize(prob.fun_eval, prob.x0, **params)
    k_min = np.argmin(prob.history_fun[-1])
    res = OptimizeResult(
        x=xopt,
        fun=prob.history_fun[-1, k_min],
        nfev=prob.eval_count,
    )
    solver_info = {"Solver Support Stopping Criteria": False}
    return res, solver_info


@wrap_dfo_solver("bobyqa")
def run_bobyqa_on_prob(
    prob: PSProblem,
    maxfev: int,
    f_target: Optional[float] = None,
    params: Dict[Any, Any] = {},
    **kwargs,
) -> Tuple[OptimizeResult, Optional[Line2D]]:
    import nlopt

    prob.clear()
    params.update({"maxfun": maxfev})

    def FunAndGrad(x, grad):
        if grad.size > 0:
            pass
        return prob.fun_eval(x)

    nlopt_opt = nlopt.opt(nlopt.LN_BOBYQA, prob.dim)
    nlopt_opt.set_min_objective(FunAndGrad)
    if "ftol_abs" in params:
        nlopt_opt.set_ftol_abs(params["ftol_abs"])
    if "ftol_rel" in params:
        nlopt_opt.set_ftol_rel(params["ftol_rel"])
    if "stopval" in params:
        nlopt_opt.set_stopval(params["stopval"])
    if f_target:  # f_target overrides params["stopval"]
        nlopt_opt.set_stopval(f_target)
    if "xtol_rel" in params:
        nlopt_opt.set_xtol_rel(params["xtol_rel"])
    if "xtol_abs" in params:
        nlopt_opt.set_xtol_abs(params["xtol_abs"])
    if maxfev:
        nlopt_opt.set_maxeval(maxfev)

    try:
        xopt = nlopt_opt.optimize(prob.x0)
        minf = nlopt_opt.last_optimum_value()
        nfev = nlopt_opt.get_numevals()
        res = OptimizeResult(x=xopt, fun=minf, nfev=nfev)
    except Exception as _:
        res = OptimizeResult(x=None, fun=np.min(prob.history_fun[-1]), nfev=prob.eval_count)

    solver_info = {"Solver Support Stopping Criteria": True}
    return res, solver_info


@wrap_dfo_solver("newuoa")
def run_newuoa_on_prob(
    prob: PSProblem,
    maxfev: int,
    f_target: Optional[float] = None,
    params: Dict[Any, Any] = {},
    **kwargs,
) -> Tuple[ScipyOptimizeResult, Optional[Line2D]]:
    import nlopt

    prob.clear()
    params.update({"maxfun": maxfev})

    def FunAndGrad(x, grad):
        if grad.size > 0:
            pass
        return prob.fun_eval(x)

    nlopt_opt = nlopt.opt(nlopt.LN_NEWUOA, prob.dim)
    nlopt_opt.set_min_objective(FunAndGrad)
    if "ftol_abs" in params:
        nlopt_opt.set_ftol_abs(params["ftol_abs"])
    if "ftol_rel" in params:
        nlopt_opt.set_ftol_rel(params["ftol_rel"])
    if "stopval" in params:
        nlopt_opt.set_stopval(params["stopval"])
    if f_target:  # f_target overrides params["stopval"]
        nlopt_opt.set_stopval(f_target)
    if "xtol_rel" in params:
        nlopt_opt.set_xtol_rel(params["xtol_rel"])
    if "xtol_abs" in params:
        nlopt_opt.set_xtol_abs(params["xtol_abs"])
    if maxfev:
        nlopt_opt.set_maxeval(maxfev)

    try:
        xopt = nlopt_opt.optimize(prob.x0)
        minf = nlopt_opt.last_optimum_value()
        nfev = nlopt_opt.get_numevals()
        res = OptimizeResult(x=xopt, fun=minf, nfev=nfev)
    except Exception as _:
        res = OptimizeResult(x=None, fun=np.min(prob.history_fun[-1]), nfev=prob.eval_count)

    solver_info = {"Solver Support Stopping Criteria": True}
    return res, solver_info


@wrap_dfo_solver("cobyla")
def run_cobyla_on_prob(
    prob: PSProblem,
    maxfev: int,
    f_target: Optional[float] = None,
    params: Dict[Any, Any] = {},
    **kwargs,
) -> Tuple[ScipyOptimizeResult, Optional[Line2D]]:
    import nlopt

    prob.clear()
    params.update({"maxfun": maxfev})

    def FunAndGrad(x, grad):
        if grad.size > 0:
            pass
        return prob.fun_eval(x)

    nlopt_opt = nlopt.opt(nlopt.LN_COBYLA, prob.dim)
    nlopt_opt.set_min_objective(FunAndGrad)
    if "ftol_abs" in params:
        nlopt_opt.set_ftol_abs(params["ftol_abs"])
    if "ftol_rel" in params:
        nlopt_opt.set_ftol_rel(params["ftol_rel"])
    if "stopval" in params:
        nlopt_opt.set_stopval(params["stopval"])
    if f_target:  # f_target overrides params["stopval"]
        nlopt_opt.set_stopval(f_target)
    if "xtol_rel" in params:
        nlopt_opt.set_xtol_rel(params["xtol_rel"])
    if "xtol_abs" in params:
        nlopt_opt.set_xtol_abs(params["xtol_abs"])
    if maxfev:
        nlopt_opt.set_maxeval(maxfev)

    try:
        xopt = nlopt_opt.optimize(prob.x0)
        minf = nlopt_opt.last_optimum_value()
        nfev = nlopt_opt.get_numevals()
        res = OptimizeResult(x=xopt, fun=minf, nfev=nfev)
    except Exception as _:
        res = OptimizeResult(x=None, fun=np.min(prob.history_fun[-1]), nfev=prob.eval_count)
    solver_info = {"Solver Support Stopping Criteria": True}
    return res, solver_info


@wrap_dfo_solver("cma-es")
def run_cmaes_on_prob(
    prob: PSProblem,
    maxfev: int,
    f_target: Optional[float] = None,
    callback: Optional[callable] = None,
    disp: bool = False,
    params: Dict[Any, Any] = {},
    **kwargs,
) -> Tuple[ScipyOptimizeResult, Optional[Line2D]]:
    import cma

    if "sigma0" not in params or ("sigma0" in params and not params["sigma0"]):
        params["sigma0"] = 1
    es = cma.CMAEvolutionStrategy(prob.x0, **params)
    if "options" in params:
        params["options"].update({"popsize": cma.popsize})
    iters = 0
    go_on_flag = True
    while not es.stop() and go_on_flag:
        solutions = es.ask()
        fvals_list = []
        for sol in solutions:
            if prob.eval_count >= maxfev:
                go_on_flag = False
                break
            fvals_list.append(prob.fun_eval(sol))
        if not go_on_flag:
            break
        es.tell(solutions, fvals_list)
        if f_target:
            if es.best.f <= f_target:
                go_on_flag = False
        if callback is not None:
            state = OptimizeResult(x=es.best.x, fun=es.best.f, nfev=prob.eval_count, nit=iters)
            if callback(state):
                go_on_flag = False
        es.logger.add()  # write data to disc to be plotted
        if disp:
            es.disp()
        iters += 1
    res = OptimizeResult(x=es.best.x, fun=es.best.f, nfev=prob.eval_count, nit=iters)
    solver_info = {
        "Solver Support Stopping Criteria": True,
    }
    return res, solver_info


@wrap_dfo_solver("cobyqa")
def run_my_cobyqa_on_prob(
    prob: PSProblem,
    maxfev: int,
    f_target: Optional[float] = None,
    disp: Optional[bool] = None,
    params: Dict[Any, Any] = {},
    **kwargs,
) -> Tuple[ScipyOptimizeResult, Optional[Line2D]]:
    import cobyqa

    if "options" not in params:
        params["options"] = {}
    params["options"].update({"maxfev": maxfev})
    if disp is not None:
        params["options"].update({"disp": disp})
    if f_target is not None:
        params["options"].update({"target": f_target})
    res = cobyqa.minimize(prob.fun_eval, prob.x0, **params)
    try:
        res1 = OptimizeResult(
            message=res.message,
            success=res.success,
            status=res.status,
            x=res.x,
            fun=res.fun,
            jac=res.jac if "jac" in res else None,
            nfev=res.nfev,
            nit=res.nit,
        )
    except Exception as _:
        res1 = res
    solver_info = {
        "Solver Support Stopping Criteria": True,
    }
    return res1, solver_info


@wrap_dfo_solver("gsls")
def run_gsls_on_prob(
    prob: PSProblem,
    maxfev: int,
    disp: Optional[bool] = None,
    params: Dict[Any, Any] = {},
    **kwargs,
) -> Tuple[ScipyOptimizeResult, Optional[Line2D]]:
    # see also: https://qiskit-community.github.io/qiskit-algorithms/apidocs/qiskit_algorithms.optimizers.html
    from qiskit_algorithms.optimizers import GSLS

    if disp is not None:
        params.update({"disp": disp})

    gsls = GSLS(max_eval=maxfev, **params)

    res = gsls.minimize(prob.fun_eval, prob.x0)
    solver_info = {
        "Solver Support Stopping Criteria": False,
    }
    return res, solver_info


@wrap_dfo_solver("umda")
def run_gsls_on_prob(
    prob: PSProblem,
    maxfev: int,
    callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    disp: Optional[bool] = None,
    params: Dict[Any, Any] = {},
    **kwargs,
) -> Tuple[ScipyOptimizeResult, Optional[Line2D]]:
    from qiskit_algorithms.optimizers import UMDA

    if disp is not None:
        params.update({"disp": disp})
    if "size_gen" not in params:
        params["size_gen"] = 20
    size_gen = params["size_gen"]

    warnings.warn(
        "qiskit_algorithms.optimizers.UMDA does not support assigning the maximum number of function evaluations, "
        f"the value of \"int(maxfev / {size_gen})\" will be treated as the maximum number of iterations."
    )
    umda = UMDA(maxiter=int(maxfev / size_gen), callback=callback, **params)

    res = umda.minimize(prob.fun_eval, prob.x0)
    solver_info = {
        "Solver Support Stopping Criteria": False,
    }
    return res, solver_info
