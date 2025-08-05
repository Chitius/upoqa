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
from copy import deepcopy
import numpy as np
import time
import os
import traceback
from typing import Dict, Any, Literal, Tuple, Optional, Callable, Union, List
from upoqa.problems import *
from upoqa.utils import OptimizeResult
from matplotlib.lines import Line2D
from itertools import cycle
from scipy.optimize import minimize as sp_minimize
import warnings

PLOT_LINE_FMT_CYCLER = iter(cycle(['-', '--', ':', '-.']))
PLOT_MARKER_FMT_CYCLER = iter(cycle(['o', 's', '^', 'D', '+', '*']))


def param_map_of_at_50(name: str):
    big_prob = [
        'CYCLOOCFLS',
        'CYCLOOCTLS',
        'DIXMAANB',
        "DIXMAANC",
        "DIXMAANE1",
        "DIXMAANF",
        "DIXMAANJ",
        "DIXMAANK",
        "DIXMAANL",
        "DTOC2",
        "DTOC3",
        "HAGER2",
        "HAGER3",
        "RAYBENDL",
    ]

    too_big_prob = [
        "DTOC2",
    ]

    if name not in big_prob:
        n = 50
    elif name in too_big_prob:
        n = 15
    else:
        n = 30

    return n


def configs_and_prob_suffix_for_wtp_set_v1(seed: int = 2333, calc_mat: bool = False):
    if isinstance(seed, int):
        np.random.seed(seed)

    default_n_dims = [10, 25, 50]
    default_p_dims = [2, 4, 6, 8]
    min_and_max_eig = (0.1, 10)
    min_eig, max_eig = min_and_max_eig[0], min_and_max_eig[1]

    for n in default_n_dims:
        eigs = np.linspace(min_eig, max_eig, n)
        for p in default_p_dims:
            A = None
            if calc_mat:
                A = rand_matrix_gen(n, min_and_max_eig[0], min_and_max_eig[1])
            rreg = abs(eigs[p - 1]) * p
            yield (n, p, rreg, A), f"({n}-{p})"


class ProblemSet:

    default_problem_set = "s2mpj_prob_set_v1"

    PROBLEM_SETS = {
        "s2mpj_prob_set_v1": {
            # number of prob: 86,
            # should filter out 54-th prob "MINSURF" afterwards, where all solvers
            # start from 1.0000000000000007 and end with 1.0000000000000007, meaningless
            "path": "./s2mpj_prob_set_v1.npy",
            "noisy": False,
            "noise_free_set": None,
            "param_map": param_map_of_at_50,
        },
        "wtp_set_v1": {
            "path": None,
            "noisy": False,
            "noise_free_set": None,
            "param_map": None,
        },
    }

    HOUSE_MADE_INDEX_START = 10000

    WTRAPNT_INDEX_START = 20000

    HOUSE_MADE_PROBS_INFO = [
        {
            "name": "WTRAPNT",
            "has_zero_fopt": False,
            "problem_class": "WeightedTracePenaltyProb",
        },
        {
            "name": "BDQUADF",
            "has_zero_fopt": True,
            "problem_class": "BlockDiagQuadForm",
        },
        {
            "name": "BDBQUADF",
            "has_zero_fopt": False,
            "problem_class": "BlockDiagBiquadForm",
        },
    ]

    def __init__(self, load_set: Optional[str] = None):
        self.problem_set_info = None
        self._valid_problem_idx = []
        self.enabled_problem_set = ""
        load_set = load_set or ProblemSet.default_problem_set
        self.load_problem_set(load_set)

    def get_info(self, set_name: Optional[str] = None):
        if set_name is None:
            return self.problem_set_info
        else:
            assert set_name in self.available_problem_sets, "unknown test problem set!"
            prob_set_path = ProblemSet.PROBLEM_SETS[set_name]["path"]
            if prob_set_path is not None:
                return np.load(prob_set_path, allow_pickle=True)
            else:
                return None

    def results_dir(self, set_name: Optional[str] = None):
        set_name = self.enabled_problem_set if set_name is None else set_name
        return "./results_" + set_name + "/"

    def valid_problem_idx(self, set_name: Optional[str] = None):
        if set_name is None:
            return deepcopy(self._valid_problem_idx)
        else:
            prob_set_info = self.get_info(set_name)
            if prob_set_info is not None:
                return [x for x in range(len(prob_set_info))]
            elif set_name.startswith("wtp_set"):
                return [
                    (ProblemSet.WTRAPNT_INDEX_START + x)
                    for x in range(len(list(configs_and_prob_suffix_for_wtp_set_v1())))
                ]
            else:
                return []

    @property
    def available_problem_sets(self):
        return list(ProblemSet.PROBLEM_SETS.keys())

    def load_problem_set(self, set_name: str):
        try:
            assert set_name in self.available_problem_sets, "unknown test problem set!"
            self.enabled_problem_set = set_name
            self.problem_set_info = self.get_info(set_name)
            self._valid_problem_idx = self.valid_problem_idx(set_name)
        except Exception as e:
            print(f"Failed to load {set_name}, error: {e}\n\n{traceback.format_exc()}")

        # self._valid_problem_idx += [
        #     (ProblemSet.HOUSE_MADE_INDEX_START + x) for x in range(len(ProblemSet.HOUSE_MADE_PROBS_INFO))
        # ]
        self._valid_problem_idx.sort()

        return self.get_info()


problem_set = ProblemSet()


def _remove_space_in_solver_name(solver_name: str) -> str:
    return solver_name.replace(" ", "_")


def clear_results(
    solver: str,
    identifier: Optional[str] = None,
    from_problem_set_name: Optional[str] = None,
    verbose: bool = False,
):
    # clear problem_set.results_dir
    try:
        result_dir = problem_set.results_dir(from_problem_set_name)
        if (not os.path.exists(result_dir)) or (not os.path.isdir(result_dir)):
            print(f"The folder '{result_dir}' does not exist or is already empty.")
            return
        dirs_that_has_target_file, dirs_that_has_target_figs = [], []
        probs = set()
        for problem_idx in problem_set.valid_problem_idx(from_problem_set_name):
            info, info_path, fig_path = get_run_from_local_cache(
                solver=solver,
                problem_idx=problem_idx,
                identifier=identifier,
                from_problem_set_name=from_problem_set_name,
                do_warn=False,
            )
            if info is None:
                continue
            # 如果 info_path 存在，则加入 dirs_that_has_target_file
            if os.path.exists(info_path):
                dirs_that_has_target_file.append(info_path)
                dirs_that_has_target_figs.append(fig_path)
                probs.add(problem_idx)

        number_of_files_to_delete = len(dirs_that_has_target_file)

        if verbose:
            print(f"files to be deleted: {dirs_that_has_target_file}")
            print(f"figs to be deleted: {dirs_that_has_target_figs}")
            time.sleep(0.5)

        if number_of_files_to_delete:
            go_on = (
                input(
                    f"The results folder contain at least {number_of_files_to_delete} "
                    f"lastest runs of {len(probs)} problems"
                    + f" for solver {solver}"
                    + (f" with identifier {identifier}" if identifier else "")
                    + ".\n"
                    + "Do you want to delete all of them? [y/n]"
                ).lower()
                == "y"
            )

            if go_on:
                # delete all the files contained in dirs_that_has_target_file
                for file_path in dirs_that_has_target_file:
                    try:
                        os.remove(file_path)
                    except FileNotFoundError:
                        pass
                for fig_path in dirs_that_has_target_figs:
                    try:
                        os.remove(fig_path)
                    except FileNotFoundError:
                        pass
                print(
                    f"Deletion done (you may need to rerun this command for multiple times to delete all files)."
                )
        else:
            print(f"All clear, nothing to delete.")
    except Exception as e:
        print(f"Error: {e}")


def _get_default_file_dir_and_name(
    prob: PSProblem,
    solver: Optional[str] = None,
    prob_suffix: str = "",
    from_problem_set_name: Optional[str] = None,
    create_dir: bool = True,
) -> Tuple[str, str, str]:
    """
    example:
    >>> _get_default_file_dir_and_name(get_noise_free_ps_problem(17, params = {"n": 20}))
    ('./results/SPARSINE/logs', './results/SPARSINE/figs', None)

    >>> _get_default_file_dir_and_name(get_noise_free_ps_problem(17, params = {"n": 20}), "bobyqa")
    ('./results/SPARSINE/logs', './results/SPARSINE/figs', 'bobyqa@SPARSINE&n=20&20240905_1640')

    >>> _get_default_file_dir_and_name(get_noisy_ps_problem(17, params = {"n": 20}, noise_dist = "uni"), "bobyqa")
    ('./results/SPARSINE-n/logs', './results/SPARSINE-n/figs', 'bobyqa@SPARSINE&n=20&noise_dist=uni&20240905_1640')
    """
    file_dir_prefix = problem_set.results_dir(from_problem_set_name)
    if create_dir and not os.path.exists(file_dir_prefix):
        os.makedirs(file_dir_prefix)

    file_dir_prefix += f"{prob.name}" + prob_suffix
    is_noisy = prob.meta_info.get("noise_info", {}).get("is_noisy", False)
    if is_noisy:
        file_dir_prefix += "-n"
    file_dir_prefix += "/"
    if create_dir and not os.path.exists(file_dir_prefix):
        os.makedirs(file_dir_prefix)

    file_name = None
    if solver and isinstance(solver, str):
        solver = _remove_space_in_solver_name(solver)
        file_name = f"{solver}@{prob.name}" + prob_suffix + "&"
        params_into_path = {}
        for key, value in prob.meta_info["init_params"].items():
            if isinstance(
                value,
                (int, float, np.float32, np.float16, np.float64, bool, str, np.bool_),
            ):
                params_into_path[key] = value
        for key, value in params_into_path.items():
            file_name += f"{key}={value}&"
        if is_noisy:
            noise_info = prob.meta_info['noise_info']
            if "noise_dist" in noise_info:
                file_name += f"noise_dist={prob.meta_info['noise_info']['noise_dist']}&"
            if "noise_scale" in noise_info:
                file_name += f"noise_scale={prob.meta_info['noise_info']['noise_scale']}&"
            if "noise_type" in noise_info:
                file_name += f"noise_type={prob.meta_info['noise_info']['noise_type']}&"
        file_name += time.strftime('%Y%m%d_%H%M')

    file_dir_log = file_dir_prefix + "logs/"
    file_dir_fig = file_dir_prefix + "figs/"
    if create_dir:
        if not os.path.exists(file_dir_log):
            os.makedirs(file_dir_log)
        if not os.path.exists(file_dir_fig):
            os.makedirs(file_dir_fig)
    return file_dir_log, file_dir_fig, file_name


def get_run_from_local_cache(
    solver: str,
    problem_idx: Optional[int] = None,
    problem_name: Optional[str] = None,
    identifier: Optional[str] = None,
    prob_suffix: str = "",
    from_problem_set_name: Optional[str] = None,
    is_noisy: bool = False,
    do_warn: bool = True,
) -> Tuple[Optional[Dict], str, str]:
    """
    Get the run result from the local cache. (lasted by default)
    """
    if problem_idx is not None and isinstance(problem_idx, int) and problem_idx >= 0:
        problem_name = prob_idx_2_prob_info_and_source(problem_idx, from_problem_set_name)[0][
            "name"
        ]
    elif problem_name is None:
        raise ValueError("problem_idx and problem_name cannot be both None.")
    tmp_prob = PSProblem()
    tmp_prob.update_meta_info(
        {
            "name": problem_name,
            "problem_idx": problem_idx,
            "noise_info": {
                "is_noisy": is_noisy,
            },
        }
    )
    log_path, fig_path, _ = _get_default_file_dir_and_name(
        tmp_prob,
        prob_suffix=prob_suffix,
        from_problem_set_name=from_problem_set_name,
        create_dir=False,
    )
    solver = _remove_space_in_solver_name(solver)
    if not os.path.exists(log_path):
        if do_warn:
            warnings.warn(
                f"No log file found for solver {solver} on problem {problem_name} as the log path {log_path} does not yet exist.",
                stacklevel=2,
            )
        return None, None, None
    elif not os.path.exists(fig_path):
        if do_warn:
            warnings.warn(
                f"No figure file found for solver {solver} on problem {problem_name} as the fig path {fig_path} does not yet exist.",
                stacklevel=2,
            )
        return None, None, None
    else:
        log_files_in_dir = os.listdir(log_path)
        log_files = [
            f for f in log_files_in_dir if f.startswith(solver + "@") and f.endswith('.npy')
        ]
        if not log_files:
            if do_warn:
                warnings.warn(
                    f"No log file found for solver {solver} on problem {problem_name} as the log dir {log_path} is empty.",
                    stacklevel=2,
                )
            return None, None, None

        fig_files_in_dir = os.listdir(fig_path)
        fig_files = [
            f for f in fig_files_in_dir if f.startswith(solver + "@") and f.endswith('.png')
        ]
        if not fig_files:
            if do_warn:
                warnings.warn(
                    f"No figure file found for solver {solver} on problem {problem_name} as the fig dir {fig_path} is empty.",
                    stacklevel=2,
                )
            return None, None, None

        # Sort logs by date first, then by time
        sorted_logs = sorted(log_files, key=lambda x: x.split('_')[-2:], reverse=True)

        if identifier is None:
            curr_log = sorted_logs[0]
            log_file_path = os.path.join(log_path, curr_log)
            fig_file_path = os.path.join(
                fig_path, curr_log.replace(".npy", ".png").replace("/logs", "/figs")
            )
            log_info = np.load(log_file_path, allow_pickle=True).item()
            return log_info, log_file_path, fig_file_path

        for curr_log in sorted_logs:
            log_file_path = os.path.join(log_path, curr_log)
            fig_file_path = os.path.join(
                fig_path, curr_log.replace(".npy", ".png").replace("/logs", "/figs")
            )
            log_info = np.load(log_file_path, allow_pickle=True).item()
            if (identifier) and ('meta_info' in log_info):
                if "Identifier" in log_info['meta_info']:
                    if log_info['meta_info']["Identifier"] == identifier:
                        return log_info, log_file_path, fig_file_path
                elif identifier == solver:
                    return log_info, log_file_path, fig_file_path

        if do_warn:
            if identifier:
                warnings.warn(
                    f"No log file found for solver {solver} on problem {problem_name} with identifier {identifier}!",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"No log file found for solver {solver} on problem {problem_name}!",
                    stacklevel=2,
                )
        return None, None, None


def get_evals_of_solver_on_prob(
    info: Optional[dict] = None,  # return value of get_run_from_local_cache()
    ftol_rel: Optional[List[float]] = None,
    fopt: Optional[float] = None,
    fx0: Optional[float] = None,
    nfev_mode: Literal["wst", "avg"] = "wst",
) -> Optional[List[Tuple[float, Optional[int]]]]:

    if ftol_rel is None:
        ftol_rel = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

    evals = []

    if info is None:
        return None

    history_fvals = info["optimize"]["History of Objective Function Values"]

    if fopt is None:
        fopt = info["optimize"]["Theoretical Optimal Objective Value"]

    if fx0 is None:
        if "Objective Function Value at x0" in info["meta_info"]:
            fx0 = info["meta_info"]["Objective Function Value at x0"]
        else:
            fx0 = fopt + max(1.0, abs(fopt))

    if len(history_fvals) == 2 or nfev_mode == "wst":
        evals_array = history_fvals[0]
    else:
        evals_array = history_fvals[1]

    for ftol in ftol_rel:
        f_target = fopt + ftol * abs(fx0 - fopt)
        solved_idx = -1
        for idx in range(len(history_fvals[-1])):
            if history_fvals[-1][idx] <= f_target:
                solved_idx = idx
                break
        evals.append((ftol, evals_array[solved_idx] if solved_idx >= 0 else None))

    return evals


def _get_meta_info_template(
    prob: PSProblem,
    solver: str,
    from_problem_set_name: Optional[str] = None,
) -> dict:
    if from_problem_set_name is None:
        from_problem_set_name = problem_set.enabled_problem_set
    ele_dims = [len(coord) for coord in prob.coords.values()]
    sum_ele_dim = sum(ele_dims)
    max_ele_dim = max(ele_dims)
    min_ele_dim = min(ele_dims)
    avg_ele_dim = sum_ele_dim / len(prob.coords)
    info = {
        "meta_info": {
            "Solver Name": solver,
            "Problem Name": prob.name,
            "Problem Set Name": from_problem_set_name,
            "Problem Index": prob.meta_info.get("problem_idx", None),
            "Problem Source": prob.meta_info.get("source", "unknown"),
            "Problem Params": prob.meta_info.get("init_params", None),
            "Noise Settings": prob.meta_info.get("noise_info", None),
            "Average Elemental Dimension": avg_ele_dim,
            "Max Elemental Dimension": max_ele_dim,
            "Min Elemental Dimension": min_ele_dim,
            "Problem Dimension": prob.dim,
        },
    }
    return info


def _save_log_for_a_run(
    prob: PSProblem,
    res: OptimizeResult,
    solver_name: str,
    prob_suffix: str = "",
    from_problem_set_name: Optional[str] = None,
    save_log: Union[bool, str] = False,
    extra_info: Optional[dict] = None,
) -> None:
    if save_log:
        info = {
            "optimize": {
                "Stopping Message": res.get("message", None),
                "Solution Point": res.get("x", None),
                "Number of Iterations": res.get("nit", None),
            },
            "meta_info": {},
        }
        if isinstance(extra_info, dict):
            if 'meta_info' in extra_info:
                info['meta_info'].update(extra_info.get('meta_info', {}))
            if 'optimize' in info:
                info['optimize'].update(extra_info.get('optimize', {}))
        if not isinstance(save_log, str):
            log_dir, _, file_name = _get_default_file_dir_and_name(
                prob,
                solver_name,
                prob_suffix=prob_suffix,
                create_dir=True,
                from_problem_set_name=from_problem_set_name,
            )
            save_log = log_dir + file_name + ".npy"
        prob.save_history(save_log, info=info)


def _save_or_print_figure_for_a_run(
    prob: PSProblem,
    label: str,
    solver_name: str,
    plot_fig: bool = True,
    show_fig: bool = False,
    prob_suffix: str = "",
    from_problem_set_name: Optional[str] = None,
    fmt: Optional[str] = None,
    save_fig: Union[bool, str] = False,
) -> Optional[Line2D]:
    if plot_fig or show_fig or save_fig:
        if fmt is None:
            fmt = next(PLOT_LINE_FMT_CYCLER)
        line = prob.disp_fun_history(label=label, fmt=fmt, show_fig=show_fig)
        if save_fig:
            if not isinstance(save_fig, str):
                _, fig_dir, file_name = _get_default_file_dir_and_name(
                    prob,
                    solver_name,
                    create_dir=True,
                    prob_suffix=prob_suffix,
                    from_problem_set_name=from_problem_set_name,
                )
                save_fig = fig_dir + file_name + ".png"
            prob.history_plot_fig.savefig(save_fig)
        return line
    return None


def prob_idx_2_prob_info_and_source(
    problem_idx: int,
    from_problem_set_name: Optional[str] = None,
) -> Tuple[dict, Literal["s2mpj", "house-made"]]:
    """
    - 0 ~ ProblemSet.HOUSE_MADE_INDEX_START: The problem_idx-th problem among the selected from the s2mpj problem set

    - >= ProblemSet.HOUSE_MADE_INDEX_START: House-made problems.
    """
    if problem_idx < ProblemSet.HOUSE_MADE_INDEX_START:
        if problem_idx not in problem_set.valid_problem_idx(from_problem_set_name):
            raise ValueError(f"problem_idx {problem_idx} is not available.")
        return problem_set.get_info()[problem_idx], "s2mpj"
    elif problem_idx < ProblemSet.WTRAPNT_INDEX_START:
        return (
            ProblemSet.HOUSE_MADE_PROBS_INFO[problem_idx - ProblemSet.HOUSE_MADE_INDEX_START],
            "house-made",
        )
    else:
        return ProblemSet.HOUSE_MADE_PROBS_INFO[0], "house-made"


def prob_name_2_prob_idx(
    problem_name: str,
    from_problem_set_name: Optional[str] = None,
) -> int:
    for i, prob in enumerate(problem_set.get_info(from_problem_set_name)):
        if prob["name"] == problem_name:
            return i
    for i, prob in enumerate(ProblemSet.HOUSE_MADE_PROBS_INFO):
        if prob["name"] == problem_name:
            return i + ProblemSet.HOUSE_MADE_INDEX_START
    return -1


def get_noise_free_ps_problem(
    problem_idx: Optional[int] = None,
    problem_name: Optional[str] = None,
    from_problem_set_name: Optional[str] = None,
    params: Dict[Any, Any] = {},
    high_precision_fopt: bool = False,
    need_sol: bool = True,
) -> PSProblem:
    if problem_idx is None and problem_name is None:
        raise ValueError("Either problem_idx or problem_name should be specified.")
    if problem_idx is None:
        problem_idx = prob_name_2_prob_idx(problem_name, from_problem_set_name)
    prob_info, prob_source = prob_idx_2_prob_info_and_source(problem_idx, from_problem_set_name)
    if prob_source == "s2mpj":
        s2mpj_prob_name = prob_info["name"]
        prob = S2MPJPSProblem(s2mpj_prob_name, **params)
        if not prob_info["has_zero_fopt"] and need_sol:
            if high_precision_fopt:
                res1 = sp_minimize(
                    prob.s2mpj_prob.fx,
                    prob.x0,
                    # method = "BFGS",
                    method="newton-cg",
                    jac=lambda x: prob.s2mpj_prob.fgx(x)[1].squeeze(),
                    hess=lambda x: prob.s2mpj_prob.fgHx(x)[2].toarray(),
                    options={'maxiter': 1e6},
                    tol=1e-18,
                )
                res2 = sp_minimize(
                    prob.s2mpj_prob.fx,
                    prob.x0,
                    method="BFGS",
                    jac=lambda x: prob.s2mpj_prob.fgx(x)[1].squeeze(),
                    options={'maxiter': 1e6},
                    tol=1e-18,
                )
                if res1.fun < res2.fun:
                    res = res1
                else:
                    res = res2
            else:
                res = sp_minimize(
                    prob.s2mpj_prob.fx,
                    prob.x0,
                    method="BFGS",
                    # method = "newton-cg",
                    jac=lambda x: prob.s2mpj_prob.fgx(x)[1].squeeze(),
                    # hess = lambda x: prob.s2mpj_prob.fgHx(x)[2].toarray(),
                    options={'maxiter': 1e5},
                    tol=1e-18,
                )
            fopt = res.fun
            prob.y_shift = -fopt
            prob.update_sol_info(xopt=res.x)
        else:
            fopt = 0.0

        prob.update_sol_info(fopt=fopt)
        prob.update_meta_info(
            {
                "name": prob_info["name"],
                "problem_idx": problem_idx,
                "source": prob_source,
                "init_params": params,
                "noise_info": {
                    "is_noisy": False,
                },
            }
        )
        return prob
    elif prob_source == "house-made":
        prob_class = eval(prob_info["problem_class"])
        assert issubclass(prob_class, PSProblem), "prob_class must be a subclass of PSProblem."
        prob = prob_class(**params)
        prob.update_meta_info(
            {
                "name": prob_info["name"],
                "problem_idx": problem_idx,
                "source": prob_source,
                "init_params": params,
                "noise_info": {
                    "is_noisy": False,
                },
            }
        )
        return prob
    else:
        raise ValueError("Invalid problem source: {}".format(prob_source))


def get_noisy_ps_problem(
    problem_idx: Optional[int] = None,
    problem_name: Optional[str] = None,
    from_problem_set_name: Optional[str] = None,
    params: Dict[Any, Any] = {},
    high_precision_fopt: bool = False,
    need_sol: bool = True,
    noise_type: Literal["add", "mult"] = "mult",
    noise_dist: Literal["gauss", "uni"] = "gauss",
    noise_scale: float = 1e-6,
    noise_wrapper: Optional[callable] = None,  # will override noise_type and noise_scale
) -> PSProblem:
    if problem_idx is None and problem_name is None:
        raise ValueError("Either problem_idx or problem_name should be specified.")
    if problem_idx is None:
        problem_idx = prob_name_2_prob_idx(problem_name, from_problem_set_name)
    prob = get_noise_free_ps_problem(
        problem_idx,
        params=params,
        from_problem_set_name=from_problem_set_name,
        high_precision_fopt=high_precision_fopt,
        need_sol=need_sol,
    )
    if noise_wrapper:
        assert callable(noise_wrapper), "noise_wrapper must be a callable."
        prob.set_noise_wrapper(noise_wrapper)
        prob.update_meta_info(
            {
                "noise_info": {
                    "is_noisy": True,
                    "noise_type": f"customized noise wrapper: {noise_wrapper.__name__}",
                }
            }
        )
    else:

        def _noise_wrapper(x: np.ndarray, *args, **kwargs) -> np.ndarray:
            if noise_type == "add":
                if noise_dist == "gauss":
                    return x + np.random.normal(scale=noise_scale)
                elif noise_dist == "uni":
                    return x + np.random.uniform(-noise_scale, noise_scale)
            elif noise_type == "mult":
                if noise_dist == "gauss":
                    return x * (1 + np.random.normal(scale=noise_scale))
                elif noise_dist == "uni":
                    return x * (1 + np.random.uniform(-noise_scale, noise_scale))
            else:
                raise ValueError("Invalid noise type: {}".format(noise_type))

        prob.set_noise_wrapper(_noise_wrapper)
        prob.update_meta_info(
            {
                "noise_info": {
                    "is_noisy": True,
                    "noise_type": noise_type,
                    "noise_scale": noise_scale,
                    "noise_dist": noise_dist,
                }
            }
        )
    return prob
