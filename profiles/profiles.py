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
from typing import Dict, Any, Literal, Tuple, Optional, Callable, Union, List
from upoqa.problems import *
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import IPython
from utils import *
from solvers import *


def plot_performance_profile_noise_free(
    solver_labels: List[str],  # ["upoqa",]
    solver_identifiers: List[str],  # [None,]
    plot_labels: Optional[List[str]] = None,
    plot_fig_and_ax: Optional[Tuple[Figure, Axes]] = None,
    problem_idx_range: Optional[List[int]] = None,
    nonplot_solver_labels: List[str] = [],  # ["upoqa",]
    ftol: float = 1e-6,
    fmt: Optional[str] = None,
    fig_size: Tuple[int] = (12, 6),
    grid_on: bool = True,
) -> Tuple[List[List[int]], Figure, Axes]:
    # score sum, components, raw comps and evals of upoqa and lbfgsb, avg ele dims, used ftol
    if problem_idx_range is None:
        problem_idx_range = [
            x for x in problem_set.valid_problem_idx() if x < ProblemSet.HOUSE_MADE_INDEX_START
        ]

    solver_num = len(solver_labels)
    problem_num = len(problem_idx_range)

    fopt_by_all_solvers = [np.inf for _ in problem_idx_range]
    fx0_by_all_solvers = [None for _ in problem_idx_range]
    converge_nfevs_on_prob = [[np.inf for _ in range(solver_num)] for _ in problem_idx_range]
    min_converge_nfev_on_prob = [np.inf for _ in problem_idx_range]
    relative_expenses_for_solver_on_prob = [
        [None for _ in problem_idx_range] for _ in range(solver_num)
    ]
    prob_dims = [None for _ in problem_idx_range]

    # get the fopt
    for i, problem_idx in enumerate(problem_idx_range):
        for solver_idx in range(solver_num):
            solver_label = solver_labels[solver_idx]
            solver_identifier = solver_identifiers[solver_idx]
            info, _, _ = get_run_from_local_cache(
                solver=solver_label, problem_idx=problem_idx, identifier=solver_identifier
            )
            if info is None:
                continue
            fopt_by_all_solvers[i] = min(
                fopt_by_all_solvers[i], info["optimize"]["Optimal Objective Value in History"]
            )
            if (
                "Objective Function Value at x0" in info["meta_info"]
                and info["meta_info"]["Objective Function Value at x0"] is not None
            ):
                fx0_by_all_solvers[i] = info["meta_info"]["Objective Function Value at x0"]
            if (
                "Problem Dimension" in info["meta_info"]
                and info["meta_info"]["Problem Dimension"] is not None
            ):
                prob_dims[i] = info["meta_info"]["Problem Dimension"]
        if fx0_by_all_solvers[i] is None:
            raise ValueError("fx0_by_all_solvers[i] is None!")

    for solver_idx in range(solver_num):
        solver_label = solver_labels[solver_idx]
        solver_identifier = solver_identifiers[solver_idx]
        for i, problem_idx in enumerate(problem_idx_range):
            info, _, _ = get_run_from_local_cache(
                solver=solver_label,
                problem_idx=problem_idx,
                identifier=solver_identifier,
                do_warn=False,
            )
            if info is None:
                continue
            ftol_reach_eval = get_evals_of_solver_on_prob(
                info,
                ftol_rel=[
                    ftol,
                ],
                fopt=fopt_by_all_solvers[i],
                fx0=fx0_by_all_solvers[i],
            )[0][1]
            if ftol_reach_eval is not None:
                converge_nfevs_on_prob[i][solver_idx] = ftol_reach_eval
                min_converge_nfev_on_prob[i] = min(min_converge_nfev_on_prob[i], ftol_reach_eval)

    for solver_idx in range(solver_num):
        for i, problem_idx in enumerate(problem_idx_range):
            relative_expenses_for_solver_on_prob[solver_idx][i] = (
                converge_nfevs_on_prob[i][solver_idx] / min_converge_nfev_on_prob[i]
            )

    start_x = np.ceil(
        np.log2(
            min(
                [
                    min(
                        [
                            1.0,
                        ]
                        + [y for y in x if y != np.inf]
                    )
                    for x in relative_expenses_for_solver_on_prob
                ]
            )
        )
    )
    end_x = np.ceil(
        np.log2(
            max(
                [
                    max(
                        [
                            0.0,
                        ]
                        + [y for y in x if y != np.inf]
                    )
                    for x in relative_expenses_for_solver_on_prob
                ]
            )
        )
    )

    for solver_idx in range(solver_num):
        solver_label = solver_labels[solver_idx]
        if solver_label in nonplot_solver_labels:
            continue
        solver_identifier = solver_identifiers[solver_idx]
        relative_expenses_for_solver = relative_expenses_for_solver_on_prob[solver_idx]
        relative_expenses_for_solver.sort()
        points_to_be_plotted = [
            [start_x, 0.0],
        ]

        for r_idx in range(problem_num):
            if relative_expenses_for_solver[r_idx] == np.inf:
                break
            if np.log2(relative_expenses_for_solver[r_idx]) == points_to_be_plotted[-1][0]:
                points_to_be_plotted[-1][1] += 1 / problem_num
            else:
                points_to_be_plotted.extend(
                    [
                        [np.log2(relative_expenses_for_solver[r_idx]), points_to_be_plotted[-1][1]],
                        [
                            np.log2(relative_expenses_for_solver[r_idx]),
                            points_to_be_plotted[-1][1] + 1 / problem_num,
                        ],
                    ]
                )

        if points_to_be_plotted[-1][0] < end_x:
            points_to_be_plotted.append([end_x, points_to_be_plotted[-1][1]])

        # points_to_be_plotted = points_to_be_plotted[1:]

        xs_to_be_plotted, ys_to_be_plotted = zip(*points_to_be_plotted)

        if plot_fig_and_ax is None or all(v is None for v in plot_fig_and_ax):
            plt.style.use('default')
            plot_fig_and_ax = plt.subplots()
            fig, ax = plot_fig_and_ax
            fig.set_size_inches(fig_size[0], fig_size[1])
            # ax.set_xscale('log')
            ax.set_xlabel(r'$\log_2(\alpha)$')
            ax.set_ylabel(r'Performance profiles $\rho_s(\alpha)$')
            if grid_on:
                ax.grid(True)
            else:
                ax.tick_params(
                    axis='both',
                    direction='in',
                    which='both',
                    bottom=True,
                    top=True,
                    left=True,
                    right=True,
                )

        fig, ax = plot_fig_and_ax
        plot_fmt = next(PLOT_LINE_FMT_CYCLER) if fmt is None else fmt

        if plot_labels and plot_labels[solver_idx] is not None:
            plot_label = plot_labels[solver_idx]
        else:
            plot_label = solver_identifier if solver_identifier else solver_label
        ax.plot(xs_to_be_plotted, ys_to_be_plotted, plot_fmt, label=plot_label)
        ax.legend()

    IPython.display.display(fig)
    plt.close(fig)  # prevent displaying twice in jupyter notebook

    return relative_expenses_for_solver_on_prob, fig, ax


def plot_data_profile_noise_free(
    solver_labels: List[str],  # ["upoqa",]
    solver_identifiers: List[str],  # [None,]
    plot_labels: Optional[List[str]] = None,
    plot_fig_and_ax: Optional[Tuple[Figure, Axes]] = None,
    problem_idx_range: Optional[List[int]] = None,
    ftol: float = 1e-6,
    fmt: Optional[str] = None,
    fig_size: Tuple[int] = (12, 6),
    grid_on: bool = True,
) -> Tuple[List[List[int]], Figure, Axes]:
    # score sum, components, raw comps and evals of upoqa and lbfgsb, avg ele dims, used ftol
    if problem_idx_range is None:
        problem_idx_range = [
            x for x in problem_set.valid_problem_idx() if x < ProblemSet.HOUSE_MADE_INDEX_START
        ]

    solver_num = len(solver_labels)
    problem_num = len(problem_idx_range)

    fopt_by_all_solvers = [np.inf for _ in problem_idx_range]
    fx0_by_all_solvers = [None for _ in problem_idx_range]
    converge_nfevs_on_prob = [[np.inf for _ in range(solver_num)] for _ in problem_idx_range]
    scaled_expenses_for_solver_on_prob = [dict() for _ in range(solver_num)]
    prob_dims = [None for _ in problem_idx_range]

    # get the fopt
    for i, problem_idx in enumerate(problem_idx_range):
        for solver_idx in range(solver_num):
            solver_label = solver_labels[solver_idx]
            solver_identifier = solver_identifiers[solver_idx]
            info, _, _ = get_run_from_local_cache(
                solver=solver_label, problem_idx=problem_idx, identifier=solver_identifier
            )
            if info is None:
                continue
            fopt_by_all_solvers[i] = min(
                fopt_by_all_solvers[i], info["optimize"]["Optimal Objective Value in History"]
            )
            if (
                "Objective Function Value at x0" in info["meta_info"]
                and info["meta_info"]["Objective Function Value at x0"] is not None
            ):
                fx0_by_all_solvers[i] = info["meta_info"]["Objective Function Value at x0"]
            if (
                "Problem Dimension" in info["meta_info"]
                and info["meta_info"]["Problem Dimension"] is not None
            ):
                prob_dims[i] = info["meta_info"]["Problem Dimension"]
        if fx0_by_all_solvers[i] is None:
            raise ValueError("fx0_by_all_solvers[i] is None!")

    for solver_idx in range(solver_num):
        solver_label = solver_labels[solver_idx]
        solver_identifier = solver_identifiers[solver_idx]
        for i, problem_idx in enumerate(problem_idx_range):
            info, _, _ = get_run_from_local_cache(
                solver=solver_label,
                problem_idx=problem_idx,
                identifier=solver_identifier,
                do_warn=False,
            )
            if info is None:
                continue
            ftol_reach_eval = get_evals_of_solver_on_prob(
                info,
                ftol_rel=[
                    ftol,
                ],
                fopt=fopt_by_all_solvers[i],
                fx0=fx0_by_all_solvers[i],
            )[0][1]
            if ftol_reach_eval is not None:
                converge_nfevs_on_prob[i][solver_idx] = ftol_reach_eval

    for solver_idx in range(solver_num):
        for i, problem_idx in enumerate(problem_idx_range):
            scaled_expenses_for_solver_on_prob[solver_idx][problem_idx] = converge_nfevs_on_prob[i][
                solver_idx
            ] / (1 + prob_dims[i])

    start_x = np.ceil(
        np.log2(
            min(
                [
                    min(
                        [
                            0.0,
                        ]
                        + [y for y in x.values() if y != np.inf]
                    )
                    for x in scaled_expenses_for_solver_on_prob
                ]
            )
        )
    )
    end_x = np.ceil(
        np.log2(
            max(
                [
                    max(
                        [
                            0.0,
                        ]
                        + [y for y in x.values() if y != np.inf]
                    )
                    for x in scaled_expenses_for_solver_on_prob
                ]
            )
        )
    )

    for solver_idx in range(solver_num):
        solver_label = solver_labels[solver_idx]
        solver_identifier = solver_identifiers[solver_idx]
        relative_expenses_for_solver = list(scaled_expenses_for_solver_on_prob[solver_idx].values())
        relative_expenses_for_solver.sort()
        points_to_be_plotted = [
            [start_x, 0.0],
        ]

        for r_idx in range(problem_num):
            if relative_expenses_for_solver[r_idx] == np.inf:
                break
            if np.log2(relative_expenses_for_solver[r_idx]) == points_to_be_plotted[-1][0]:
                points_to_be_plotted[-1][1] += 1 / problem_num
            else:
                points_to_be_plotted.extend(
                    [
                        [np.log2(relative_expenses_for_solver[r_idx]), points_to_be_plotted[-1][1]],
                        [
                            np.log2(relative_expenses_for_solver[r_idx]),
                            points_to_be_plotted[-1][1] + 1 / problem_num,
                        ],
                    ]
                )

        if points_to_be_plotted[-1][0] < end_x:
            points_to_be_plotted.append([end_x, points_to_be_plotted[-1][1]])

        xs_to_be_plotted, ys_to_be_plotted = zip(*points_to_be_plotted)

        if plot_fig_and_ax is None or all(v is None for v in plot_fig_and_ax):
            plt.style.use('default')
            plot_fig_and_ax = plt.subplots()
            fig, ax = plot_fig_and_ax
            fig.set_size_inches(fig_size[0], fig_size[1])
            # ax.set_xscale('log')
            ax.set_xlabel(r'$\log_2(\alpha)$')
            ax.set_ylabel(r'Data profiles $d_s(\alpha)$')
            if grid_on:
                ax.grid(True)
            else:
                ax.tick_params(
                    axis='both',
                    direction='in',
                    which='both',
                    bottom=True,
                    top=True,
                    left=True,
                    right=True,
                )

        fig, ax = plot_fig_and_ax
        plot_fmt = next(PLOT_LINE_FMT_CYCLER) if fmt is None else fmt

        if plot_labels and plot_labels[solver_idx] is not None:
            plot_label = plot_labels[solver_idx]
        else:
            plot_label = solver_identifier if solver_identifier else solver_label
        ax.plot(xs_to_be_plotted, ys_to_be_plotted, plot_fmt, label=plot_label)
        ax.legend()

    IPython.display.display(fig)
    plt.close(fig)  # prevent displaying twice in jupyter notebook

    return scaled_expenses_for_solver_on_prob, converge_nfevs_on_prob, fig, ax


def plot_speedup_profile_noise_free(
    solver_labels: List[str],  # [..., "upoqa (single)", "upoqa"]
    solver_identifiers: List[str],
    plot_fig_and_ax: Optional[Tuple[Figure, Axes]] = None,
    plot_label: Optional[str] = None,
    problem_idx_range: Optional[List[int]] = None,
    ftol: float = 1e-6,
    fmt: Optional[str] = None,
    fig_size: Tuple[int] = (18, 6),
    grid_on: bool = True,
) -> Tuple[List[List[int]], Figure, Axes]:
    # score sum, components, raw comps and evals of upoqa and lbfgsb, avg ele dims, used ftol
    if problem_idx_range is None:
        problem_idx_range = [
            x for x in problem_set.valid_problem_idx() if x < ProblemSet.HOUSE_MADE_INDEX_START
        ]

    solver_num = len(solver_labels)
    problem_num = len(problem_idx_range)

    fopt_by_all_solvers = [np.inf for _ in problem_idx_range]
    fx0_by_all_solvers = [None for _ in problem_idx_range]
    converge_nfevs_on_prob = [[np.inf for _ in range(solver_num)] for _ in problem_idx_range]
    rela_speedup_on_prob_pos, rela_speedup_on_prob_neg = [], []
    prob_dims = [None for _ in problem_idx_range]
    ele_avg_dims = [None for _ in problem_idx_range]
    ele_max_dims = [None for _ in problem_idx_range]
    # get the fopt

    for i, problem_idx in enumerate(problem_idx_range):

        for solver_idx in range(solver_num):
            solver_label = solver_labels[solver_idx]
            solver_identifier = solver_identifiers[solver_idx]
            info, _, _ = get_run_from_local_cache(
                solver=solver_label, problem_idx=problem_idx, identifier=solver_identifier
            )
            if info is None:
                continue
            fopt_by_all_solvers[i] = min(
                fopt_by_all_solvers[i], info["optimize"]["Optimal Objective Value in History"]
            )
            if (
                "Objective Function Value at x0" in info["meta_info"]
                and info["meta_info"]["Objective Function Value at x0"] is not None
            ):
                fx0_by_all_solvers[i] = info["meta_info"]["Objective Function Value at x0"]
            if (
                "Problem Dimension" in info["meta_info"]
                and info["meta_info"]["Problem Dimension"] is not None
            ):
                prob_dims[i] = info["meta_info"]["Problem Dimension"]
            if (
                "Average Elemental Dimension" in info["meta_info"]
                and info["meta_info"]["Average Elemental Dimension"] is not None
            ):
                ele_avg_dims[i] = info["meta_info"]["Average Elemental Dimension"]
            if (
                "Max Elemental Dimension" in info["meta_info"]
                and info["meta_info"]["Max Elemental Dimension"] is not None
            ):
                ele_max_dims[i] = info["meta_info"]["Max Elemental Dimension"]
        if fx0_by_all_solvers[i] is None:
            raise ValueError("fx0_by_all_solvers[i] is None!")

    for solver_idx in range(solver_num):
        solver_label = solver_labels[solver_idx]
        solver_identifier = solver_identifiers[solver_idx]
        for i, problem_idx in enumerate(problem_idx_range):
            info, _, _ = get_run_from_local_cache(
                solver=solver_label,
                problem_idx=problem_idx,
                identifier=solver_identifier,
                do_warn=False,
            )
            if info is None:
                continue
            ftol_reach_eval = get_evals_of_solver_on_prob(
                info,
                ftol_rel=[
                    ftol,
                ],
                fopt=fopt_by_all_solvers[i],
                fx0=fx0_by_all_solvers[i],
            )[0][1]
            if ftol_reach_eval is not None:
                converge_nfevs_on_prob[i][solver_idx] = ftol_reach_eval

    for i, problem_idx in enumerate(problem_idx_range):
        val = (converge_nfevs_on_prob[i][-2] / converge_nfevs_on_prob[i][-1]) / (
            prob_dims[i] / ele_max_dims[i]
        )
        if not np.isnan(val):
            if val >= 1:
                rela_speedup_on_prob_pos.append(val)
            else:
                rela_speedup_on_prob_neg.append(val)
        # if np.any(np.isnan(rela_speedup_on_prob[i])):
        #     raise ValueError

    # valid_prob_num = len(rela_speedup_on_prob)

    start_x = -3
    end_x = 3

    solver_label = solver_labels[solver_idx]
    solver_identifier = solver_identifiers[solver_idx]
    rela_speedup_on_prob_pos.sort()
    rela_speedup_on_prob_neg.sort(reverse=True)
    points_pos_to_be_plotted = [
        [0.0, 0.0],
    ]
    points_neg_to_be_plotted = [
        [0.0, 0.0],
    ]

    num_pos = len(rela_speedup_on_prob_pos)
    for r_idx in range(num_pos):
        # if np.log2(rela_speedup_on_prob[r_idx]) >= end_x:
        if rela_speedup_on_prob_pos[r_idx] == np.inf:
            break
        if min(np.log2(rela_speedup_on_prob_pos[r_idx]), end_x) <= points_pos_to_be_plotted[-1][0]:
            points_pos_to_be_plotted[-1][1] += 1 / problem_num
        else:
            points_pos_to_be_plotted.extend(
                [
                    [
                        min(np.log2(rela_speedup_on_prob_pos[r_idx]), end_x),
                        points_pos_to_be_plotted[-1][1],
                    ],
                    [
                        min(np.log2(rela_speedup_on_prob_pos[r_idx]), end_x),
                        points_pos_to_be_plotted[-1][1] + 1 / problem_num,
                    ],
                ]
            )

    # if points_pos_to_be_plotted[-1][0] < end_x:
    #     points_pos_to_be_plotted.append([end_x, points_pos_to_be_plotted[-1][1]])

    # log2(alpha) < 0 part
    num_neg = len(rela_speedup_on_prob_neg)
    for r_idx in range(num_neg):
        if rela_speedup_on_prob_neg[r_idx] == np.inf:
            break
        if (
            max(np.log2(rela_speedup_on_prob_neg[r_idx]), start_x)
            >= points_neg_to_be_plotted[-1][0]
        ):
            points_neg_to_be_plotted[-1][1] += 1 / problem_num
        else:
            points_neg_to_be_plotted.extend(
                [
                    [
                        max(np.log2(rela_speedup_on_prob_neg[r_idx]), start_x),
                        points_neg_to_be_plotted[-1][1],
                    ],
                    [
                        max(np.log2(rela_speedup_on_prob_neg[r_idx]), start_x),
                        points_neg_to_be_plotted[-1][1] + 1 / problem_num,
                    ],
                ]
            )

    # if points_neg_to_be_plotted[-1][0] < start_x:
    #     points_neg_to_be_plotted.append([start_x, points_neg_to_be_plotted[-1][1]])

    points_to_be_plotted = points_neg_to_be_plotted[::-1] + points_pos_to_be_plotted
    xs_to_be_plotted, ys_to_be_plotted = zip(*points_to_be_plotted)

    if plot_fig_and_ax is None or all(v is None for v in plot_fig_and_ax):
        plt.style.use('default')
        plot_fig_and_ax = plt.subplots()
        fig, ax = plot_fig_and_ax
        fig.set_size_inches(fig_size[0], fig_size[1])
        # ax.set_xscale('log')
        ax.set_xlabel(r'$\log_2(\alpha)$')
        ax.set_ylabel(r'Speed-up profile $\mathrm{su}(\alpha)$')
        if grid_on:
            ax.grid(True)
        else:
            ax.tick_params(
                axis='both',
                direction='in',
                which='both',
                bottom=True,
                top=True,
                left=True,
                right=True,
            )

    fig, ax = plot_fig_and_ax
    plot_fmt = next(PLOT_LINE_FMT_CYCLER) if fmt is None else fmt
    marker_fmt = next(PLOT_MARKER_FMT_CYCLER) if fmt is None else fmt

    ax.plot(
        xs_to_be_plotted,
        ys_to_be_plotted,
        plot_fmt,
        # marker = marker_fmt,
        # markevery=15,
        # markersize=6,       # 标记大小
        label=plot_label,
    )

    ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
    ax.axvspan(-1, 1, color='gray', alpha=0.06)  # alpha 控制透明度（0-1）

    if plot_label:
        ax.legend(loc="lower right")

    IPython.display.display(fig)
    plt.close(fig)  # prevent displaying twice in jupyter notebook

    return points_to_be_plotted, fig, ax
