# This file contains modified code from Py-BOBYQA
# Original Copyright (c) 2017-2024, Lindon Roberts, Alexander Senier
# and David Carlisle
# Modifications Copyright (c) 2025, Yichuan Liu and Yingzhou Li
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
Algorithm Manager
=================

Maintain two classes that command the behaviors of the UPOQA algorithm.
"""

import numpy as np
import numpy.linalg as LA
import itertools
import sys
import time
import threading
from .model import QuadSurrogate, OverallSurrogate, SurrogateLinAlgError
from .interp_set import InterpSet, OverallInterpSet
from copy import deepcopy
from .sub_solver import cauchy_geometry, spider_geometry
import scipy.stats as STAT
from collections import deque
from .params import UPOQAParameterList
from typing import Any, Dict, Callable, Optional, Union, Tuple, List, Union
import traceback
import signal

__all__ = [
    'UPOQAManager',
    'MaxEvalNumReached',
    'ExitInfo',
    'ExitStatus',
]


class ExitStatus:

    AUTO_DETECT_RESTART_WARNING = 2  # warning, auto-detected restart criteria
    SLOW_WARNING = 1  # warning, maximum number of slow (successful) iterations reached
    SUCCESS = 0  # successful finish (rho = resolution_final, or reach maxiter or maxfev)
    INPUT_ERROR = -1  # error, bad inputs
    TR_INCREASE_ERROR = -2  # error, trust region step increased model value
    LINALG_ERROR = -3  # error, linalg error (singular matrix encountered)
    INVALID_EVAL_ERROR = -4  # error, funtion values with invalid shape or type or NaN encountered
    UNKNOWN_ERROR = -5  # error, raised by unknown error source


class MaxEvalNumReached(Exception):
    """
    Exception raised when the maximum number of evaluations of the objective function has
    been reached.
    """

    def __init__(self, ele_idx: Optional[int] = None, ele_name: Any = "") -> None:
        self.ele_idx = ele_idx
        self.ele_name = ele_name


class ExitInfo:
    """
    Class that contains the exit information.

    Attributes
    ----------
    flag : int
        Flag that represents the type of the cause of an exit,
        and should be one of the following flags:

            ```
            ExitStatus.AUTO_DETECT_RESTART_WARNING = 2  # warning, auto-detected restart criteria
            ExitStatus.SLOW_WARNING = 1                 # warning, maximum number of slow (successful) iterations reached
            ExitStatus.SUCCESS = 0                      # successful finish (rho = resolution_final, or reach maxiter or maxfev)
            ExitStatus.INPUT_ERROR = -1                 # error, bad inputs
            ExitStatus.TR_INCREASE_ERROR = -2           # error, trust region step increased model value
            ExitStatus.LINALG_ERROR = -3                # error, linalg error (singular matrix encountered)
            ExitStatus.INVALID_EVAL_ERROR = -4          # error, funtion values with invalid shape or type or NaN encountered
            ExitStatus.UNKNOWN_ERROR = -5               # error, raised by unknown error source
            ```
    msg : str
        Message that describes the detailed reason for an exit.
    """

    def __init__(
        self,
        flag: int,
        msg: str,
        exception: Optional[Exception] = None,
        traceback: Optional[str] = None,
    ) -> None:
        self.flag = flag
        self.msg = msg
        self.exception = exception
        self.traceback = traceback

    def message(self, with_stem: bool = True) -> str:
        """
        Return the message that describes the detailed reason for an exit.

        Returns
        -------
        str
            The message that describes the detailed reason for an exit.
            if `with_stem` is True, the returned message will be prefixed with
            "Error" or "Warning".
        """
        if not with_stem:
            return self.msg
        elif self.flag == ExitStatus.SUCCESS:
            return "Success: " + self.msg
        elif self.flag == ExitStatus.SLOW_WARNING:
            return "Warning (slow progress): " + self.msg
        elif self.flag == ExitStatus.AUTO_DETECT_RESTART_WARNING:
            return "Warning (auto detect restart): " + self.msg
        elif self.flag == ExitStatus.INPUT_ERROR:
            return "Error (bad input): " + self.msg
        elif self.flag == ExitStatus.TR_INCREASE_ERROR:
            return "Error (trust region increase): " + self.msg
        elif self.flag == ExitStatus.LINALG_ERROR:
            return "Error (linear algebra): " + self.msg
        elif self.flag == ExitStatus.INVALID_EVAL_ERROR:
            return "Error (invalid function evaluation): " + self.msg
        elif self.flag == ExitStatus.UNKNOWN_ERROR:
            return "Error (unknown): " + self.msg
        else:
            return "Unknown exit flag: " + self.msg


class UPOQAManager:
    r"""
    Manage the execution and state of the UPOQA algorithm.

    Parameters
    ----------
    resolution_init : float
        The initial value of trust region resolution and elemental radii.
    resolution_final : float
        The final value of trust region resolution and elemental radii.
    maxiter : int
        Maximum number of iterations to go before termination.
    maxfev :  list of int
        Maximum function evaluations per element (list parallel to `ele_names`).
    proj_onto_ele : callable
        Projection function mapping full-space points to element subspaces:

            ``(full_point: ndarray, element_name: Any) -> element_point: ndarray``

        For example, `proj_onto_ele([1.0, 2.0], "f_1")` yields `[1.0,]`, where
        `"f_1"` denotes a function that depends only on the first variable of `x`.
    coords : list of 1D arrays
        Variable indices for each element's subspace (parallel to `ele_names`).
    xform_bounds : list of tuples
        List in which each member is a tuple of the form `(l_val, h_val)`, representing
        the interval type domain of each transformation function in `xforms`.
    params : `~upoqa.utils.UPOQAParameterList`
        Parameter list used for the algorithm.
    ele_names : list
        List containing the names of all elements in order
    disp : int, optional
        Verbosity level:
            0: silent
            1: brief
            2: verbose
            3: verbose with debug information

    Attributes
    ----------
    n
    npts : ndarray, shape (ele_num,)
        Number of interpolation points for each element
    resolution : float
        Current trust region resolution
    radii : ndarray
        Trust-region radii for each element.
    ele_names : list
        List containing the names of all elements in order
    ele_num : int
        Number of elements
    proj_onto_ele : callable
        Projection function mapping full-space points to element subspaces:

            ``(full_point: ndarray, element_name: Any) -> element_point: ndarray``

        For example, `proj_onto_ele([1.0, 2.0], "f_1")` yields `[1.0,]`, where
        `"f_1"` denotes a function that depends only on the first variable of `x`.
    interp_set : ``~upoqa.utils.OverallInterpSet``
        The interpolation set used by the algorithm
    model : ``~upoqa.utils.OverallSurrogate``
        The surrogate model used by the algorithm
    ele_models
    xforms
    weights
    extra_fun
    params : `~upoqa.utils.UPOQAParameterList`
        Parameter list used for the algorithm
    maxiter : int
        Maximum number of iterations to go before termination
    maxfev : list of int
        Maximum function evaluations per element (list parallel to `ele_names`).
    nfev : list of int
        Numbers of element function evaluations performed so far
    nrun : int
        Number of runs (restarts) performed so far
    disp : int
        Verbosity level:
            0: silent
            1: brief
            2: verbose
            3: verbose with debug information
    x_opt
    f_opt

    """

    def __init__(
        self,
        resolution_init: float,
        resolution_final: float,
        maxiter: int,
        maxfev: List[int],
        proj_onto_ele: Callable[[np.ndarray, Any], np.ndarray],
        coords: List[Union[List, np.ndarray]],
        xform_bounds: List[Tuple[float]],
        params: UPOQAParameterList,
        ele_names: List[Any],
        disp: int = 0,
    ) -> None:
        self.resolution, self.resolution_init = resolution_init, resolution_init
        self.resolution_final = resolution_final

        self.ele_names = ele_names
        self.ele_idxs = list(range(len(ele_names)))
        self.ele_num = len(self.ele_idxs)

        self.radii = np.ones(self.ele_num) * resolution_init
        self.interp_set, self.model = None, None

        # Problem geometry setup
        self.proj_onto_ele = proj_onto_ele
        assert callable(self.proj_onto_ele)
        self.coords = coords
        self.xform_bounds = xform_bounds

        # Algorithm parameters
        self.params = params
        self.resolution_restart = (
            self.params("restarts.resolution_restart_init_ratio") * resolution_init
        )

        self.maxiter = maxiter
        self.maxfev = maxfev
        self.nfev = [0 for _ in range(self.ele_num)]

        # Restart info monitoring
        self.nrun = 0
        self.last_run_f_opt = np.inf
        self.total_unsuccessful_restarts = 0
        self.num_slow_iters = 0
        self.history_fvals = deque(
            [], self.params("slow.history_for_slow")
        )  # Recent objective history

        # Output control
        self.disp = int(disp)

        # Initialize tracking structures
        self.reset_restart_track_info()
        self.init_coord_overlap_relation()

    @property
    def n(self) -> Optional[int]:
        """Problem dimension"""
        return self.interp_set.n if self.interp_set is not None else None

    @property
    def npts(self) -> Optional[np.ndarray]:
        """Number of interpolation points for each element"""
        return self.interp_set.npts if self.interp_set is not None else None

    @property
    def ele_models(self) -> Optional[List[QuadSurrogate]]:
        """List of elemental surrogate models"""
        return self.model.ele_models if self.model is not None else None

    @property
    def ele_interp_sets(self) -> List[InterpSet]:
        """List of elemental interpolation sets"""
        return self.interp_set.ele_interp_sets if self.interp_set is not None else None

    @property
    def xforms(
        self,
    ) -> Optional[List[Optional[List[Callable[[np.ndarray], Union[np.ndarray, float]]]]]]:
        r"""Transformations :math:`h_1, \ldots, h_q` applied to element function outputs"""
        return self.model.xforms if self.model is not None else None

    @property
    def weights(self) -> Optional[np.ndarray]:
        r"""Weights :math:`w_1, \ldots, w_q` for each element in the model"""
        return self.model.weights if self.model is not None else None

    @property
    def extra_fun(self) -> Optional[List[Callable[[np.ndarray], Union[np.ndarray, list, float]]]]:
        r"""
        The white-box component :math:`f_0` of the objective function, which is a list of length 3
        containing callables to evaluate the function values, gradients, and Hessians.
        """
        return self.model.extra_fun if self.model is not None else None

    @property
    def x_opt(self) -> Optional[np.ndarray]:
        r"""
        The point with the lowest objective function value among the currently tracked
        iterates in the overall interpolation point set.
        """
        return self.interp_set.x_opt if self.interp_set is not None else None

    @property
    def f_opt(self) -> Optional[float]:
        """
        Value of objective function at the point with the lowest objective function value
        among the currently tracked iterates in the overall interpolation point set.
        """
        return self.interp_set.f_opt if self.interp_set is not None else None

    def init_coord_overlap_relation(self):
        """
        Initialize adjacency relationships between elements based on their coordinate overlaps.
        """
        self.ele_adj = np.zeros((self.ele_num, self.ele_num), dtype=bool)
        for i in range(self.ele_num):
            for j in range(i + 1, self.ele_num):
                self.ele_adj[i, j] = np.any(
                    np.isin(self.coords[self.ele_idxs[i]], self.coords[self.ele_idxs[j]])
                )
                self.ele_adj[j, i] = self.ele_adj[i, j]
        # not contain oneself
        self.ele_neighbours = [np.argwhere(self.ele_adj[i]) for i in self.ele_idxs]

    def _get_max_radius_overlapped(self, idx: int) -> float:
        mr_accu = 0
        for nb in self.ele_neighbours[idx]:
            mr_accu += self.radii[int(nb)] ** 2
        return mr_accu**0.5

    def has_reached_maxfev(self) -> bool:
        """Check if any element has exceeded its maximum function evaluation limit."""
        for ele_idx in self.ele_idxs:
            if self.maxfev[ele_idx] <= self.nfev[ele_idx]:
                return True
        return False

    def reset_restart_track_info(self) -> None:
        """
        Reset tracking information used for automatic restart detection.
        """
        # Attempting to auto-detect restart? Need to keep a history of delta and ||chg J||
        # for non-safety iterations
        # have we filled up the whole history vectors yet? Don't restart from this if not
        self.restart_auto_detect_ready = False
        if self.params("restarts.use_restarts") and self.params("restarts.auto_detect"):
            self.restart_auto_detect_delta = -1.0 * np.ones(
                (self.params("restarts.auto_detect.history"),)
            )
            self.restart_auto_detect_chg_grad = -1.0 * np.ones(
                (self.params("restarts.auto_detect.history"),)
            )
            self.restart_auto_detect_chg_hess = -1.0 * np.ones(
                (self.params("restarts.auto_detect.history"),)
            )

    def set_radius(self, radius: float, ele_idx: Optional[int] = None) -> None:
        """
        Set the trust region radius to `radius`.

        If `ele_idx` is specified, only updates the radius for that element. Otherwise,
        updates all elements' trust region radii. The radius is constrained to be:
        At least the resolution (if below θ₂ * resolution)
        At most θ₆ * resolution

        Parameters
        ----------
        radius : float
            The new trust region radius.
        ele_idx : int, optional
            Index of the element whose radius is to be updated. (None means update all
            elements.)
        """
        if radius <= self.params("tr_radius.theta2") * self.resolution:
            radius = self.resolution
        radius = min(radius, self.resolution * self.params("tr_radius.theta6"))

        if ele_idx is None:
            for ele_idx2 in self.ele_idxs:
                self.radii[ele_idx2] = radius
        else:
            self.radii[ele_idx] = radius

    def set_resolution(self, resolution: float) -> None:
        """
        Set the trust region resolution to `resolution`.

        Parameters
        ----------
        radius : float
            The new trust region resolution.
        """
        self.resolution = resolution

    def set_weights(self, weights: np.ndarray) -> None:
        """Update the weights of all elements in the model."""
        self.model.set_weights(weights)

    def set_xforms(
        self,
        xforms: List[Optional[List[Callable[[np.ndarray], Union[np.ndarray, float]]]]],
    ) -> None:
        """Update the transformation functions for element outputs."""
        self.model.set_xforms(xforms)

    def update_weights(
        self, weights: Union[Dict[Any, Union[int, float]], List[Union[int, float]], np.ndarray]
    ) -> None:
        """
        Update model weights, handling both list and dictionary inputs.

        If `weights` is a dictionary, it maps element names to weight values.
        Missing elements retain their current weights.

        Parameters
        ----------
        weights : list or dict or ndarray
            New weights for model elements. Can be a 1D array, or a list of weights
            for all elements or a dictionary mapping element names to weights.
        """
        if isinstance(weights, dict):
            weights_list = []
            for ele_idx in self.ele_idxs:
                ele_name = self.ele_names[ele_idx]
                if ele_name in weights:
                    weights_list.append(float(weights[ele_name]))
                else:
                    weights_list.append(float(self.weights[ele_idx]))
            weights = weights_list
        self.model.update_weights(weights)

    def update_xforms(
        self,
        xforms: Union[
            Dict[Any, Optional[List[Callable[[np.ndarray], Union[np.ndarray, float]]]]],
            List[Optional[List[Callable[[np.ndarray], Union[np.ndarray, float]]]]],
        ],
    ) -> None:
        """
        Update transformation functions with input validation.

        Handle both list and dictionary inputs. For dictionary inputs:
        Validate each value is a list of 3 callables
          (which are function, gradient and hessian.)
        Missing elements retain their current transformations
        Apply bounds wrapping if `xform_bounds` is enabled

        Parameters
        ----------
        xforms : list or dict
            New transformation functions. Can be either:
                1. A dictionary mapping element names to transformation lists
                2. A list of transformation lists for all elements

            each transformation list should contain exactly 3 callables:
            `[function, gradient, hessian]`.
        """
        if isinstance(xforms, dict):
            xforms_list = []
            for ele_idx in self.ele_idxs:
                ele_name = self.ele_names[ele_idx]
                if ele_name in xforms:
                    if xforms[ele_name] is not None:
                        assert isinstance(xforms[ele_name], list), (
                            f"xforms[{ele_name}] should be a list,"
                            f"but got {type(xforms[ele_name])}"
                        )
                        assert len(xforms[ele_name]) == 3, (
                            f"xforms[{ele_name}] should be a list of length 3,"
                            f" but got length {len(xforms[ele_name])}"
                        )
                        assert all(
                            [isinstance(op, Callable) for op in xforms[ele_name]]
                        ), f"xforms[{ele_name}] should contain only callable elements."
                else:
                    xforms[ele_name] = self.xforms[ele_idx]
                xforms_list.append(xforms[ele_name])
            xforms = xforms_list

        if self.xform_bounds:
            xforms = self.wrap_xforms_with_bounds(xforms)
        self.model.update_xforms(xforms)

    def shift_center_to(self, x: np.ndarray) -> None:
        """
        Shift the centers of overall and element models to `x` if movement exceeds
        threshold, then update the coefficients of the models and the interpolation systems.

        Parameters
        ----------
        x : ndarray, shape (n,)
            The new model center.
        """
        self.model.shift_center_to(x)
        for ele_idx in self.ele_idxs:
            ele_model = self.ele_models[ele_idx]
            x_ele = self.proj_onto_ele(x, ele_idx)
            if (
                LA.norm(x_ele - ele_model.model_center)
                >= self.params("general.center_shift_threshold") * self.radii[ele_idx]
            ):
                ele_model.shift_center_to(x_ele)

    def couple_with_utils(
        self,
        interp_set: OverallInterpSet,
        model: OverallSurrogate,
    ) -> None:
        """
        Connect the manager with interpolation set and surrogate model utilities.

        Parameters
        ----------
        interp_set : ``~upoqa.utils.OverallInterpSet``
            The overall interpolation set.
        model : ``~upoqa.utils.OverallSurrogate``
            The overall surrogate model.
        """
        self.interp_set = interp_set
        self.model = model

    def build_fval(self, fval_eles: List[float], extra_fval: float) -> float:
        """
        Construct overall function value from element contributions.

        Combines element values (optionally transformed) with weights and extra value.
        Performs detailed NaN checking when debug mode is enabled.

        Parameters
        ----------
        fval_eles : list of float
            Function values from each element.
        extra_fval : float
            Function value of the white-box component to include in the total.

        Returns
        -------
        float
            The calculated overall function value.

        Raises
        -------
        ValueError
            If NaN values are detected in element contributions or final result.
        """
        fval = extra_fval
        for ele_idx in self.ele_idxs:
            if self.xforms[ele_idx] is not None:
                fval_ele = self.weights[ele_idx] * self.xforms[ele_idx][0](fval_eles[ele_idx])
            else:
                fval_ele = self.weights[ele_idx] * fval_eles[ele_idx]
            if self.params("debug.check_nan_fval") and np.any(np.isnan(fval_ele)):
                raise ValueError(
                    f"(element {self.ele_names[ele_idx]}) NaN encountered in the return"
                    f" value of the surrogate function. \nreturn value = {fval_ele}"
                )
            fval += fval_ele
        if np.isnan(fval):
            raise ValueError(f"NaN overall surrogate function value encountered.")
        return fval

    def get_index_to_remove(
        self, x: Optional[np.ndarray] = None, center: Optional[np.ndarray] = None
    ) -> Tuple[
        List[int],
        List[float],
        List[Optional[Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]]],
    ]:
        """
        Select the interpolation points to be deleted for each element based on possibly
        provided new point `x`.

        Parameters
        ----------
        x : ndarray, shape (n,), optional
            The new candidate point to be added to the elemental interpolation sets.

            If None, for each `ele_idx`-th element, selects the farthest point in
            the elemental interpolation set from `self.proj_onto_ele(center, ele_idx)`.

            If provided, for each `ele_idx`-th element, calculates sigma values (the
            denominator of the update formula of the interpolation system) by replacing
            each existing interpolation point with `self.proj_onto_ele(x, ele_idx)` and
            selects the one with the maximum (sigma * distance_to_center ** 4) metric.
        center : ndarray, shape (n,), optional
            Center point used for distance calculations. If None, uses the optimal
            point from the overall interpolation set as the center.

        Returns
        -------
        list of int
            List of indices of points to be removed from each elemental interpolation set.
        list of float
            List of distances between the center point (`self.proj_onto_ele(center, ele_idx)`
            for `ele_idx`-th element) and the points to be removed, for each elemental
            interpolation set.
        list
            List of cached outputs of

                ``ele_model.get_determinant_ratio(self.proj_onto_ele(x, ele_idx))``

            for each element model `ele_model` if `x` is provided, else None for each
            element.
        """
        knew_dict = []
        distance_power_to_Y_from_center_eles = []
        cached_KKT_info_eles = []
        global_center = center if center is not None else self.interp_set.x_opt

        for ele_idx in self.ele_idxs:
            ele_interp_set = self.ele_interp_sets[ele_idx]
            Y, _ = ele_interp_set.get_interp_set()
            center = self.proj_onto_ele(global_center, ele_idx)
            distance_power_to_Y_from_center = LA.norm(Y - center, axis=1) ** 2
            if x is None:
                sigma = 1.0
                weights = distance_power_to_Y_from_center
                cached_KKT_info_eles.append(None)
            else:
                x_ele = self.proj_onto_ele(x, ele_idx)
                cached_KKT_info = self.ele_models[ele_idx].get_determinant_ratio(x_ele)
                cached_KKT_info_eles.append(cached_KKT_info)
                sigma = cached_KKT_info[3]
                weights = (
                    distance_power_to_Y_from_center
                    / max(0.1 * self.radii[ele_idx], self.resolution) ** 2.0
                ) ** 2.0
            k_max = np.argmax(weights * np.abs(sigma))
            knew_dict.append(int(k_max))
            distance_power_to_Y_from_center_eles.append(
                float(np.sqrt(distance_power_to_Y_from_center[k_max]))
            )
        return knew_dict, distance_power_to_Y_from_center_eles, cached_KKT_info_eles

    def get_geometry_step(self, idx_eles: List[int], want_GI: List[bool]) -> Tuple[
        List[Optional[np.ndarray]],
        List[Optional[Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]]],
    ]:
        """
        Compute and return the geometry-improving steps for the elemental interpolation sets.

        Parameters
        ----------
        idx_eles : list of int
            List of indices indicating target positions in each elemental interpolation set
            where new points will be placed.
        want_GI : list of bool
            List of flags indicating whether a geometry improvement step is needed
            for each elemental interpolation set.

        Returns
        -------
        list
            List of computed geometry-improving step vectors for each elemental interpolation
            set. Return None for elements where geometry improvement is not needed.
        list
            List of cached outputs of `ele_model.get_determinant_ratio(x_GI)` for each
            element model `ele_model`, where `x_GI` denotes the geometry-improving point
            for the current element. Return None for elements where geometry improvement
            is not needed.

        Notes
        -----
        The method computes two potential steps for each element:

        1. A constrained Cauchy step
        2. A step along lines connecting interpolation points

        The step with better geometry improvement (larger absolute sigma value) is selected.
        If both steps result in zero determinant ratio, raises a `~upoqa.utils.SurrogateLinAlgError`.
        """
        start_x = self.interp_set.x_opt
        step_eles = []
        cached_kkt_info_eles = []
        for ele_idx in self.ele_idxs:
            interp_set = self.ele_interp_sets[ele_idx]
            GI_radius = max(self.params("tr_radius.alpha3") * self.radii[ele_idx], self.resolution)
            if want_GI[ele_idx]:
                worst_idx = idx_eles[ele_idx]
                model = self.ele_models[ele_idx]
                start_x_ele = self.proj_onto_ele(start_x, ele_idx)
                coord_vec = np.squeeze(np.eye(1, interp_set.npt, worst_idx))
                lag_interp_set = deepcopy(interp_set)
                lag_interp_set.set_interp_set_fval(coord_vec)

                lag = QuadSurrogate(
                    interp_set=lag_interp_set,
                    center=model.model_center,
                    ref_surrogate=model,
                )
                g_lag = lag.grad_eval(start_x_ele)
                f_lag = lag.fun_eval(start_x_ele)

                step = cauchy_geometry(
                    f_lag, g_lag, lambda v: lag.hess_operator(v).dot(v), GI_radius
                )
                cached_kkt_info = model.get_determinant_ratio(start_x_ele + step)
                sigma = cached_kkt_info[3][worst_idx]

                Y, _ = interp_set.get_interp_set()
                xpt = Y - start_x_ele
                norms = LA.norm(xpt, axis=1)
                xpt = xpt[norms != 0.0]
                step_alt = spider_geometry(
                    f_lag, g_lag, lambda v: lag.hess_operator(v).dot(v), xpt, GI_radius
                )
                cached_kkt_info_alt = model.get_determinant_ratio(start_x_ele + step_alt)
                sigma_alt = cached_kkt_info_alt[3][worst_idx]

                if sigma_alt == 0.0 and sigma == 0.0:
                    raise SurrogateLinAlgError(
                        "The determinant ratio is zero. ",
                        ele_idx=ele_idx,
                        ele_name=self.ele_names[ele_idx],
                    )

                if abs(sigma_alt) > abs(sigma):
                    if self.disp >= 3:
                        print(
                            f"  (element {self.ele_names[ele_idx]}) Use spider"
                            "-geometry-improvement,"
                            f" abs({sigma_alt:.3e}) > abs({sigma:.3e})"
                        )
                    step_eles.append(step_alt)
                    cached_kkt_info_eles.append(cached_kkt_info_alt)
                else:
                    if self.disp >= 3:
                        print(
                            f"  (element {self.ele_names[ele_idx]}) Use cauchy"
                            "-geometry-improvement,"
                            f" abs({sigma:.3e}) >= abs({sigma_alt:.3e})"
                        )
                    step_eles.append(step)
                    cached_kkt_info_eles.append(cached_kkt_info)
            else:
                step_eles.append(None)
                cached_kkt_info_eles.append(None)

        return step_eles, cached_kkt_info_eles

    def wrap_xforms_with_bounds(self, xforms: List[List[callable]]) -> List[List[callable]]:
        """
        Wrap transformation functions with bounds checking for each element.

        Parameters
        ----------
        xforms : list
            List of lists containing transformation functions for each element.
            Each inner list contains one element's transformation's function, gradient and
            hessian.

        Returns
        -------
        list
            List of lists containing wrapped transformation functions with bounds checking.
            Return None for elements with no transformations.
        """

        def generate_wrapped_xforms(
            func: Optional[Callable[[np.ndarray], Union[np.ndarray, float]]], ele_idx: int
        ):
            if func is None:
                return None

            low_bound, high_bound = self.xform_bounds[ele_idx]

            if (low_bound != -np.inf) or (high_bound != np.inf):

                def bound_checked_xform(Y: Union[float, np.ndarray]):
                    if isinstance(Y, np.ndarray):
                        return func(
                            np.clip(Y, low_bound, high_bound, out=Y)
                        )  # do the clip in-place.
                    else:
                        return func(np.clip(Y, low_bound, high_bound))

                return bound_checked_xform
            else:
                return func

        return [
            (
                [generate_wrapped_xforms(func, ele_idx) for func in xforms[ele_idx]]
                if xforms[ele_idx] is not None
                else None
            )
            for ele_idx in self.ele_idxs
        ]

    def _update_single_radius(self, tr_vio: float, score: int, ele_idx: int) -> None:
        """
        Update the trust region radius for a single element based on its score.

        Parameters
        ----------
        tr_vio : float
            Trust region violation ratio, denotes the ratio of the norm of the trust region
            step to the radius.
        score : int
            Elemental score under the combined separation criterion suggested by [1]_,
            ranging from 0 to 4. High scores indicate expanding trust region radius, low
            scores indicate contracting.
        ele_idx : int
            Index of the element whose radius is to be updated.

        References
        ----------
        .. [1] Johara S. Shahabuddin. 1996. *Structured Trust Region Algorithms for the
            Minimization of Nonlinear Functions*. Ph. D. Dissertation. Cornell University, USA.
        """
        if score <= 0:
            self.radii[ele_idx] *= self.params("tr_radius.theta1")  # *= 0.5
        elif score == 1:
            self.radii[ele_idx] *= self.params("tr_radius.theta5")  # *= 0.5 ** 0.5
        elif score == 2:
            # *= max(0.5 ** 0.5, min(1.0, 1.414 * tr_vio)), in [0.5 ** 0.5, 1.0] * radius
            self.radii[ele_idx] *= max(
                self.params("tr_radius.theta5"), min(1.0, self.params("tr_radius.theta3") * tr_vio)
            )
        elif score == 3:
            # *= min(1.414, max(1.0, 2.0 * tr_vio)), in [1, 1.414] * radius
            self.radii[ele_idx] *= min(
                self.params("tr_radius.theta3"),
                max(1.0, self.params("tr_radius.theta4") * tr_vio),
            )
        else:
            # *= min(2.0, max(1.0, 2.0 * tr_vio)), in [1, 2.0] * radius
            self.radii[ele_idx] *= min(
                self.params("tr_radius.theta4"),
                max(1.0, self.params("tr_radius.theta4") * tr_vio),
            )
        self.set_radius(self.radii[ele_idx], ele_idx)

    def update_radius(
        self,
        tr_vios: np.ndarray,
        ratio: float,
        ratios: List[float],
        decrease_mval: float,
        decrease_extra_fval: float,
        decrease_wx_m_eles: List[float],
        decrease_wx_f_eles: List[float],
        spherical_step_radius: float = 1.0,
        use_spherical_tr: bool = False,
    ) -> None:
        """
        Update elemental trust region radii based on the combined separation criterion
        suggested by [1]_.

        Parameters
        ----------
        tr_vios : ndarray, shape (ele_num,)
            Trust region violation ratios for each element. Trust region violation ratio
            denotes the ratio of the norm of the trust region step to the radius.
        ratio : float
            Actual-to-predicted reduction ratio of the objective function value.
        ratios : list of float
            Actual-to-predicted reduction ratios of the function value of each element
            function.
        decrease_mval : float
            Decrease of the overall model value.
        decrease_extra_fval : float
            Decrease of the extra function value. The extra function is the white-box
            component to include in the total objective function.
        decrease_wx_m_eles : list of float
            Decrease of the weighted and transformed value of each element model. Note that
            for each element, the default weight is 1.0, and the default transformation
            (xform) is the identity.
        decrease_wx_f_eles : list of float
            Decrease of the weighted and transformed value of each element function. Note
            that for each element, the default weight is 1.0, and the default transformation
            (xform) is the identity.
        spherical_step_radius : float, default=1.0
            The ratio of the norm of the trust region step to the radius. This is only used
            when `use_spherical_tr` is True.
        use_spherical_tr : bool, default=False
            Whether the spherical trust region is enabled. If enabled, the truncated
            conjugate gradient method is used to solve the trust region subproblem.

        References
        ----------
        .. [1] Johara S. Shahabuddin. 1996. *Structured Trust Region Algorithms for the
            Minimization of Nonlinear Functions*. Ph. D. Dissertation. Cornell University, USA.
        """
        delta_m_pos, delta_m_neg = 0, 0
        for ele_idx in self.ele_idxs:
            tmp = decrease_wx_m_eles[ele_idx]
            if tmp > 0:
                delta_m_pos += tmp
            else:
                delta_m_neg += tmp

        if decrease_extra_fval > 0:
            delta_m_pos += decrease_extra_fval
        else:
            delta_m_neg += decrease_extra_fval

        rho = delta_m_neg / delta_m_pos
        mu1, mu2 = self.params("tr_radius.eta1"), self.params("tr_radius.eta2")
        eta1, eta2 = -(1 - mu1) * rho, -(1 - mu2) * rho
        mu1p, mu2p = mu1 + eta1, mu2 + eta2

        def get_criterion_alpha(mu, rho):
            # rho in [-1, 0], alpha in [mu, 1]
            return (mu - (2 - mu) * rho) / (1 - rho)

        alpha_1, alpha_2 = get_criterion_alpha(mu1p, rho), get_criterion_alpha(mu2p, rho)
        ele_num = self.ele_num

        tau_global_score = -1
        if ratio >= mu2:
            tau_global_score = 2
        elif ratio < mu1:
            tau_global_score = 0
        else:
            tau_global_score = 1

        tau_scores = [-1] * self.ele_num

        for ele_idx in self.ele_idxs:

            if use_spherical_tr:
                tau_scores[ele_idx] = tau_global_score + 1 if tau_global_score > 0 else 0
                continue

            ratio_ele = ratios[ele_idx]
            delta_wx_f_ele = decrease_wx_f_eles[ele_idx]
            delta_wx_m_ele = decrease_wx_m_eles[ele_idx]

            if delta_wx_m_ele >= 0:
                if (ratio_ele >= alpha_2) or (
                    delta_wx_f_ele >= delta_wx_m_ele - eta2 * decrease_mval / ele_num
                ):
                    tau_scores[ele_idx] = 2
                elif (ratio_ele >= alpha_1) or (
                    delta_wx_f_ele >= delta_wx_m_ele - eta1 * decrease_mval / ele_num
                ):
                    tau_scores[ele_idx] = 1
                else:
                    tau_scores[ele_idx] = 0
            else:
                if (ratio_ele <= 2 - alpha_2) or (
                    delta_wx_f_ele >= delta_wx_m_ele - eta2 * decrease_mval / ele_num
                ):
                    tau_scores[ele_idx] = 2
                elif (ratio_ele <= 2 - alpha_1) or (
                    delta_wx_f_ele >= delta_wx_m_ele - eta1 * decrease_mval / ele_num
                ):
                    tau_scores[ele_idx] = 1
                else:
                    tau_scores[ele_idx] = 0

            tau_scores[ele_idx] += tau_global_score

        if tau_global_score == 0:
            # if ratio < eta1, make sure at least one elemental radius is to be actually reduced.
            eles_who_actually_can_reduce = []
            actual_reduce_is_called = False
            for ele_idx in self.ele_idxs:
                if self.radii[ele_idx] > self.resolution:
                    eles_who_actually_can_reduce.append(
                        (ele_idx, (tau_scores[ele_idx], tr_vios[ele_idx]))
                    )
                    if (tau_scores[ele_idx] <= 1) or (
                        (
                            tau_scores[ele_idx] == 2
                            and tr_vios[ele_idx] <= 1 / self.params("tr_radius.theta4")
                        )
                    ):
                        actual_reduce_is_called = True

            if eles_who_actually_can_reduce and (not actual_reduce_is_called):
                eles_who_actually_can_reduce.sort(key=lambda x: x[1])
                tau_scores[eles_who_actually_can_reduce[0][0]] = 0

        for ele_idx in self.ele_idxs:
            if use_spherical_tr:
                self._update_single_radius(spherical_step_radius, tau_scores[ele_idx], ele_idx)
            else:
                self._update_single_radius(tr_vios[ele_idx], tau_scores[ele_idx], ele_idx)

    def get_reduction_ratio(
        self,
        fval: float,
        old_fval: float,
        decrease: float,
        fval_eles: List[float],
        old_fval_eles: List[float],
        decrease_eles: List[float],
        rreg: Optional[float] = None,
    ) -> Tuple[float, List[float], Optional[ExitInfo]]:
        """
        Compute the trust-region reduction ratios between actual and predicted decreases,
        both for the overall objective function and for each element function.

        Parameters
        ----------
        fval : float
            Objective function value at the trial point `xk + sk`, where `sk` is the
            trust-region step.
        old_fval : float
            Objective function value at the iterate `xk`.
        decrease : float
            Decrease of the overall model.
        fval_eles : list of float
            Objective function values of each element at the trial point `xk + sk`.
        old_fval_eles : list of float
            Objective function values of each element at the iterate `xk`.
        decrease_eles : list of float
            Decrease of each element model.
        rreg : float, optional
            Regularization term to stabilize ratio calculation. Defaults to machine epsilon
            scaled by the function values.

        Returns
        -------
        ratio : float
            The reduction ratio for the objective function. Return `-1` if the denominator
            (predicted decrease) is invalid (positive or too close to zero).
        ratios : list of float
            The reduction ratios for each element function. Return `-1` if the denominator
            (predicted decrease) is invalid (positive or too close to zero).
        exit_info : ``~upoqa.utils.ExitInfo`` or None
            An ``~upoqa.utils.ExitInfo`` object with an `ExitStatus.TR_INCREASE_ERROR` flag if the
            denominator is invalid, None otherwise.
        """
        rreg = rreg or np.finfo(np.float64).eps * np.max([1.0, fval, old_fval])

        numerator = old_fval - fval + rreg
        denominator = -decrease - rreg  # should be negative
        if denominator > 0.0:
            return (
                -1,
                [],
                ExitInfo(
                    ExitStatus.TR_INCREASE_ERROR,
                    f"Trust region step gave model increase = {denominator}.",
                ),
            )

        if abs(denominator) > np.finfo(float).tiny * abs(numerator):
            ratio = (numerator) / (-denominator)
        else:
            ratio = -1

        ratios = []
        for ele_idx in self.ele_idxs:
            numerator = old_fval_eles[ele_idx] - fval_eles[ele_idx] + rreg
            denominator = -decrease_eles[ele_idx] - rreg  # should be negative
            if abs(denominator) > np.finfo(float).tiny * abs(numerator):
                ratios.append(float(numerator / (-denominator)))
            else:
                ratios.append(-1)

        return ratio, ratios, None

    def reduce_resolution(self) -> bool:
        """
        Reduce the resolution of the trust-region.

        Returns
        -------
        bool
            True if resolution has already reached its minimum value (`radius_final`),
            False otherwise.
        """
        if self.resolution <= self.resolution_final:
            return True

        if 250.0 * self.resolution_final < self.resolution:
            self.resolution *= self.params("tr_radius.alpha1")
        elif 16.0 * self.resolution_final < self.resolution:
            self.resolution = np.sqrt(self.resolution * self.resolution_final)
        else:
            self.resolution = self.resolution_final

        for ele_idx in self.ele_idxs:
            self.set_radius(
                max(
                    self.params("tr_radius.alpha2") * self.radii[ele_idx],
                    self.resolution,
                ),
                ele_idx,
            )

        return False

    @property
    def average_radius(self) -> float:
        """
        The average of elemental trust-region radii
        """
        return np.mean([self.radii[ele_idx] for ele_idx in self.ele_idxs])

    @property
    def max_radius(self) -> float:
        """
        The maximum of elemental trust-region radii
        """
        return max(self.radii)

    def calc_detailed_decrease(
        self,
        xk: np.ndarray,
        x_hold: np.ndarray,
        old_fval_eles: List[float],
        fval_eles: List[float],
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Obtain extra information about the decrease of values of surrogate model and
        objective function for each element. The returned values are then used to determine
        whether to expand or shrink the trust-region radius for each element.

        Parameters
        ----------
        xk : ndarray
            The current iterate of the optimization process
        x_hold : ndarray
            The trial point `xk + sk`, where `sk` is the trust-region step.
        old_fval_eles : list of float
            Objective function values of each element at the iterate `xk`.
        fval_eles : list of float
            Objective function values of each element at the trial point `xk + sk`.

        Returns
        -------
        decrease_eles : list of float
            Predicted decrease of each element model.
        decrease_wx_m_eles : list of float
            Decrease of the weighted and transformed value of each element model. Note that
            for each element, the default weight is 1.0, and the default transformation
            (xform) is the identity.
        decrease_wx_f_eles : list of float
            Decrease of the weighted and transformed value of each element function.
        fval_wx_eles : list of float
            Weighted and transformed values of each element function at the trial
            point `xk + sk`.
        """
        new_fval_wx_eles = [None for _ in self.ele_names]
        decrease_eles = [None for _ in self.ele_names]
        decrease_wx_m_eles = [None for _ in self.ele_names]
        decrease_wx_f_eles = [None for _ in self.ele_names]
        for ele_idx in self.ele_idxs:
            ele_model = self.model.ele_models[ele_idx]
            old_mval = ele_model.fun_eval(self.proj_onto_ele(xk, ele_idx))
            new_mval = ele_model.fun_eval(self.proj_onto_ele(x_hold, ele_idx))
            decrease_eles[ele_idx] = old_mval - new_mval

            if self.xforms[ele_idx] is not None:
                old_wx_mval = self.weights[ele_idx] * self.xforms[ele_idx][0](old_mval)
                new_wx_mval = self.weights[ele_idx] * self.xforms[ele_idx][0](new_mval)
                old_wx_fval = self.weights[ele_idx] * self.xforms[ele_idx][0](
                    old_fval_eles[ele_idx]
                )
                new_wx_fval = self.weights[ele_idx] * self.xforms[ele_idx][0](fval_eles[ele_idx])
            else:
                old_wx_mval = self.weights[ele_idx] * old_mval
                new_wx_mval = self.weights[ele_idx] * new_mval
                old_wx_fval = self.weights[ele_idx] * old_fval_eles[ele_idx]
                new_wx_fval = self.weights[ele_idx] * fval_eles[ele_idx]

            new_fval_wx_eles[ele_idx] = float(new_wx_fval)
            decrease_wx_m_eles[ele_idx] = float(old_wx_mval - new_wx_mval)
            decrease_wx_f_eles[ele_idx] = float(old_wx_fval - new_wx_fval)

            if self.params("debug.check_nan_fval") and (
                np.any(np.isnan(old_wx_fval))
                or np.any(np.isnan(new_wx_fval))
                or np.any(np.isnan(old_wx_mval))
                or np.any(np.isnan(new_wx_mval))
            ):
                raise ValueError(
                    f"(element {self.ele_names[ele_idx]}) NaN encountered in the return "
                    "value of the surrogate function. "
                    "This may be caused by some numerical error or matrix singularity."
                )

        return decrease_eles, decrease_wx_m_eles, decrease_wx_f_eles, new_fval_wx_eles

    def update_restart_track_info(
        self, xk: np.ndarray, old_grad: np.ndarray, old_hess: np.ndarray
    ) -> None:
        """
        Track gradient and Hessian changes to detect when a restart is needed.

        Parameters
        ----------
        xk : ndarray
            The current iterate of the optimization process.
        old_grad : ndarray, shape (n,)
            Gradient from previous iteration (`self.model.model_grad`).
        old_hess : ndarray, shape (n, n)
            Hessian from previous iteration (`self.model.model_hess`).

        Notes
        -----
        Maintains three circular buffers tracking:
        1. Trust region radius history (`restart_auto_detect_delta`)
        2. Gradient change norms (`restart_auto_detect_chg_grad`)
        3. Hessian change norms (`restart_auto_detect_chg_hess`)
        """
        if not self.params("restarts.use_restarts") or not self.params("restarts.auto_detect"):
            return
        norm_chg_grad = LA.norm(self.model.grad_eval(xk) - old_grad)
        norm_chg_hess = LA.norm(self.model.hess_eval(xk) - old_hess)
        if self.restart_auto_detect_ready:
            # Maintain circular buffers by removing oldest and appending newest
            self.restart_auto_detect_delta = np.append(
                np.delete(self.restart_auto_detect_delta, [0]), self.average_radius
            )
            self.restart_auto_detect_chg_grad = np.append(
                np.delete(self.restart_auto_detect_chg_grad, [0]), norm_chg_grad
            )
            self.restart_auto_detect_chg_hess = np.append(
                np.delete(self.restart_auto_detect_chg_hess, [0]), norm_chg_hess
            )
        else:
            idx = np.argmax(self.restart_auto_detect_delta < 0.0)  # Find first empty slot
            self.restart_auto_detect_delta[idx] = self.average_radius
            self.restart_auto_detect_chg_grad[idx] = norm_chg_grad
            self.restart_auto_detect_chg_hess[idx] = norm_chg_hess
            # Check if all slots are now filled
            self.restart_auto_detect_ready = idx >= len(self.restart_auto_detect_delta) - 1

    def decide_to_restart_due_to_exploding_coeff(self) -> bool:
        """
        Decide whether to restart using the auto-detection strategy.

        Returns
        -------
        bool
            True if restart conditions are met, False otherwise.

        Notes
        -----
        The decision is based on three criteria:
        1. Trust region radius history analysis (flat/down/up trends)
        2. Gradient coefficient growth rate (log-linear regression slope)
        3. Hessian coefficient growth rate (log-linear regression slope)

        A restart is triggered when:
        No radius increases observed (only flat/decreasing)
        Both gradient and Hessian coefficients show significant increasing trends
        Correlation coefficients exceed minimum thresholds
        """
        if (
            (not self.params("restarts.use_restarts"))
            or (not self.params("restarts.auto_detect"))
            or (not self.restart_auto_detect_ready)
        ):
            return False
        iters_delta_flat = np.where(
            np.abs(self.restart_auto_detect_delta[1:] - self.restart_auto_detect_delta[:-1]) < 1e-15
        )[0]
        iters_delta_down = np.where(
            self.restart_auto_detect_delta[1:] - self.restart_auto_detect_delta[:-1] < -1e-15
        )[0]
        iters_delta_up = np.where(
            self.restart_auto_detect_delta[1:] - self.restart_auto_detect_delta[:-1] > 1e-15
        )[0]
        if self.disp >= 3:
            print(
                f"  Found iterations where delta stays flat ({len(iters_delta_flat)}), "
                f"going up ({len(iters_delta_up)}) and going down ({len(iters_delta_down)})."
            )
        if len(iters_delta_up) == 0:
            # Only proceed if no radius increases (only flat/decreasing),
            # then check chg_grad and chg_hess criteria.
            # Fit line to k vs. log(||chg_grad||_2) and log(||chg_hess||_F) separately; both have to increase
            slope, intercept, r_value, _, _ = STAT.linregress(
                np.arange(len(self.restart_auto_detect_chg_grad)),
                np.log(np.maximum(self.restart_auto_detect_chg_grad, 1e-15)),
            )
            slope2, intercept2, r_value2, _, _ = STAT.linregress(
                np.arange(len(self.restart_auto_detect_chg_hess)),
                np.log(np.maximum(self.restart_auto_detect_chg_hess, 1e-15)),
            )
            if self.disp >= 3:
                print(
                    "  Restart Info Report on Grad: (slope, intercept, r_value) = (%g, %g, %g)"
                    % (slope, intercept, r_value)
                )
                print(
                    "  Restart Info Report on Hess: (slope, intercept, r_value) = (%g, %g, %g)"
                    % (slope2, intercept2, r_value2)
                )
            # increasing trend, with at least some positive correlation
            return min(slope, slope2) > self.params(
                "restarts.auto_detect.min_chg_model_slope"
            ) and min(r_value, r_value2) > self.params("restarts.auto_detect.min_correl")
        return False

    def check_slow_iteration(self) -> Tuple[bool, bool]:
        """
        Monitor optimization progress and detect slow convergence patterns.

        Returns
        -------
        bool
            True if current iteration shows insufficient progress (slow iteration),
            False otherwise. Progress is measured by comparing the current function value
            with the value from `history_for_slow` iterations ago.
        bool
            True if maximum allowed consecutive slow iterations (`max_slow_iters`) has been
            reached, indicating potential need for restart/termination, False otherwise.
        """
        self.history_fvals.append(self.interp_set.f_opt)
        if len(self.history_fvals) < self.params("slow.history_for_slow"):
            is_slow = False
        else:
            f_start, f_end = self.history_fvals[0], self.history_fvals[-1]
            bnd = (
                self.params("slow.thresh_for_slow") * (np.abs(f_start) + np.abs(f_end))
                + np.finfo(np.float64).eps
            )
            is_slow = (self.params("slow.thresh_for_slow") >= 0) and (
                2.0 * float(np.abs(f_start - f_end)) <= bnd
            )
            if self.disp >= 3:
                print(
                    f"  Slow iteration: "
                    + ("yes. " if is_slow else "no. ")
                    + f"2 * abs(f_start - f_end) = {2.0 * float(np.abs(f_start - f_end)):.6}, checked bound = {bnd:.6}."
                )
                print(f"  Number of slow iterations: {self.num_slow_iters + 1}")

        # Update consecutive slow iteration counter
        self.num_slow_iters = self.num_slow_iters + 1 if is_slow else 0
        return is_slow, self.num_slow_iters >= self.params("slow.max_slow_iters")

    def prepare_for_a_restart(self, force: bool = False) -> Optional[ExitInfo]:
        """
        Prepare for a restart. This function should be called before launching a restart,
        and the return value is to be checked to determine whether a restart is allowed.

        Parameters
        ----------
        force : bool, default=False
            If True, bypasses all restart checks (for debugging purposes only).

        Returns
        -------
        exit_info : ``~upoqa.utils.ExitInfo`` or None
            None if restart is permitted, otherwise an ``~upoqa.utils.ExitInfo`` object
            containing the reason for preventing restart.
        """
        _, f_opt = self.interp_set.get_opt()
        if f_opt < self.last_run_f_opt:
            self.last_successful_run = self.nrun
        else:
            # Unsuccessful run
            self.total_unsuccessful_restarts += 1
            self.resolution_restart *= self.params(
                "restarts.resolution_restart_scale_after_unsuccessful_restart"
            )

        if self.disp >= 3:
            print(
                f"  total_unsuccessful_restarts = {self.total_unsuccessful_restarts}, "
                f"last_successful_run = {self.last_successful_run}"
            )

        self.last_run_f_opt = f_opt

        ready_to_restart = (
            (
                self.nrun - self.last_successful_run
                < self.params("restarts.max_unsuccessful_restarts")
            )
            and (not self.has_reached_maxfev())
            and self.total_unsuccessful_restarts
            < self.params("restarts.max_unsuccessful_restarts_total")
        )
        ready_to_restart = True if force else ready_to_restart

        if not ready_to_restart:
            # last outputs are (exit_flag, exit_str, return_to_new_tr_iteration)
            exit_info = ExitInfo(ExitStatus.SUCCESS, "Objective has been called MAXFUN times.")
            if self.nrun - self.last_successful_run >= self.params(
                "restarts.max_unsuccessful_restarts"
            ):
                exit_info = ExitInfo(
                    ExitStatus.SUCCESS,
                    "Reached maximum number of consecutive unsuccessful restarts",
                )
            elif self.total_unsuccessful_restarts >= self.params(
                "restarts.max_unsuccessful_restarts_total"
            ):
                exit_info = ExitInfo(
                    ExitStatus.SUCCESS,
                    "Reached maximum total number of unsuccessful restarts",
                )
            return exit_info
        return None

    def get_update_requests(
        self,
        idx_to_replaced_eles: List[int],
        tr_vios: np.ndarray,
        cached_kkt_info_eles: List[Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]],
    ) -> Tuple[List[bool], Optional[ExitInfo]]:
        """
        Determine whether each elemental model and interpolation set need to be updated
        based on the trust region step sizes in each elemental subspace and the given
        intermediate quantities about the determinant when updating the KKT matrix for
        each elemental interpolation system.

        The decision is based on a scoring strategy, which is largely empirical. Updates
        that significantly degrade the numerical stability and the poisedness of the
        interpolation set are rejected, while others are accepted. The threshold is
        controlled by the parameter `general.filter_element_point_thresh`.

        Parameters
        ----------
        idx_to_replaced_eles : list of int
            A list where each value represents the index of an interpolation point in
            the interpolation set of the corresponding element. The algorithm needs to
            decide whether to replace the marked interpolation point with a new one for
            each element.
        tr_vios : ndarray, shape (ele_num,)
            Trust region violation ratios for each element. Trust region violation ratio
            denotes the ratio of the norm of the trust region step to the radius.
        cached_kkt_info_eles : list of tuples
            List of cached outputs of

                ``ele_model.get_determinant_ratio(self.proj_onto_ele(xk + sk, ele_idx))``

            for each element model `ele_model`, where `xk` denotes the current iterate,
            `sk ` denotes the trust-region step.

        Returns
        -------
        flags : list of bool
            A list indicating whether each elemental model and interpolation set need to
            be updated.
        exit_info : ``~upoqa.utils.ExitInfo`` or None
            Return None if any update is accepted. Otherwise, return an ``~upoqa.utils.ExitInfo``
            object with an `ExitStatus.LINALG_ERROR` flag and terminate the algorithm.
        """
        exit_info = None
        need_update = [True for _ in self.ele_idxs]
        if not self.params('general.filter_element_point'):
            return need_update, exit_info

        scores = []
        for ele_idx in self.ele_idxs:
            idx_to_replaced = idx_to_replaced_eles[ele_idx]
            beta, sigma = (
                cached_kkt_info_eles[ele_idx][1],
                cached_kkt_info_eles[ele_idx][3][idx_to_replaced],
            )
            score = sigma * min(1.0, tr_vios[ele_idx] * self.radii[ele_idx] / self.resolution)
            if beta < -1 / np.finfo(float).eps ** 0.25:
                score = -np.inf
            elif beta < -np.finfo(float).eps ** 0.5:
                beta_penalty = (-beta / np.finfo(float).eps ** 0.5) ** 0.6
                score = score / beta_penalty if score >= 0 else score * beta_penalty
            scores.append((ele_idx, float(score)))
        scores.sort(key=lambda x: x[1])

        out_num, max_out_num, success = 0, len(idx_to_replaced_eles) - 1, False

        if self.disp >= 3:
            print(f"  Elemental scores for new point: \n    {dict(scores)}")

        for ele_idx, score in scores:
            # filter from small to large
            if (out_num < max_out_num) and (
                score < self.params('general.filter_element_point_thresh')
            ):
                need_update[ele_idx] = False
                out_num += 1
            else:
                need_update[ele_idx] = True
                success = True

        if not success:
            exit_info = ExitInfo(
                flag=ExitStatus.LINALG_ERROR,
                msg=(
                    "All updates to the element models and interpolation sets are rejected. "
                    "This may be caused by some numerical error or matrix singularity. "
                ),
            )

        return need_update, exit_info

    def soft_restart(
        self, funs: Callable[[np.ndarray], Tuple[List[float], float]]
    ) -> Optional[ExitInfo]:
        """
        Perform a soft restart of the optimization process by resetting key parameters
        and improving the geometry of the interpolation set.

        Parameters
        ----------
        funs : callable
            The element functions and the extra function (white-box component) to evaluate:

                ``funs(x: ndarray) -> tuple[list[float], float]``

            Where `x` is the input vector, and the function returns a tuple containing:

                1. List of element function evaluations (list of floats)
                2. Extra function evaluation (float)

            The element functions should be properly wrapped to be capable of raising an
            exception ``~upoqa.utils.MaxEvalNumReached`` if the maximum number of evaluations
            has been reached.

        Returns
        -------
        exit_info : ``~upoqa.utils.ExitInfo`` or None
            None if restart succeeds, otherwise an ``~upoqa.utils.ExitInfo`` object containing
            the reason for the failure.
        """
        # A successful run is one where we reduced fopt
        x_opt, _ = self.interp_set.get_opt()
        n = x_opt.size

        # Resetting method: reset delta and rho, then move the closest 'num_geom_steps' points to
        # xk to improve geometry.
        # Note: closest points because we are suddenly increasing delta & rho, so we want to encourage
        # spreading out points
        resolution = min(
            self.resolution_init,
            self.params("restarts.resolution_restart_rel_ratio") * self.resolution,
            max(10 * self.resolution, self.resolution_restart),
        )
        for ele_idx in self.ele_idxs:
            self.radii[ele_idx] = resolution
        self.resolution = resolution
        if self.disp >= 2:
            print(f"  Start soft restart, resolution reset to be {resolution}.")

        # Forget history of slow iterations
        self.history_fvals.clear()
        self.num_slow_iters = 0

        # Find closest points for every elemental interpolation set
        closest_points = []
        upper_limit = self.params("restarts.num_geom_steps")
        for ele_idx in self.ele_idxs:
            interp_set = self.ele_interp_sets[ele_idx]
            Y, _ = interp_set.get_interp_set()
            all_sq_dist = LA.norm(Y - self.proj_onto_ele(x_opt, ele_idx), axis=1) ** 2
            closest_points.append(np.argsort(all_sq_dist)[: self.params("restarts.num_geom_steps")])
            upper_limit = min(upper_limit, interp_set.npt)

        def _get_score(ele_scores: np.ndarray) -> float:
            # \Pi_i si / \sum_i si, negative if any si < 0
            has_negative = np.any(ele_scores < -np.finfo(float).eps)
            np.clip(ele_scores, np.finfo(float).eps, np.inf, out=ele_scores)
            return np.prod(ele_scores) / np.sum(ele_scores) * (-1 if has_negative else 1)

        stay_iter, i = 0, 0
        self.interp_set.clear_with_only_one_left()

        while i < upper_limit:
            un_updated_eles = []

            # generate random steps with norm ≈ self.resolution
            trial_points = np.random.randn(self.params("restarts.random_trial_num"), n) / np.sqrt(
                n
            )  # the expectation of the norm is 1
            trial_points *= self.resolution
            trial_points = trial_points + x_opt
            determinant_score = []
            merge_score = np.zeros(trial_points.shape[0])
            for ele_idx in self.ele_idxs:
                model = self.ele_models[ele_idx]
                determinant_score.append(np.zeros(trial_points.shape[0]))
                for j in range(trial_points.shape[0]):
                    determinant_score[ele_idx][j] = model.get_determinant_ratio(
                        self.proj_onto_ele(trial_points[j], ele_idx),
                        idx=closest_points[ele_idx][i],
                    )[3]
            for j in range(trial_points.shape[0]):
                merge_score[j] = _get_score(
                    np.asarray([determinant_score[ele_idx][j] for ele_idx in self.ele_idxs])
                )

            if np.max(merge_score) < 0 and stay_iter < 10:
                stay_iter += 1
                continue

            # the best point among trial_points
            winner_idx = np.argsort(merge_score)[-1]

            # using trial_points[winner_idx] as a new point
            x_new = trial_points[winner_idx]

            winner_sigmas = {
                ele_idx: determinant_score[ele_idx][winner_idx] for ele_idx in self.ele_idxs
            }
            if self.disp >= 3:
                print(
                    f"  Restart candidate point determinant ratios "
                    f"({stay_iter + 1} rolls): {winner_sigmas}"
                )

            try:
                fval_eles, extra_fval = funs(x_new)
                fval = self.build_fval(fval_eles, extra_fval)
            except MaxEvalNumReached as e:
                return ExitInfo(
                    ExitStatus.SUCCESS,
                    f"(element {e.ele_name}) Objective has been called MAXFUN times.",
                )
            except ValueError as e:
                return ExitInfo(
                    ExitStatus.INVALID_EVAL_ERROR,
                    "(During restart) " + str(e),
                    exception=e,
                    traceback=traceback.format_exc(),
                )

            self.interp_set.update_point_on_idx(
                x_new,
                [closest_points[ele_idx][i] for ele_idx in self.ele_idxs],
                fval_eles,
                fval,
                extra_fval,
            )

            for ele_idx in self.ele_idxs:
                if winner_sigmas[ele_idx] < -np.finfo(float).eps:
                    # we do not update an element model if the sigma is bad.
                    un_updated_eles.append(ele_idx)

            try:
                self.model.update(
                    want_update=[
                        (True if ele_idx not in un_updated_eles else False)
                        for ele_idx in self.ele_idxs
                    ],
                )
            except SurrogateLinAlgError as e:
                return ExitInfo(
                    flag=ExitStatus.LINALG_ERROR,
                    msg=f"(During restart) (element {e.ele_name}) " + str(e),
                    exception=e,
                    traceback=traceback.format_exc(),
                )

            x_opt, _ = self.interp_set.get_opt()

            # For those who fail to update, we reinit their surrogate models.
            for ele_idx in un_updated_eles:
                if self.disp >= 3:
                    print(f"  Reinit surrogate model (element {self.ele_names[ele_idx]}).")
                self.ele_models[ele_idx].reinit()

            i += 1
            stay_iter = 0

        # Otherwise, we are doing a restart
        self.nrun += 1
        return None  # exit_info = None

    def find_best_start_point(
        self,
    ) -> Tuple[np.ndarray, float, List[float], float, Optional[ExitInfo]]:
        """
        Refine the user-specified starting point by exploring multiple solutions to a
        Constraint Satisfaction Problem (CSP) using the Minimum Remaining Values (MRV)
        heuristic. In this CSP:

        A *state* is a complete point `x`.

        The *constraints* require that for any element index `ele_idx`, the
            projection `self.proj_onto_ele(x, ele_idx)` must lie within the
            `ele_idx`-th interpolation set.
        """
        trial_point_num = 0
        best_x, best_fval, best_fval_eles, best_extra_fval = None, np.inf, [], np.inf
        neighbours = self.ele_neighbours
        init_remain_selection = [
            [i, list(range(self.npts[self.ele_idxs[i]]))] for i in range(self.ele_num)
        ]

        pbar = None
        if self.disp >= 1:
            from tqdm import tqdm

            pbar = tqdm(
                total=self.params("init.search_opt_x0_max_trials"),
                desc="Searching for better x0...",
                bar_format='{desc}{postfix}',
                # leave=False,
                mininterval=0.2,
            )
            pbar.set_postfix_str(f"point num = 0, best fun = inf")
            pbar.refresh()

        def update_progress():
            if pbar is not None:
                pbar.set_postfix_str(f"point num = {trial_point_num}, best fun = {best_fval}")
                pbar.refresh()

        def _proj_onto_ele_rela_coord(
            source_coord: np.ndarray, target_coord: np.ndarray
        ) -> np.ndarray:
            coord_local = []
            for i in range(target_coord.size):
                if target_coord[i] in source_coord:
                    coord_local.append(i)
            return np.array(coord_local)  # at most size = target_coord.size

        def _get_new_remain_selection(
            x: np.ndarray,
            curr_coord: np.ndarray,
            next_model_i: int,
            remain_selection: List[Tuple[int, List[int]]],
        ) -> List[Tuple[int, List[int]]]:
            remaim_model_num = len(remain_selection)
            new_remain_selection = []
            for k in range(remaim_model_num):
                i, remain_selection_on_i = remain_selection[k]
                if i in neighbours[next_model_i]:
                    ele_idx = self.ele_idxs[i]
                    x_ele = self.proj_onto_ele(x, ele_idx)
                    local_rela_coord = _proj_onto_ele_rela_coord(curr_coord, self.coords[ele_idx])
                    x_ele_used = x_ele[local_rela_coord]
                    new_remain_selection_on_i = []
                    Y, _ = self.ele_interp_sets[ele_idx].get_interp_set()
                    for point_idx in remain_selection_on_i:
                        y_local = Y[point_idx][local_rela_coord]
                        if np.array_equal(x_ele_used, y_local):
                            new_remain_selection_on_i.append(point_idx)
                    new_remain_selection.append([i, new_remain_selection_on_i])
                else:
                    new_remain_selection.append(remain_selection[k])
            return new_remain_selection

        def _dfs(
            x: np.ndarray,
            selected_point_idx: List[int],
            curr_coord: np.ndarray,
            remain_selection: List[Tuple[int, List[int]]],
        ) -> Optional[ExitInfo]:

            nonlocal trial_point_num, best_x, best_fval, best_fval_eles, best_extra_fval

            if trial_point_num >= self.params("init.search_opt_x0_max_trials"):
                return None

            if not remain_selection:
                fval_eles = []
                for ele_idx in self.ele_idxs:
                    point_idx = selected_point_idx[ele_idx]
                    # assert np.array_equal(self.proj_onto_ele(x, ele_idx), self.ele_interp_sets[ele_idx].get_interp_set(idx = point_idx)[0])
                    fval_eles.append(self.ele_interp_sets[ele_idx].get_interp_set(idx=point_idx)[1])
                try:
                    extra_fval = float(self.extra_fun[0](x))
                except Exception as e:
                    return ExitInfo(
                        ExitStatus.INVALID_EVAL_ERROR,
                        '(When searching the optimal starting point) ' + str(e),
                        exception=e,
                        traceback=traceback.format_exc(),
                    )
                fval = self.build_fval(fval_eles, extra_fval)
                if fval < best_fval:
                    best_x, best_fval, best_fval_eles, best_extra_fval = (
                        x.copy(),
                        fval,
                        fval_eles,
                        extra_fval,
                    )
                trial_point_num += 1
                if self.disp >= 1:
                    update_progress()
                return None

            remain_selection_num = np.array([len(x[1]) for x in remain_selection], dtype=np.int32)
            next_model_raw_i = remain_selection_num.argmin(axis=0)
            next_model_i, remain_selection_point_idx = remain_selection[next_model_raw_i]
            next_ele_idx = self.ele_idxs[next_model_i]
            next_coord = self.coords[next_ele_idx]
            new_coord = np.unique(np.hstack((curr_coord, next_coord)))
            remain_selection = (
                remain_selection[0:next_model_raw_i] + remain_selection[next_model_raw_i + 1 :]
            )
            new_selected_point_idx = deepcopy(selected_point_idx)
            next_interp_set, next_interp_set_fval = self.ele_interp_sets[
                next_ele_idx
            ].get_interp_set()
            rs_point_idx_and_local_fval = [
                (y_idx, next_interp_set_fval[y_idx]) for y_idx in remain_selection_point_idx
            ]
            rs_point_idx_and_local_fval.sort(key=lambda x: x[1])
            for y_idx, _ in rs_point_idx_and_local_fval:
                y = next_interp_set[y_idx]
                new_x = x.copy()
                new_x[next_coord] = y
                new_selected_point_idx[next_ele_idx] = y_idx
                new_rs = _get_new_remain_selection(new_x, new_coord, next_model_i, remain_selection)
                exit_info = _dfs(new_x, new_selected_point_idx, new_coord, new_rs)
                if exit_info:
                    return exit_info
                elif trial_point_num >= self.params("init.search_opt_x0_max_trials"):
                    return None

        exit_info = _dfs(
            np.zeros(self.n),
            [None for _ in self.ele_idxs],
            np.array([], dtype=np.int32),
            init_remain_selection,
        )
        if pbar is not None:
            pbar.set_description_str(f"Searching for better x0... done")
            pbar.set_postfix_str(f"point num = {trial_point_num}, best fun = {best_fval}")
            time.sleep(0.3)
            pbar.close()
        return best_x, best_fval, best_fval_eles, best_extra_fval, exit_info
