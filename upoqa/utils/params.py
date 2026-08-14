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
Parameters
==========

A container for all parameter values.
"""

import numpy as np
from typing import Any, Dict, Optional, Union, Tuple, List, Literal

__all__ = [
    "UPOQAParameterList",
]


class ParameterList(object):
    """
    Base class representing a parameter list.
    """

    def __init__(self) -> None:
        self.params: Dict[str, Any] = {}
        self.params_changed: Dict[str, bool] = {}

    def __call__(
        self, key: str, new_value: Optional[Union[float, int, str]] = None
    ) -> Union[float, int, str]:
        """
        Access or update a parameter value.

        Parameters
        ----------
        key : str
            The name of the parameter.
        new_value : float or int or str, optional
            The new value to assign to the parameter. If None, the current value
            is returned.

        Returns
        -------
        float or int or str
            The current or updated value of the parameter.

        Raises
        ------
        ValueError
            If the parameter does not exist or is being updated for a second time.
        """
        if key in self.params:
            if new_value is None:
                return self.params[key]
            else:
                if self.params_changed[key]:
                    raise ValueError(
                        "Trying to update parameter '%s' for a second time" % key
                    )
                self.params[key] = new_value
                self.params_changed[key] = True
                return self.params[key]
        else:
            raise ValueError("Unknown parameter '%s'" % key)

    def param_type(self, key=str) -> Tuple[Optional[str], bool, Any, Any]:
        # Use the check_* methods below, but switch based on key
        return None, False, None, None

    def check_param(self, key: str, value: Union[float, int, str]) -> bool:
        type_str, nonetype_ok, lower, upper = self.param_type(key)
        if type_str == "int":
            val_ok, converted_val = check_integer(
                value, lower=lower, upper=upper, allow_nonetype=nonetype_ok
            )
        elif type_str == "float":
            val_ok, converted_val = check_float(
                value, lower=lower, upper=upper, allow_nonetype=nonetype_ok
            )
        elif type_str == "bool":
            val_ok, converted_val = check_bool(value, allow_nonetype=nonetype_ok)
        elif type_str == "str":
            val_ok, converted_val = check_str(value, allow_nonetype=nonetype_ok)
        else:
            assert False, "Unknown type_str '%s' for parameter '%s'" % (type_str, key)
        if converted_val is not None:
            self.params[key] = converted_val
        return val_ok

    def check_all_params(self) -> Tuple[bool, list]:
        """
        Check the types and boundary conditions of all parameters.
        """
        bad_keys = []
        for key in self.params:
            if not self.check_param(key, self.params[key]):
                bad_keys.append(key)
        return len(bad_keys) == 0, bad_keys

    def params_start_with(self, prefix="") -> Dict[str, Any]:
        """
        Return a dictionary of all parameters that start with the given prefix.
        """
        para = {}
        for key, value in self.params.items():
            if key.startswith(prefix):
                para[key] = value
        return para

    def __str__(self) -> str:
        return str(self.params)


class UPOQAParameterList(ParameterList):
    """
    Parameter list for UPOQA.

    Parameters
    ----------
    ele_names : list
        List containing the names of all elements in order.
    ele_dims : list of int
        Elemental dimensions of the problem.
    maxfev : list
        Maximum number of function evaluations to go before termination for each element.
    noise_level : int, default=0
        Indicates how noisy the objective function is. Should be 0, 1, or 2.
        Default number of interpolation points (``npt``) increases as the ``noise_level``
        increases, and when ``noise_level`` > 0, restart mechanism is enabled.
    seek_global_minimum : bool, default=False
        Whether to seek a global minimum. Defaults to False. If True, restart mechanism
        is enabled, but the parameters for restart are different.
    debug : bool, default=False
        Whether to enable debug mode. Defaults to False. If True, the algorithm will check
        for NaN values in the objective function and the surrogate model.

        Note that NaN check will incur additional runtime costs.
    """

    def __init__(
        self,
        ele_names: List[Any],
        ele_dims: List[int],
        maxfev: List[Union[float, int]],
        noise_level: int = 0,
        seek_global_minimum: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.ele_names = ele_names

        ## General
        # Number of interpolation points, 2 * n + 1 by default, and slightly increased
        # when the objective function is noisy.
        self.params["general.npt"] = []  # list of npt for each element
        for ele_dim in ele_dims:
            max_npt = (ele_dim + 1) * (ele_dim + 2) // 2
            if noise_level == 1:
                self.params["general.npt"].append(
                    min(
                        max_npt,
                        max(
                            2 * ele_dim + 1,
                            int(np.floor(0.8 * ele_dim**1.5 + ele_dim + 1)),
                        ),
                    )
                )
            elif noise_level == 2:
                self.params["general.npt"].append(max_npt)
            else:
                self.params["general.npt"].append(2 * ele_dim + 1)
        # decide when to shift surrogate model center according to norm(center - xk)
        self.params["general.center_shift_threshold"] = 5.0
        # maximum tolerable short steps
        self.params["general.max_short_steps"] = 5 + 2 * noise_level
        # if the step size is smaller than `general.short_step_thresh` * `resolution`,
        # it is considered as a short step.
        self.params["general.short_step_thresh"] = 0.5
        # maximum tolerable very short steps
        self.params["general.max_very_short_steps"] = 3 + 1 * noise_level
        # if all the elemental step sizes are smaller than `general.very_short_step_thresh` * `resolution`,
        # it is considered as a very short step.
        self.params["general.very_short_step_thresh"] = 0.1
        # maximum tolerable steps for switching to an alternative surrogate model
        self.params["general.max_alt_model_steps"] = 3
        # decide whether we want to switch to an alternative model according to norm of grad_k
        self.params["general.alt_model_thresh"] = 10.0
        # only when reduction ratio < alt_model_check_thresh, can we consider switching to an alternative model
        self.params["general.alt_model_check_thresh"] = 0.01
        # when updating sub-interpolation-set, a new point may be declined if it's too close to the center
        # and provides bad update on the interpolation system.
        self.params["general.filter_element_point"] = True
        # the threshold on sigma of an acceptable new point.
        self.params["general.filter_element_point_thresh"] = 1e-5
        # the maximum size of the overall interpolation point set will be params["general.overall_interp_set_size_factor"] * dimension of the problem.
        self.params["general.overall_interp_set_size_factor"] = 10.0

        ## Initialisation
        # whether to search for the optimal start point, otherwise we will use x0 as the start point
        self.params["init.search_opt_x0"] = True
        # the maximum number of trial points to be tested when searching for the optimal start point
        self.params["init.search_opt_x0_max_trials"] = 100
        # the initialized size of the overall interpolation point set
        self.params["init.overall_interp_set_size"] = 10

        ## Debug
        # if NaN detected in the return value of the objective function or the surrogate model,
        # raise error and terminate the algorithm immediately.
        # Note that NaN check will incur additional runtime costs.
        self.params["debug.check_nan_fval"] = True if debug else False

        ## Trust Region Radius Management
        # eta1 and eta2 are thresholds of reduction ratio when updating radius.
        self.params["tr_radius.eta1"] = 0.1
        self.params["tr_radius.eta2"] = 0.7
        # theta1 ~ theta6 are factors to be multiplied onto radius when updating radius.
        self.params["tr_radius.theta1"] = 0.98 if noise_level else 0.5
        self.params["tr_radius.theta2"] = 1.05
        self.params["tr_radius.theta3"] = 2.0**0.5
        self.params["tr_radius.theta4"] = 2.0
        self.params["tr_radius.theta5"] = 0.5**0.5
        self.params["tr_radius.theta6"] = 10000.0
        # when reduce resolution, new_resolution = alpha1 * old_resolution
        self.params["tr_radius.alpha1"] = 0.9 if noise_level else 0.1
        # when reduce resolution or encounter a short step, new_radius = alpha2 * old_radius
        self.params["tr_radius.alpha2"] = 0.95 if noise_level else 0.5
        # when launching a geometry-improvement step, the shrinkage factor for the radius.
        self.params["tr_radius.alpha3"] = 0.1

        ## Slow Progress Thresholds
        # whether to terminate the algorithm when the iterations are slow.
        self.params["slow.terminate_when_slow"] = (
            True if noise_level or seek_global_minimum else False
        )
        # length of timing window for checking slow iteration
        self.params["slow.history_for_slow"] = 5
        # The iteration is slow if the relative difference between f[k - history_for_slow] and f[k] is < thresh_for_slow.
        # (the slow iteration check rule is disabled when thresh_for_slow is negative)
        self.params["slow.thresh_for_slow"] = 1e-8
        # maximum tolerable slow iterations
        self.params["slow.max_slow_iters"] = (
            int(20 * max(ele_dims)) if seek_global_minimum else int(10 * max(ele_dims))
        )

        ## Restarts
        # whether to enable restarts, enabled by default when the objective is noisy or we seek a global minimum.
        self.params["restarts.use_restarts"] = (
            True if (noise_level or seek_global_minimum) else False
        )
        self.params["restarts.max_unsuccessful_restarts"] = 10
        self.params["restarts.max_unsuccessful_restarts_total"] = (
            20 if seek_global_minimum else max(maxfev)
        )
        self.params["restarts.resolution_final_scale"] = (
            1.0  # how much to decrease resolution_final by after each restart
        )
        # a restart will reset the resolution to be no more than resolution_restart_init_ratio * radius_init
        self.params["restarts.resolution_restart_init_ratio"] = 1.0
        # a restart will reset the resolution to be no more than resolution_restart_rel_ratio * resolution
        self.params["restarts.resolution_restart_rel_ratio"] = 1e3
        self.params["restarts.random_trial_num"] = 10
        self.params["restarts.resolution_restart_scale_after_unsuccessful_restart"] = (
            1.1 if seek_global_minimum else 1.0
        )
        # how many new points are needed to launch a soft restart
        self.params["restarts.num_geom_steps"] = 5 + 5 * max(noise_level - 1, 0)
        # enable auto detect for restart or not
        self.params["restarts.auto_detect"] = True
        self.params["restarts.auto_detect.history"] = 30
        # a restart will be called if the norm of the change of model_grad shows greater slope than min_chg_model_slope
        self.params["restarts.auto_detect.min_chg_model_slope"] = 1.5e-2
        # a restart will be called if the norm of the change of model_hess shows greater slope than min_correl
        self.params["restarts.auto_detect.min_correl"] = 0.1

        for p in self.params:
            self.params_changed[p] = False

    def param_type(self, key=str) -> Tuple[str, bool, Any, Any]:
        # Use the check_* methods below, but switch based on key
        if key == "general.center_shift_threshold":
            type_str, nonetype_ok, lower, upper = "float", False, 0.0, None
        elif key == "general.max_short_steps":
            type_str, nonetype_ok, lower, upper = "int", False, 1, None
        elif key == "general.short_step_thresh":
            type_str, nonetype_ok, lower, upper = "float", False, 0.0, 1.0
        elif key == "general.max_very_short_steps":
            type_str, nonetype_ok, lower, upper = "int", False, 1, None
        elif key == "general.very_short_step_thresh":
            type_str, nonetype_ok, lower, upper = "float", False, 0.0, 1.0
        elif key == "general.max_alt_model_steps":
            type_str, nonetype_ok, lower, upper = "int", False, 1, None
        elif key == "general.alt_model_thresh":
            type_str, nonetype_ok, lower, upper = "float", False, 1.0, None
        elif key == "general.alt_model_check_thresh":
            type_str, nonetype_ok, lower, upper = "float", False, None, None
        elif key == "general.filter_element_point":
            type_str, nonetype_ok, lower, upper = "bool", False, None, None
        elif key == "general.filter_element_point_thresh":
            type_str, nonetype_ok, lower, upper = "float", False, 0.0, None
        elif key == "general.overall_interp_set_size_factor":
            type_str, nonetype_ok, lower, upper = "float", False, 1.0, None
        elif key == "init.search_opt_x0":
            type_str, nonetype_ok, lower, upper = "bool", False, None, None
        elif key == "init.search_opt_x0_max_trials":
            type_str, nonetype_ok, lower, upper = "int", False, 0, None
        elif key == "init.overall_interp_set_size":
            type_str, nonetype_ok, lower, upper = "int", False, 1, None
        elif key == "debug.check_nan_fval":
            type_str, nonetype_ok, lower, upper = "bool", False, None, None
        elif key == "tr_radius.eta1":
            type_str, nonetype_ok, lower, upper = "float", False, 0.0, 1.0
        elif key == "tr_radius.eta2":
            type_str, nonetype_ok, lower, upper = "float", False, 0.0, 1.0
        elif key == "tr_radius.theta1":
            type_str, nonetype_ok, lower, upper = "float", False, 0.0, 1.0
        elif key == "tr_radius.theta2":
            type_str, nonetype_ok, lower, upper = "float", False, 1.0, None
        elif key == "tr_radius.theta3":
            type_str, nonetype_ok, lower, upper = "float", False, 1.0, None
        elif key == "tr_radius.theta4":
            type_str, nonetype_ok, lower, upper = "float", False, 1.0, None
        elif key == "tr_radius.theta5":
            type_str, nonetype_ok, lower, upper = "float", False, 0.0, 1.0
        elif key == "tr_radius.theta6":
            type_str, nonetype_ok, lower, upper = "float", False, 1.0, None
        elif key == "tr_radius.alpha1":
            type_str, nonetype_ok, lower, upper = "float", False, 0.0, 1.0
        elif key == "tr_radius.alpha2":
            type_str, nonetype_ok, lower, upper = "float", False, 0.0, 1.0
        elif key == "tr_radius.alpha3":
            type_str, nonetype_ok, lower, upper = "float", False, 0.0, 1.0
        elif key == "slow.terminate_when_slow":
            type_str, nonetype_ok, lower, upper = "bool", False, None, None
        elif key == "slow.history_for_slow":
            type_str, nonetype_ok, lower, upper = "int", False, 0, None
        elif key == "slow.thresh_for_slow":
            type_str, nonetype_ok, lower, upper = "float", False, 0, None
        elif key == "slow.max_slow_iters":
            type_str, nonetype_ok, lower, upper = "int", False, 0, None
        elif key == "restarts.use_restarts":
            type_str, nonetype_ok, lower, upper = "bool", False, None, None
        elif key == "restarts.max_unsuccessful_restarts":
            type_str, nonetype_ok, lower, upper = "int", False, 0, None
        elif key == "restarts.max_unsuccessful_restarts_total":
            type_str, nonetype_ok, lower, upper = "int", False, 0, None
        elif key == "restarts.resolution_final_scale":
            type_str, nonetype_ok, lower, upper = "float", False, 0.0, None
        elif key == "restarts.resolution_restart_init_ratio":
            type_str, nonetype_ok, lower, upper = "float", False, 0.0, None
        elif key == "restarts.resolution_restart_rel_ratio":
            type_str, nonetype_ok, lower, upper = "float", False, 0.0, None
        elif key == "restarts.random_trial_num":
            type_str, nonetype_ok, lower, upper = "int", False, 1, None
        elif key == "restarts.resolution_restart_scale_after_unsuccessful_restart":
            type_str, nonetype_ok, lower, upper = "float", False, 0.0, None
        elif key == "restarts.num_geom_steps":
            type_str, nonetype_ok, lower, upper = "int", False, 0, None
        elif key == "restarts.auto_detect":
            type_str, nonetype_ok, lower, upper = "bool", False, None, None
        elif key == "restarts.auto_detect.history":
            type_str, nonetype_ok, lower, upper = "int", False, 1, None
        elif key == "restarts.auto_detect.min_chg_model_slope":
            type_str, nonetype_ok, lower, upper = "float", False, 0.0, None
        elif key == "restarts.auto_detect.min_correl":
            type_str, nonetype_ok, lower, upper = "float", False, 0.0, 1.0
        else:
            assert False, "ParameterList.param_type() has unknown key: %s" % key
        return type_str, nonetype_ok, lower, upper

    def __call__(
        self, key: str, new_value: Optional[Union[dict, list, float, int, str]] = None
    ) -> Union[float, int, str]:
        """
        Access or update a parameter value.

        Parameters
        ----------
        key : str
            The name of the parameter.
        new_value : dict or list or float or int or str, optional
            The new value to assign to the parameter. If None, the current value
            is returned.

        Returns
        -------
        dict or list or float or int or str
            The current or updated value of the parameter.

        Raises
        ------
        ValueError
            If the parameter does not exist or is being updated for a second time.
        """
        if key == "general.npt":
            # Only update `general.npt`, not overwrite
            if new_value is None:
                return self.params["general.npt"]
            else:
                # Check the type of `new_value`
                if isinstance(new_value, list):
                    if len(new_value) != len(self.ele_names):
                        raise ValueError(
                            f"Invalid length for `general.npt`: {len(new_value)}"
                        )
                    for ele_idx, npt in enumerate(new_value):
                        self.params["general.npt"][ele_idx] = npt
                elif isinstance(new_value, dict):
                    for ele_idx, ele_name in enumerate(self.ele_names):
                        if ele_name in new_value:
                            self.params["general.npt"][ele_idx] = new_value[ele_name]
                else:
                    raise ValueError(
                        f"Invalid type for `general.npt`: {type(new_value)}"
                    )
                self.params_changed["general.npt"] = True
                return self.params["general.npt"]
        else:
            return super().__call__(key, new_value)

    def check_all_params(self) -> Tuple[bool, list]:
        """
        Check the types and boundary conditions of all parameters.
        """
        bad_keys = []
        for key in self.params:
            if (key != "general.npt") and (not self.check_param(key, self.params[key])):
                # no check for `general.npt`, whose check is done in `upoqa.utils.preprocess`
                bad_keys.append(key)
        return len(bad_keys) == 0, bad_keys


def check_integer(
    val: int,
    lower: Optional[int] = None,
    upper: Optional[int] = None,
    allow_nonetype: bool = False,
) -> Tuple[bool, Optional[int]]:
    """
    Check that val is an integer (or convertible to an integer) and (optionally) that lower <= val <= upper
    """
    if val is None:
        return allow_nonetype, None
    elif isinstance(val, int):
        return (lower is None or val >= lower) and (upper is None or val <= upper), None
    else:
        try:
            if val not in [np.inf, -np.inf]:
                converted_val = int(val)  # Try converting val to an integer
            else:
                converted_val = val
            return (lower is None or converted_val >= lower) and (
                upper is None or converted_val <= upper
            ), converted_val
        except (ValueError, TypeError):
            return False, None


def check_float(
    val: float,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    allow_nonetype: bool = False,
) -> Tuple[bool, Optional[float]]:
    """
    Check that val is a float (or convertible to a float) and (optionally) that lower <= val <= upper
    """
    if val is None:
        return allow_nonetype, None
    elif isinstance(val, float):
        return (lower is None or val >= lower) and (upper is None or val <= upper), None
    else:
        try:
            if val not in [np.inf, -np.inf]:
                converted_val = float(val)  # Try converting val to a float
            else:
                converted_val = val
            return (lower is None or converted_val >= lower) and (
                upper is None or converted_val <= upper
            ), converted_val
        except (ValueError, TypeError):
            return False, None


def check_bool(val: bool, allow_nonetype: bool = False) -> Tuple[bool, Optional[bool]]:
    """
    Check that val is a bool (or convertible to a bool)
    """
    if val is None:
        return allow_nonetype, None
    elif isinstance(val, bool):
        return True, None
    else:
        try:
            converted_val = bool(val)  # Try converting val to a bool
            return True, converted_val
        except (ValueError, TypeError):
            return False, None


def check_str(val: str, allow_nonetype: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Check that val is a str (or convertible to a str)
    """
    if val is None:
        return allow_nonetype, None
    elif isinstance(val, str):
        return True, None
    else:
        try:
            converted_val = str(val)  # Try converting val to a str
            return True, converted_val
        except (ValueError, TypeError):
            return False, None
