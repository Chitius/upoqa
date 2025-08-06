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
Input Parameter Pre-Processor
=============================
"""

import numpy as np
import traceback
import inspect
import warnings
from .params import UPOQAParameterList
from .manager import ExitInfo, ExitStatus
from typing import List, Any, Dict, Tuple, Optional, Callable, Union
from functools import reduce


def get_reorganized_inputs_or_exit_info(
    fun,
    n,
    maxiter,
    maxfev,
    npt,
    coords,
    weights,
    xforms,
    xform_bounds,
    extra_fun,
    radius_init,
    radius_final,
    noise_level,
    seek_global_minimum,
    callback,
    debug,
    tr_shape,
    options,
) -> Union[ExitInfo, Dict[str, Any]]:
    """Preprocess the input parameters of UPOQA."""
    # Initialize and check the settings of sub functions
    default_ele_name = "fun"
    # default_xform = [lambda X: X, lambda _: 1, lambda _: 0]  # fun, grad, hess
    default_xform = None
    if callable(fun):
        ele_names, ele_idxs = [
            default_ele_name,
        ], [
            0,
        ]
        fun = {default_ele_name: fun}
        # if `xforms` is [fun, grad, hess], then we unsqueeze it to a dict.
        xforms = default_xform if not xforms else xforms
        xforms = (
            {default_ele_name: list(xforms)}
            if (xforms and isinstance(xforms, (list, tuple)) and len(xforms) == 3)
            else xforms
        )
        xform_bounds = (
            {default_ele_name: tuple(xform_bounds)}
            if (
                isinstance(xform_bounds, (list, tuple, np.ndarray))
                and len(xform_bounds) == 2
            )
            else xform_bounds
        )
    elif isinstance(fun, dict):
        ele_names = list(fun.keys())
        ele_idxs = list(range(len(ele_names)))
    elif isinstance(fun, list):
        ele_names = ["fun[" + str(i) + "]" for i in range(len(fun))]
        ele_idxs = list(range(len(ele_names)))
        fun, _ = change_list_to_dict(fun, ele_names)
    else:
        return ExitInfo(
            ExitStatus.INPUT_ERROR,
            f"Invalid type ({type(fun)}) of fun = {fun}, which should be a dictionary, list or callable function.",
        )

    num_model = len(ele_names)
    if num_model == 0:
        return ExitInfo(
            ExitStatus.INPUT_ERROR,
            "Number of objective functions cannot be zero, please check the validness of `fun` parameter.",
        )

    if num_model == 1:
        msg = (
            "An appropriate `fun` for upoqa is a dict or list indicating the partially separable "
            "structure of the objective function. "
            "If your objective is not partially separable, we suggest adopting optimizers "
            "aimed at a single objective function."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    if not isinstance(tr_shape, str):
        return ExitInfo(
            ExitStatus.INPUT_ERROR,
            f"`tr_shape` must be a string, but get {type(tr_shape)}",
        )

    avail_shapes = ["structured", "spherical"]
    tr_shape = tr_shape.lower()
    if tr_shape not in avail_shapes:
        return ExitInfo(
            ExitStatus.INPUT_ERROR,
            f"`tr_shape` must be in {avail_shapes}",
        )

    # check coords and make it a dict
    exit_info, coords = _preprocess(coords, "coords", np.arange(n), ele_names)
    if exit_info:
        return exit_info

    # check weights and make it a dict
    weights = (
        get_default_valued_dict(ele_names, weights)
        if isinstance(weights, (int, float))
        else weights
    )
    exit_info, weights = _preprocess(weights, "weights", 1.0, ele_names)
    if exit_info:
        return exit_info

    # check npt and make it a dict
    npt = (
        get_default_valued_dict(ele_names, npt)
        if isinstance(npt, (int, float))
        else npt
    )
    exit_info, npt = _preprocess(npt, "npt", None, ele_names)
    if exit_info:
        return exit_info

    # check maxfev and make it a dict
    maxfev = (
        get_default_valued_dict(ele_names, maxfev)
        if isinstance(maxfev, (int, float))
        else maxfev
    )
    exit_info, maxfev = _preprocess(maxfev, "maxfev", np.inf, ele_names)
    if exit_info:
        return exit_info

    # check xforms and make it a dict
    exit_info, xforms = _preprocess(xforms, "xforms", default_xform, ele_names)
    if exit_info:
        return exit_info

    # check xform_bounds and make it a dict
    exit_info, xform_bounds = _preprocess(
        xform_bounds, "xform_bounds", (-np.inf, np.inf), ele_names
    )
    if exit_info:
        return exit_info

    # By now, parameters have become dicts, but they may have invalid (key, value).
    fun, exit_info = check_dict_validness_and_sort(
        fun, ele_names, "fun", False, None, False, Callable
    )
    is_extra_fun_scalar = False
    if exit_info is None:
        # extra_fun do not need to support parallel evaluation.
        if isinstance(extra_fun, (int, float)):
            is_extra_fun_scalar = True
            extra_fun = [
                lambda _: float(extra_fun),
                lambda _: np.zeros((n,), dtype=np.float64),
                lambda _: np.zeros((n, n), dtype=np.float64),
            ]
        if not extra_fun:
            is_extra_fun_scalar = True
            extra_fun = [
                lambda _: 0.0,
                lambda _: np.zeros((n,), dtype=np.float64),
                lambda _: np.zeros((n, n), dtype=np.float64),
            ]
        if len(extra_fun) != 3:
            return ExitInfo(
                ExitStatus.INPUT_ERROR,
                f"Invalid `extra_fun` parameter: {extra_fun}, which should be of length 3.",
            )
        elif not isinstance(extra_fun, (list, tuple)):
            return ExitInfo(
                ExitStatus.INPUT_ERROR,
                f"Invalid `extra_fun` parameter: {extra_fun}, which should be a list or a tuple.",
            )
        for i in range(3):
            if not isinstance(extra_fun[i], Callable):
                return ExitInfo(
                    ExitStatus.INPUT_ERROR,
                    f"Invalid extra_fun[{i}] parameter: {extra_fun[i]}, which should be a callable function.",
                )

    if exit_info is None:
        coords, exit_info = check_dict_validness_and_sort(
            coords,
            ele_names,
            "coords",
            True,
            np.arange(n),
            True,
            (np.ndarray, list),
        )
        if exit_info is None:
            for ele_name in ele_names:
                if isinstance(coords[ele_name], list):
                    coords[ele_name] = np.array(coords[ele_name], dtype=np.int64)
                else:
                    coords[ele_name] = coords[ele_name].astype(np.int64)
                if len(coords[ele_name]) != len(set(coords[ele_name])):
                    return ExitInfo(
                        ExitStatus.INPUT_ERROR,
                        f"coords[{ele_name}] ({coords[ele_name]}) contains duplicate coordinates, "
                        "please shrink it to a lower dimensional representation.",
                    )

                if coords[ele_name].size == 0:
                    return ExitInfo(
                        ExitStatus.INPUT_ERROR,
                        f"coords[{ele_name}] = {coords[ele_name]} is empty!",
                    )
    if exit_info is None:
        weights, exit_info = check_dict_validness_and_sort(
            weights, ele_names, "weights", True, 1.0, True, (float, int)
        )
        if exit_info is None:
            for ele_name in ele_names:
                if weights[ele_name] < 0.0:
                    return ExitInfo(
                        ExitStatus.INPUT_ERROR,
                        f"Invalid weights[{ele_name}] = {weights[ele_name]}, which should be non-negative.",
                    )

    if exit_info is None:
        xforms, exit_info = check_dict_validness_and_sort(
            xforms,
            ele_names,
            "xforms",
            True,
            default_xform,
            False,
            (tuple, list),
        )
        if exit_info is None:
            for ele_name in ele_names:
                if xforms[ele_name] is None:
                    continue
                if len(xforms[ele_name]) != 3:
                    return ExitInfo(
                        ExitStatus.INPUT_ERROR,
                        f"Invalid xforms[{ele_name}] parameter: {xforms[ele_name]}, which should be of length 3.",
                    )
                for i in range(3):
                    if not isinstance(xforms[ele_name][i], Callable):
                        return ExitInfo(
                            ExitStatus.INPUT_ERROR,
                            f"Invalid xforms[{ele_name}][{i}] parameter: {xforms[ele_name][i]},"
                            "which should be a callable function.",
                        )
    if exit_info is None:
        npt, exit_info = check_dict_validness_and_sort(
            npt, ele_names, "npt", True, None, False, int
        )
    if exit_info is None:
        maxfev, exit_info = check_dict_validness_and_sort(
            maxfev, ele_names, "maxfev", True, np.inf, True, (int, float)
        )
    if exit_info is None:
        xform_bounds, exit_info = check_dict_validness_and_sort(
            xform_bounds,
            ele_names,
            "xform_bounds",
            True,
            (-np.inf, np.inf),
            True,
            tuple,
        )
        if exit_info is None:
            for ele_name in ele_names:
                if (
                    len(xform_bounds[ele_name]) == 2
                    and isinstance(xform_bounds[ele_name][0], (int, float))
                    and isinstance(xform_bounds[ele_name][1], (int, float))
                ):
                    if xform_bounds[ele_name][1] < xform_bounds[ele_name][0]:
                        return ExitInfo(
                            ExitStatus.INPUT_ERROR,
                            f"Invalid xform_bounds[{ele_name}]: {xform_bounds[ele_name]}. "
                            "The lower bound cannot exceed the upper bound!",
                        )
                else:
                    return ExitInfo(
                        ExitStatus.INPUT_ERROR,
                        f"Invalid type or length of xform_bounds[{ele_name}]: {xform_bounds[ele_name]}.",
                    )
    if (exit_info is None) and (noise_level not in [0, 1, 2]):
        return ExitInfo(
            ExitStatus.INPUT_ERROR,
            f"Invalid `noise_level` parameter: {noise_level}, which should be 0, 1, or 2.",
        )

    if (
        exit_info is None
    ) and is_extra_fun_scalar:  # If `extra_fun` is a function (dependent on all variables of x), disable this check.
        full_coord = reduce(np.union1d, coords.values())
        should_be_full_coord = list(range(0, n))
        if (len(should_be_full_coord) != len(full_coord)) or np.any(
            [(full_coord[i] != should_be_full_coord[i]) for i in range(n)]
        ):
            return ExitInfo(
                ExitStatus.INPUT_ERROR,
                f"Members of coords cannot constitute [0, ..., {n - 1}] by union operations.",
            )

    if exit_info:
        return exit_info

    if maxiter is None:
        max_maxfev = max(list(maxfev.values()))
        if max_maxfev != np.inf:
            maxiter = int(10 * max_maxfev)
        else:
            maxiter = 1000 * n

    # Dimension for each sub-fun
    dims = [len(coords[ele_names[ele_idx]]) for ele_idx in ele_idxs]

    # Initialize the Parameter List
    params = UPOQAParameterList(
        ele_names=ele_names,
        ele_dims=dims,
        maxfev=list(maxfev.values()),
        noise_level=noise_level,
        seek_global_minimum=seek_global_minimum,
        debug=debug,
    )

    if options:
        for key, val in options.items():
            try:
                params(key, new_value=val)
            except ValueError as e:
                return ExitInfo(
                    ExitStatus.INPUT_ERROR,
                    "(When initializing hyper-parameters) "
                    + f"{e}\n\n{traceback.format_exc()}",
                )

    # Set npt
    for ele_name in ele_names:
        ele_dim = len(coords[ele_name])
        max_npt = (ele_dim + 1) * (ele_dim + 2) // 2
        min_npt = ele_dim + 1
        if npt[ele_name] is not None:
            if npt[ele_name] > max_npt:
                warning_msg = (
                    f"(element {ele_name}) "
                    f"The number of interpolation points (which is npt = {npt[ele_name]}) is "
                    f"too large, and has been adjusted to (n + 1) * (n + 2) / 2 = {max_npt}. "
                )
                warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)
                npt[ele_name] = max_npt
            elif npt[ele_name] < min_npt:
                warning_msg = (
                    f"(element {ele_name}) "
                    f"The number of interpolation points (which is npt = {npt[ele_name]}) is  "
                    f"too small, and has been set to n + 1 = {min_npt}. "
                )
                warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)
                npt[ele_name] = min_npt
        else:
            npt[ele_name] = params("general.npt")[ele_names.index(ele_name)]

        if maxfev[ele_name] <= npt[ele_name]:
            return ExitInfo(
                ExitStatus.INPUT_ERROR,
                f"(element {ele_name}) maxfev (which is {maxfev[ele_name]}) "
                f"must be larger than npt (which is {npt[ele_name]}).",
            )

    params("general.npt", npt)

    # Check the validness of other inputs
    radius_init, radius_final = float(radius_init), float(radius_final)
    resolution_init, resolution_final = radius_init, radius_final
    if resolution_init < 0:
        return ExitInfo(
            ExitStatus.INPUT_ERROR, "radius_init must be strictly positive."
        )
    if resolution_final < 0:
        return ExitInfo(
            ExitStatus.INPUT_ERROR, "radius_final must be strictly positive."
        )
    if resolution_init <= resolution_final:
        return ExitInfo(
            ExitStatus.INPUT_ERROR, "radius_init must be larger than radius_final"
        )
    if maxiter < 0:
        return ExitInfo(ExitStatus.INPUT_ERROR, "maxiter cannot be negative.")
    if callback is not None and (not callable(callback)):
        return ExitInfo(ExitStatus.INPUT_ERROR, "callback must be a callable object.")
    callback_sig = None
    if callback is not None and callable(callback):
        callback_sig = inspect.signature(callback)

    all_ok, bad_keys = params.check_all_params()
    if not all_ok:
        return ExitInfo(
            ExitStatus.INPUT_ERROR,
            f"Bad parameters: {str(bad_keys)}. Please ensure that they have valid types and values.",
        )

    # reorganize dict params into list params
    reorganized_inputs = dict(
        ele_names=ele_names,
        ele_idxs=ele_idxs,
        fun=list(fun.values()),
        coords=list(coords.values()),
        weights=np.asarray(list(weights.values())),
        npt=np.asarray(list(npt.values()), dtype=int),
        xforms=list(xforms.values()),
        xform_bounds=list(xform_bounds.values()),
        maxfev=list(maxfev.values()),
        dims=dims,
        extra_fun=extra_fun,
        maxiter=int(maxiter),
        resolution_init=resolution_init,
        resolution_final=resolution_final,
        params=params,
        tr_shape=tr_shape,
        callback_sig=callback_sig,
    )
    return reorganized_inputs


def change_list_to_dict(
    lis: Union[Dict, List[Any]], ele_names: List[Any]
) -> Tuple[Optional[Dict[Any, Any]], bool]:
    if isinstance(lis, dict):
        return lis, False
    if len(lis) != len(ele_names):
        return None, True
    new_dict = dict()
    for idx, ele_name in enumerate(ele_names):
        new_dict[ele_name] = lis[idx]
    return new_dict, False


def _preprocess(
    obj: Any,
    obj_name: str,
    default_value: Any,
    ele_names: List[Any],
):
    extra_keys = check_extra_ele_keys(obj, ele_names)
    if extra_keys:
        return (
            ExitInfo(
                ExitStatus.INPUT_ERROR, f"Unexpected keys {extra_keys} in `{obj_name}`."
            ),
            obj,
        )
    obj = (
        get_default_valued_dict(ele_names, default_value, obj)
        if (not obj or isinstance(obj, dict))
        else obj
    )
    obj, is_invalid_length = (
        change_list_to_dict(obj, ele_names)
        if (isinstance(obj, list) and len(obj) > 0)
        else (obj, False)
    )
    if is_invalid_length:
        # if obj is a list and len(obj) < num_model
        return (
            ExitInfo(
                ExitStatus.INPUT_ERROR,
                f"Length of the list `{obj_name}` (which is {len(obj)}) must be "
                f"the same with that of `fun` (which is {len(ele_names)}).",
            ),
            obj,
        )
    return None, obj


def check_extra_ele_keys(obj: Any, ele_names: List[Any]) -> List[Any]:
    if isinstance(obj, dict):
        return list(set(obj.keys()).difference(ele_names))
    else:
        return []


def check_dict_validness_and_sort(
    obj: Any,
    ele_names: List[Any],
    obj_name: str,
    allow_default: bool = True,
    default_value: Any = None,
    reset_none_value: bool = False,
    type_of_val: Optional[type] = None,
) -> Tuple[Optional[Dict[Any, Any]], Optional[ExitInfo]]:
    """
    Arrange the order of keys of ``obj`` to make it consistent with ``ele_names``.
    If find unexpected keys in ``obj``, return an :class:`~upoqa.utils.ExitInfo` 
    object to terminate the algorithm.
    """
    if not isinstance(obj, dict):
        return None, ExitInfo(
            ExitStatus.INPUT_ERROR,
            f"Invalid type ({type(obj)}) of `{obj_name}` = {obj}, "
            f"which should be a dictionary or list.",
        )
    extra_key = check_extra_ele_keys(obj, ele_names)
    if len(extra_key) > 0:
        return None, ExitInfo(
            ExitStatus.INPUT_ERROR, f"Unexpected keys {extra_key} in `{obj_name}`."
        )
    new_dict = dict()
    for ele_name in ele_names:
        if ele_name in obj:
            if obj[ele_name] is None:
                new_dict[ele_name] = default_value if reset_none_value else None
            elif (type_of_val is not None) and (
                not isinstance(obj[ele_name], type_of_val)
            ):
                return None, ExitInfo(
                    ExitStatus.INPUT_ERROR,
                    f"The value of key {ele_name} in `{obj_name}` = {obj} "
                    f"is not of type {type_of_val}.",
                )
            else:
                new_dict[ele_name] = obj[ele_name]
        elif allow_default:
            new_dict[ele_name] = default_value
        else:
            return None, ExitInfo(
                ExitStatus.INPUT_ERROR,
                f"Key {ele_name} and its value cannot be found in `{obj_name}` = {obj}.",
            )
    return new_dict, None


def get_default_valued_dict(
    ele_names: List[Any],
    default_value: Any,
    old_dict: Optional[Dict[Any, Any]] = None,
) -> Dict[Any, Any]:
    new_dict = old_dict if old_dict else dict()
    for ele_name in ele_names:
        if ele_name not in new_dict:
            new_dict[ele_name] = default_value
    return new_dict
