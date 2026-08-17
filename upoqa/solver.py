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
UPOQA Solver
============

Maintain the main minimizer. 
"""

import numpy as np
import numpy.linalg as LA
import traceback
from .utils import *
from typing import List, Any, Dict, Tuple, Optional, Callable, Union, Literal, Iterable
from copy import deepcopy
import warnings

__all__ = [
    "minimize",
]

TERMINATION_MSG = "\n# Optimization terminated: {msg}"


def minimize(
    fun: Union[
        Callable[[np.ndarray], float],
        Dict[Any, Callable[[np.ndarray], float]],
        List[Callable[[np.ndarray], float]],
    ],
    x0: np.ndarray,
    coords: Union[
        Dict[Any, Union[List, np.ndarray]], List[Union[List, np.ndarray]]
    ] = dict(),
    maxiter: Optional[int] = None,
    maxfev: Optional[Union[int, Dict[Any, int], List[int]]] = dict(),
    weights: Union[
        Dict[Any, Union[float, int]], List[Union[float, int]], int, float
    ] = dict(),
    xforms: Union[
        Dict[Any, Optional[List[Callable[[np.ndarray], Union[np.ndarray, float]]]]],
        List[Optional[List[Callable[[np.ndarray], Union[np.ndarray, float]]]]],
    ] = dict(),
    xform_bounds: Union[
        Tuple[float, float], Dict[Any, Tuple[float, float]], List[Tuple[float, float]]
    ] = dict(),
    extra_fun: Optional[List[Callable[[np.ndarray], Union[np.ndarray, float]]]] = None,
    npt: Optional[Union[int, Dict[Any, int], List[int]]] = None,
    radius_init: float = 1.0,
    radius_final: float = 1e-6,
    noise_level: int = 0,
    seek_global_minimum: bool = False,
    f_target: Optional[float] = None,
    tr_shape: Literal["structured", "spherical"] = "structured",
    callback: Optional[
        Union[
            Callable[[OptimizeResult], Optional[bool]],
            Callable[[np.ndarray], Optional[bool]],
        ]
    ] = None,
    disp: bool = True,
    verbose: bool = False,
    debug: bool = False,
    return_internals: bool = False,
    options: Dict[Any, Any] = dict(),
    **kwargs,
) -> OptimizeResult:
    r"""
    Minimize an unconstrained objective function that has a partially-separable 
    structure using a derivative-free model-based trust-region method.
    
    Partial separability means the objective can be expressed as:

    .. math::

        \min_{x\in\mathbb{R}^n} \quad \sum_{i=1}^q f_i(U_i x),

    where :math:`f_i:\mathbb{R}^{|\mathcal{I}_i|} \to \mathbb{R}` are black-box 
    functions (termed as *element function* or *element*) whose gradients and 
    Hessians are unavailable, :math:`U_i:\mathbb{R}^n \to \mathbb{R}^{|\mathcal{I}_i|}` 
    are projection operators, and :math:`\mathcal{I}_i \subset [i]` is an index 
    set satisfying :math:`|\mathcal{I}_i| < n` (element functions depend on small 
    subsets of variables). 
    
    The solver also supports a more general objective form:

    .. math::

        \min_{x\in\mathbb{R}^n} \quad f_0(x) + 
        \sum_{i=1}^q w_i h_i\left(f_i(U_i x)\right),
        
    where: 
    
    - :math:`f_0: \mathbb{R}^n \to \mathbb{R}` is a white-box component with 
      computable gradients and Hessians
    - :math:`w_i \in \mathbb{R}` are element weights
    - :math:`h_i: \mathbb{R}\to\mathbb{R}` are smooth transformations on the 
      outputs of :math:`f_i`, with computable gradients/Hessians

    Parameters
    ----------
    fun : list of callables, dict of callables, or callable
        The objective element function(s). Can be:
        
        - A dictionary mapping element names to element functions
        - A list containing all element functions
        - A single function (not recommended; treated as a single element without 
          exploiting structure)
        
        When using dictionaries, element names can be any hashable type. Dictionary 
        input is recommended for better readability, especially when elements have 
        different contexts or lack inherent ordering.
        
        For list input, the solver assigns default names as ``"fun[0]"``, ``"fun[1]"``, 
        etc.
    x0 : ndarray, shape (n,)
        Initial guess for the optimization variables.
    coords : dict of array_like or list of array-like, optional
        Mapping of element names to variable indices. Each value specifies the 
        variables an element depends on.
        
        Example for dictionary input: 
            ``{'f1': [0, 1], 'f2': [2, 3]}`` indicates:
            
            - Element f1 depends on variables 0 and 1
            - Element f2 depends on variables 2 and 3
            
        Example for list input: 
            ``[[0, 1], [2, 3]]`` assigns indices to elements in list order.
        
        If not provided, all elements default to depending on all variables.
    maxiter : int, optional
        Maximum number of iterations. Default: ``1000 * n``.
    maxfev : int, dict of int or list of int, optional
        Maximum function evaluations:
        
        - dict: Per-element limits (keys match ``fun``)
        - list: Per-element limits (order matches ``fun``)
        - int: Uniform limit for all elements
    weights : dict or list, optional
        Nonnegative weights :math:`w_i` for elements. Formats:
        
        - dict: Per-element weights (keys match ``fun``)
        - list: Per-element weights (order matches ``fun``)
        - scalar: Uniform weight for all elements
    xforms : dict or list, optional
        Transformations :math:`h_i` for element outputs. Each entry must be a list 
        of length 3 containing [function, gradient, Hessian] callables.
        
        Example for two simple transformations:
        
        .. code-block:: python
        
            xforms = {
                'ele1': [lambda x: x**2, lambda x: 2*x, lambda x: 2],
                'ele2': [
                    lambda x: np.sqrt(x), 
                    lambda x: 1/(2*np.sqrt(x)), 
                    lambda x: -1/(4*x**(3/2))
                ]
            }
        
        Note: Appropriate use of transformations can accelerate convergence speed 
        but may introduce instability. For simple cases, consider incorporating 
        them directly into element functions.
    xform_bounds : dict of tuple, list of tuple, optional
        Bounds for the input domain of transformation functions. 
        
        Each entry must be a tuple of length 2 (lower_bound, upper_bound) defining 
        the valid input range for a transformation. When provided, the solver 
        automatically performs bound checking on transformation inputs, ensuring 
        they remain within specified limits.
        
        Example: ``{'log_transform': (1e-8, np.inf)}`` prevents log(0) errors.
        
        Note: Even if element function outputs naturally stay within bounds, 
        surrogate model outputs may exceed valid ranges during optimization. If your
        transformation functions cannot gracefully handle out-of-bound inputs, it is 
        strongly recommended to provide this parameter to prevent algorithm crashes 
        due to invalid transformation evaluations.
    extra_fun : list of callables, optional
        White-box component :math:`f_0` of the objective function, which is a list 
        of length 3 specified as ``[function_value, gradient, Hessian]`` callables.
    npt : int or dict or list, optional
        Number of interpolation points per element. Formats similar to ``maxfev``.
    radius_init : float, default=1.0
        Initial trust region radius
    radius_final : float, default=1e-6
        Final (minimum) trust region radius
    noise_level : {0, 1, 2}, default=0
        Estimated noise level in objective function:
        
        - ``noise_level=0``: No noise
        - ``noise_level=1``: Moderate noise. Restart mechanisms enabled and the 
          default interpolation points are set to 
          
          .. math::
                
                \max\{\min\{0.8 n_i ^{1.5} + n_i + 1, (n_i + 1)(n_i + 2) / 2\}, 
                2 n_i + 1\}
                
          where :math:`n_i` are the dimension of the :math:`i`-th element.
        - ``noise_level=2``: Significant noise. Restart mechanisms enabled and the default 
          interpolation points are set to 
          
          .. math::
                
                (n_i + 1)(n_i + 2) / 2
                
          for better noise resilience.
    seek_global_minimum : bool, default=False
        Whether to enable global optimization restarts. When True, activates 
        restart mechanisms with parameters optimized for global search (similar 
        to Py-BOBYQA [1]_).
    f_target : float, optional
         Target on the objective function value. The optimization procedure is 
         terminated when the objective function value at the iterate is less 
         than or equal to this target.
    tr_shape : {'structured', 'spherical'}, default='structured'
        Geometry of the trust region:
        
        - ``'structured'``: Complex region formed by intersecting cylinders with 
          element-specific radii. In simple cases (e.g., two orthogonal cylinders), 
          resembles a Steinmetz solid. Allows larger steps in well-modeled 
          directions while restricting poorly-modeled ones. 
        - ``'spherical'``: Traditional spherical trust region. Simpler but less 
          adaptive to element-wise model accuracy. Recommended if the 
          ``'structured'`` option incurs excessive runtime overhead (typically 
          when the number of elements is very large).
    callback : callable, optional
        A callback executed at each objective function evaluation. The method
        terminates if a ``StopIteration`` exception is raised by the callback
        function. Its signature can be one of the following:

            ``callback(intermediate_result)``

        where ``intermediate_result`` is a keyword parameter that contains an
        instance of :class:`~upoqa.utils.result.OptimizeResult`, with attributes 
        ``x``, ``fun`` and other useful information, being the point at which the 
        objective function is evaluated and the value of the objective function, 
        respectively. The name of the parameter must be ``intermediate_result`` 
        for the callback to be passed an instance of 
        :class:`~upoqa.utils.result.OptimizeResult`.

        Alternatively, the callback function can have the signature:

            ``callback(xk)``

        where ``xk`` is the point at which the objective function is evaluated.
        Introspection is used to determine which of the signatures to invoke.
    disp : bool, default=True
        Whether to display progress.
    verbose : bool, default=False
        Whether to display detailed progress information.
    debug : bool, default=False
        Enable debugging features:
        
        - ``debug=1``: NaN checking in function evaluations
        - ``debug=2``: Include solver internals (manager, interp_set, model) in result
        - ``debug=3``: Enhanced verbosity when both disp and verbose are True
    return_internals : bool, default=False
        If True, the returned result will contain the fields ``manager``,
        ``interp_set`` and ``model`` for debugging purposes even if 
        ``debug=False``.
    options : dict
        Advanced algorithm parameters. See the source code of 
        :class:`~upoqa.utils.params.UPOQAParameterList` for available options.
        
    Returns
    -------
    :class:`~upoqa.utils.result.OptimizeResult`
        Result of the optimization procedure, with the following fields:
    
            message : str
                Description of the cause of the termination.
            success : bool
                Whether the optimization procedure terminated successfully.
            fun : float
                Value of objective function at the solution.
            funs : dict or list
                Values of the element functions evaluated at the solution. The 
                type matches the input ``fun``:
                
                - If ``fun`` was a dictionary, return a dictionary mapping element
                  names to their function values.
                - If ``fun`` was a list, return a list of function values in the 
                  same order as the input list.
            extra_fun : float
                Value of extra function (the white-box component :math:`f_0`) 
                at the solution
            x : ndarray
                The solution of the optimization.
            jac, hess : ndarray
                Values of objective function's Jacobian and its Hessian at ``x``.
                The Hessian is an approximation.
            nit : int
                Number of iterations performed by the optimizer.
            nfev : dict or list
                Number of evaluations of element functions. 
                
                The type matches the input ``fun``:
                
                - If ``fun`` was a dictionary, return a dictionary mapping element 
                  names to their numbers of function evaluations.
                - If ``fun`` was a list, return a list of evaluation numbers in the
                  same order as the input list.
            max_nfev : int
                Maximum number of evaluations of element functions.
            avg_nfev : float
                Average number of evaluations of element functions.
            nrun : int
                Number of runs (when enabling restarts).
        
        If ``debug=True`` or ``return_internals=True``, the result also has the 
        following fields:
                
            manager : :class:`~upoqa.utils.manager.UPOQAManager`
                Algorithm manager.
            interp_set : :class:`~upoqa.utils.interp_set.OverallInterpSet`
                Interpolation point set.
            model : :class:`~upoqa.utils.model.OverallSurrogate`
                Surrogate model.
                
        If an exception is raised during the optimization that cannot be handled  
        by the algorithm, the result will also contain the following fields:
        
            exception : Exception
                Exception raised during the optimization that caused the 
                termination, if any.
            traceback : str
                Traceback of the exception, if any.
                
    References
    ----------
    .. [1] C. Cartis, J. Fiala, B. Marteau, and L. Roberts. Improving the 
        flexibility and robustness of model-based derivative-free optimization 
        solvers. *ACM Trans. Math. Softw.* 45, 3 (2019), 1-41. 
        `doi:10.1145/3338517 <https://doi.org/10.1145/3338517>`_.
            
    
    Examples
    --------
    Here is a simple example of using the solver to minimize a function with 
    two elements. Consider the following problem:

    .. math::
    
        \min_{x,y,z\in \mathbb{R}} \quad x^2 + 2y^2 + z^2 + 2xy - (y + 1)z

    let's replace :math:`[x,y,z]` with :math:`\mathbf{x} = [x_1,x_2,x_3]`, and 
    rewrite this problem into

    .. math::
    
        \min_{\mathbf{x}\in\mathbb{R}^3} \quad f_1(x_1, x_2) + f_2(x_2, x_3),

    where

    .. math::

        \begin{align*}
            &f_1(x_1, x_2) = x_1^2 + x_2^2 + 2x_1 x_2, \\
            &f_2(x_2, x_3) = x_2^2 + x_3^2 - (x_2 + 1)x_3,
        \end{align*}

    then the objective function can be implemented as:
    
    >>> def f1(x):    # f1(x,y)
    ...     return x[0] ** 2 + x[1] ** 2 + 2 * x[0] * x[1]   # x^2 + y^2 + 2xy
    >>> def f2(x):    # f2(y,z)
    ...    return x[0] ** 2 + x[1] ** 2 - (x[0] + 1) * x[1]  # y^2 + z^2 - (y+1)z
    >>> fun =    {'xy': f1,     'yz': f2     }
    >>> coords = {'xy': [0, 1], 'yz': [1, 2]}
    
    The problem can be solved by:
    
    >>> x0 = [0, 0, 0]
    >>> res = minimize(fun, x0, coords = coords, disp = False)
    >>> res.x
    array([-0.33333332,  0.3333333 ,  0.66666665])
    >>> res.fun
    np.float64(-0.33333333333333215)
    """
    ## catch any exception that may break the algorithm
    try:
        # Initialize basic parameters
        x0 = np.array(x0, dtype=np.float64)
        xk, x_best = x0.copy(), x0.copy()
        n = x0.size
        fval_k, f_best, extra_f_best = np.inf, np.inf, np.inf
        exit_info = None
        options.update(kwargs)

        if maxiter is None and (
            maxfev is None or (isinstance(maxfev, Iterable) and len(maxfev) == 0)
        ):
            maxiter = 1000 * n

        if disp:
            disp_level = 1
            if verbose:
                disp_level = 2
                if debug:
                    disp_level = 3
        else:
            disp_level = 0

        weights_xforms_enabled = bool(xforms or weights)
        extra_fun_enabled = bool(extra_fun)
        is_dict_input = isinstance(fun, dict)

        reorganized_inputs = get_reorganized_inputs_or_exit_info(
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
        )
        if isinstance(reorganized_inputs, ExitInfo):
            return _get_result_of_a_bad_run(
                exit_info=reorganized_inputs,
                disp=disp_level,
                return_internals=debug or return_internals,
                is_dict_input=is_dict_input,
            )

        # extract parameters from preprocessed inputs
        (
            ele_names,
            ele_idxs,
            params,
            resolution_init,
            resolution_final,
            callback_sig,
            tr_shape,
            fun,
            coords,
            weights,
            dims,
            npt,
            maxfev,
            maxiter,
            xforms,
            xform_bounds,
            extra_fun,
        ) = map(
            reorganized_inputs.get,
            [
                "ele_names",
                "ele_idxs",
                "params",
                "resolution_init",
                "resolution_final",
                "callback_sig",
                "tr_shape",
                "fun",
                "coords",
                "weights",
                "dims",
                "npt",
                "maxfev",
                "maxiter",
                "xforms",
                "xform_bounds",
                "extra_fun",
            ],
        )

        def proj_onto_ele(X: np.ndarray, ele_idx: int) -> np.ndarray:
            """project point x onto the elemental range space"""
            if X.ndim == 1:
                return X[coords[ele_idx]]
            elif X.ndim == 2:
                return X[:, coords[ele_idx]]
            return X

        # The radius size of the enveloping sphere trust region <= envelope_factor * max(radii)
        ele_num = len(ele_names)
        envelope_factor: float = min(n, ele_num) ** 0.5

        # Bool masks for the elemental variable indices
        coords_mask = _build_coords_mask(coords, n)

        # Initialize the algorithm manager
        manager = UPOQAManager(
            resolution_init,
            resolution_final,
            maxiter,
            maxfev,
            proj_onto_ele,
            coords,
            xform_bounds,
            params,
            ele_names,
            disp_level,
        )

        # apply bound check for xforms if `bounds` is not None
        xforms = manager.wrap_xforms_with_bounds(xforms) if xform_bounds else xforms

        def element_generator(ele_idx: int) -> Callable[[np.ndarray], float]:
            def ele_fun_eval(x: np.ndarray) -> float:
                """
                Wrapped element function, which internally tracks its evaluation count and
                raises :class:`~upoqa.utils.manager.MaxEvalNumReached` if the maximum number 
                of evaluations is exceeded, or ``ValueError`` if the function returns an 
                invalid value.
                """
                nonlocal manager
                if manager.nfev[ele_idx] >= manager.maxfev[ele_idx]:
                    raise MaxEvalNumReached(
                        ele_idx=ele_idx, ele_name=ele_names[ele_idx]
                    )

                manager.nfev[ele_idx] += 1  # increase the evaluation counter
                try:
                    eval_result = fun[ele_idx](x)
                except Exception as e:
                    raise ValueError(
                        f"(element {ele_names[ele_idx]}) Exception raised by "
                        "the objective function: "
                        f"{e}\n\n{traceback.format_exc()}"
                    )

                if np.any(~np.isfinite(eval_result)):
                    raise ValueError(
                        f"(element {ele_names[ele_idx]}) Non-finite (NaN or Inf) value "
                        "encountered in the return value of the objective function. "
                        f"\nreturn value = {eval_result}"
                    )

                return float(eval_result)

            return ele_fun_eval

        # Initialize the interpolation sets, the surrogate models and some other parameters.
        element_funcs: Dict[Any, Callable] = dict()
        ele_interp_sets: List[InterpSet] = [None for _ in ele_idxs]
        ele_models: List[QuadSurrogate] = [None for _ in ele_idxs]

        if disp_level >= 1:
            import tqdm

            progress_bar = tqdm.tqdm(
                total=len(ele_names), desc="Initializing surrogate models"
            )

        for ele_idx in ele_idxs:
            if disp_level >= 1:
                progress_bar.set_postfix({"element": ele_names[ele_idx]})
            element_funcs[ele_idx] = element_generator(ele_idx)
            ele_interp_set = InterpSet(n=dims[ele_idx], npt=npt[ele_idx])
            ele_interp_sets[ele_idx] = ele_interp_set.init_interp_set(
                fun=element_funcs[ele_idx],
                center=proj_onto_ele(x_best, ele_idx),
                step_size=resolution_init,
            )
            ele_models[ele_idx] = QuadSurrogate(
                interp_set=ele_interp_set,
                center=proj_onto_ele(x_best, ele_idx),
                n=dims[ele_idx],
            )
            if disp_level >= 1:
                progress_bar.update(1)

        if disp_level >= 1:
            progress_bar.close()

        def fun_eles_eval(x: np.ndarray) -> Tuple[List[float], float]:
            """Given input x, return [ele1(x), ele2(x), ..., elek(x)] and extra_fun(x)."""
            fval_eles = [
                element_funcs[ele_idx](proj_onto_ele(x, ele_idx))
                for ele_idx in ele_idxs
            ]
            try:
                extra_fval = float(extra_fun[0](x))
            except Exception as e:
                raise ValueError(
                    f"Exception raised by the extra smooth objective function: "
                    f"{e}\n\n{traceback.format_exc()}"
                )
            if params("debug.check_nan_fval") and np.any(np.isnan(extra_fval)):
                raise ValueError(
                    f"NaN encountered in the return value"
                    f" of the extra objective function. \nreturn value = {extra_fval}"
                )
            return fval_eles, extra_fval

        # Initialize the overall interpolation set
        overall_interp_set = OverallInterpSet(
            n=n,
            set_size=params("init.overall_interp_set_size"),
            max_size=int(
                max(
                    params("general.overall_interp_set_size_factor") * n,
                    params("init.overall_interp_set_size"),
                )
            ),
            ele_interp_sets=ele_interp_sets,
            proj_onto_ele=proj_onto_ele,
            ele_names=ele_names,
        )

        # Initialize the overall surrogate model
        overall_surrogate = OverallSurrogate(
            n=n,
            interp_set=overall_interp_set,
            proj_onto_ele=proj_onto_ele,
            coords=coords,
            ele_models=ele_models,
            extra_fun=extra_fun,
            ele_names=ele_names,
            params=params,
        )

        manager.couple_with_utils(overall_interp_set, overall_surrogate)
        manager.set_weights(weights)
        manager.set_xforms(xforms)

        if params("init.search_opt_x0"):
            # Search the optimal start point from combinations of elemental interpolation points.
            start_x, start_fval, fval_eles, extra_fval, exit_info = (
                manager.find_best_start_point()
            )
        else:
            # otherwise, start from x0.
            start_x = x0
            fval_eles = [
                ele_interp_sets[ele_idx].get_interp_set(0)[1] for ele_idx in ele_idxs
            ]
            extra_fval = float(extra_fun[0](x0))
            start_fval = manager.build_fval(fval_eles, extra_fval)

        # Prepare some useful auxiliary functions
        def update_short_step_count(step_max_ele_norm: float):
            """Updates the count of short steps and very short steps."""
            nonlocal n_short_steps, n_very_short_steps
            if (
                step_max_ele_norm
                <= params("general.short_step_thresh") * manager.resolution
            ):
                if manager.max_radius > manager.resolution:
                    n_short_steps, n_very_short_steps = 0, 0
                else:
                    n_short_steps += 1
                    n_very_short_steps += 1
                    if (
                        step_max_ele_norm
                        > params("general.very_short_step_thresh") * manager.resolution
                    ):
                        n_very_short_steps = 0
                return True
            return False

        ele_fs_best: List[float] = [None for _ in ele_idxs]

        def update_x_best_in_history(
            x_new: np.ndarray, f_new: float, ele_fs_new: List[float], extra_f_new: float
        ) -> Optional[ExitInfo]:
            """
            Updates the best x and its elemental, overall and extra function value in history.
            """
            nonlocal x_best, f_best, ele_fs_best, extra_f_best
            if f_new < f_best:
                x_best, f_best, ele_fs_best, extra_f_best = (
                    x_new.copy(),
                    float(f_new),
                    [float(x) for x in deepcopy(ele_fs_new)],
                    float(extra_f_new),
                )
            # if the current objective function value is already <= f_target, terminate.
            if f_target is not None and f_best <= f_target:
                return ExitInfo(
                    flag=ExitStatus.SUCCESS,
                    msg=(
                        f"The current objective function value {f_best:.4} "
                        "is already <= ftaget."
                    ),
                )
            return None

        def refresh_f_best() -> None:
            """
            refresh the best elemental, overall and extra function value in history
            in case the weights or xforms are updated by the user.
            """
            nonlocal f_best, ele_fs_best, extra_f_best
            extra_f_best = float(manager.extra_fun[0](x_best))
            f_best = manager.build_fval(ele_fs_best, extra_f_best)

        def check_and_try_to_restart(
            exit_info: Optional[ExitInfo] = None,
        ) -> Tuple[Optional[ExitInfo], bool]:
            """
            Trying to launch a restart, and return the updated ``exit_info`` and a flag
            indicating whether a restart has been initiated successfully.
            """
            nonlocal nit_in_a_run, n_short_steps, n_very_short_steps, n_alt_models, manager, xk, fval_k
            if exit_info:
                if manager.params("restarts.use_restarts"):
                    if disp_level >= 3:
                        print(
                            f"  Trying to launch a restart instead of terminating "
                            f"by: {exit_info.message()}"
                        )

                    exit_info = manager.prepare_for_a_restart()
                    if exit_info:
                        return exit_info, False

                    exit_info = manager.soft_restart(fun_eles_eval)
                    if exit_info:
                        return exit_info, False

                    nit_in_a_run, n_short_steps, n_very_short_steps = 0, 0, 0
                    n_alt_models = [0 for _ in ele_names]
                    manager.resolution_final *= manager.params(
                        "restarts.resolution_final_scale"
                    )
                    manager.reset_restart_track_info()
                    xk, fval_k = overall_interp_set.get_opt()
                    return None, True
                else:
                    return exit_info, False  # quit
            return None, False

        def wrapped_callback() -> bool:
            """The wrapped callback function."""
            if callback:
                try:
                    if set(callback_sig.parameters) == {"intermediate_result"}:
                        # callback function with signature `callback(intermediate_result)`
                        state_dict = dict(
                            x=np.copy(xk),
                            fun=fval_k,
                            jac=np.copy(grad_k),
                            funs=deepcopy(ele_fvals_k),
                            extra_fun=extra_fval_k,
                            nit=nit + 1,
                            nrun=manager.nrun + 1,
                            nfev=deepcopy(manager.nfev),
                            avg_nfev=np.mean(manager.nfev),
                            max_nfev=max(manager.nfev),
                        )
                        if debug or return_internals:
                            state_dict.update(
                                dict(
                                    manager=manager,
                                    interp_set=overall_interp_set,
                                    model=overall_surrogate,
                                )
                            )
                        if not is_dict_input:
                            state_dict.update(
                                dict(
                                    funs=list(state_dict["funs"].values()),
                                    nfev=list(state_dict["nfev"].values()),
                                )
                            )
                        state = OptimizeResult(**state_dict)
                        return callback(state)
                    else:
                        # callback function with signature `callback(xk)`
                        return callback(np.copy(xk))
                except StopIteration:
                    return True
            return False

        if exit_info:
            return _get_result_of_a_bad_run(
                exit_info=exit_info,
                manager=manager,
                disp=disp_level,
                return_internals=return_internals or debug,
                is_dict_input=is_dict_input,
            )

        # Add the start point into the overall interpolation set and get ready to
        # start the algorithm
        overall_interp_set.append(start_x, start_fval, fval_eles, extra_fval)
        manager.shift_center_to(start_x)
        update_x_best_in_history(*(overall_interp_set.get_opt(verbose=True)))

        xk, fval_k = overall_interp_set.get_opt()
        grad_k = overall_surrogate.grad_eval(xk)
        old_grad, old_hess = grad_k, overall_surrogate.hess_eval(xk)

        # Initialize the iteration counter and other tracking variables
        nit_in_a_run, n_short_steps, n_very_short_steps = 0, 0, 0
        n_alt_models = [0 for _ in ele_names]

        # Print header
        if disp_level >= 2:
            print(
                f"# UPOQA ready to start (nrun: {manager.nrun})\n"
                f"    nfev:                  \n      {dict(zip(ele_names, manager.nfev))}\n"
                f"    radii:                 \n      {dict(zip(ele_names, manager.radii))}\n"
                f"    resolution:            {manager.resolution:.12}\n"
                f"    obj:                   {fval_k:.12}"
            )
        elif disp_level == 1:
            print(
                "{:^5}{:^7}{:^10}{:^10}{:^11}{:^12}{:^7}".format(
                    "Run", "Iter", "Obj", "Grad", "Delta", "Resolution", "Evals"
                )
            )

        # Start the main loop
        nit = -1  # necessary if maxiter = 0.
        for nit in range(maxiter):
            n_want_GI = 0
            last_radii = deepcopy(manager.radii)
            want_reduce_resolution, reach_minimum_resolution_flag = False, False
            want_improve_geometry = [False for _ in ele_names]
            nit_in_a_run += 1

            xk, fval_k, ele_fvals_k, extra_fval_k = overall_interp_set.get_opt(
                verbose=True
            )
            grad_k = overall_surrogate.grad_eval(xk)

            if disp_level >= 2:
                print(f"\n# Iteration {nit_in_a_run} (nrun: {manager.nrun + 1})")

            # callback. The user is free to vary the `weights` and `xforms` inside the callback
            # function `callback(state)` by calling `state.manager.update_weights(new_weights)`
            # and `state.manager.update_xforms(new_xforms)`.
            if wrapped_callback():
                exit_info = ExitInfo(
                    ExitStatus.SUCCESS, "Terminated by user-defined callback. "
                )
                break

            # Update f_best in case the user has changed the weights or xforms.
            refresh_f_best()
            overall_interp_set.update_fval_with_new_weights_and_xforms(
                weights=manager.weights,
                xforms=manager.xforms,
                extra_fun_eval=manager.extra_fun[0],
            )

            # Detect whether to call a restart
            restart_detected_flag = manager.decide_to_restart_due_to_exploding_coeff()
            if restart_detected_flag:
                exit_info, continue_flag = check_and_try_to_restart(
                    ExitInfo(
                        ExitStatus.AUTO_DETECT_RESTART_WARNING,
                        "Auto-detected restart. ",
                    )
                )
                if exit_info:
                    break
                elif continue_flag:
                    continue

            # Update xk, x_best
            xk, fval_k, ele_fvals_k, extra_fval_k = overall_interp_set.get_opt(
                verbose=True
            )
            grad_k = overall_surrogate.grad_eval(xk)
            exit_info = update_x_best_in_history(xk, fval_k, ele_fvals_k, extra_fval_k)
            if exit_info:
                break

            # Shift model center to xk
            manager.shift_center_to(xk)

            # Calculate the trust region step
            try:
                surrogate_fvalk = overall_surrogate.fun_eval(xk)
                if tr_shape == "structured":
                    # Solve the trust region subproblem using gradient projection method
                    envelope_radius = manager.max_radius * envelope_factor

                    sk = conjugate_gradient_proj_steinmetz(
                        fun=lambda s: overall_surrogate.fun_eval(xk + s)
                        - surrogate_fvalk,
                        grad=lambda s: overall_surrogate.grad_eval(xk + s),
                        hess_prod=lambda s, v: overall_surrogate.hess_operator(
                            xk + s, v
                        ),
                        coords_mask=coords_mask,
                        n=n,
                        deltas=np.array(manager.radii),
                        envelope_delta=envelope_radius,
                    )
                elif tr_shape == "spherical":
                    # Solve the trust region subproblem using tangential_byrd_omojokun
                    sk = tangential_byrd_omojokun(
                        fun=lambda s: overall_surrogate.fun_eval(xk + s)
                        - surrogate_fvalk,
                        grad=lambda s: overall_surrogate.grad_eval(xk + s),
                        hess_prod=lambda s, v: overall_surrogate.hess_operator(
                            xk + s, v
                        ),
                        delta=min(manager.radii),
                        n=n,
                    )
                else:
                    raise RuntimeError(f"Unknown tr_shape: {tr_shape}.")

                x_hold = xk + sk
                decrease = surrogate_fvalk - overall_surrogate.fun_eval(x_hold)
                s_norm = LA.norm(sk)

                # tr_vio is short for "trust region violation", which is the ratio of the norm
                # of the trust region step to the radius.
                tr_vios = get_violation_ratio(sk, coords, np.array(manager.radii))
                tr_max_vio: float = tr_vios.max()
                step_max_ele_norm: float = (tr_vios * np.asarray(manager.radii)).max()
                if tr_shape == "structured" and not extra_fun_enabled:
                    checked_step_norm = step_max_ele_norm
                else:
                    checked_step_norm = s_norm

            except (LA.LinAlgError, ZeroDivisionError) as e:
                # If there is a numerical error when calculating the trust region step,
                # try to restart the algorithm.
                exit_info, continue_flag = check_and_try_to_restart(
                    ExitInfo(
                        flag=ExitStatus.LINALG_ERROR,
                        msg=(
                            "Numerical error encountered when calculating trust region"
                            f" step: {e} "
                        ),
                        exception=e,
                        traceback=traceback.format_exc(),
                    )
                )
                if exit_info:
                    break
                elif continue_flag:
                    continue

            # Decide whether or not to accept the new point `x_hold` depending on the norm
            # of trust region step.
            if update_short_step_count(checked_step_norm):
                # If the norm of trust region step is small
                if disp_level == 3:
                    print(
                        f"  Detect short step (step norm = {s_norm:.6}, "
                        f"max elemental step norm = {step_max_ele_norm:.6}, "
                        f"n_short_steps = {n_short_steps},"
                        f" n_very_short_steps = {n_very_short_steps})"
                    )
                elif disp_level == 2:
                    print(
                        f"  Detect short step (step norm = {s_norm:.6}, "
                        f"max elemental step norm = {step_max_ele_norm:.6})"
                    )

                for ele_idx in ele_idxs:
                    manager.set_radius(
                        params("tr_radius.alpha2") * manager.radii[ele_idx], ele_idx
                    )

                if n_short_steps >= params(
                    "general.max_short_steps"
                ) or n_very_short_steps >= params("general.max_very_short_steps"):
                    # If the number of short steps exceeds the threshold, reduce the trust region resolution.
                    n_short_steps, n_very_short_steps = 0, 0
                    want_reduce_resolution = True
                else:
                    try:
                        idx_to_replaced_eles, dist_new_eles, _ = (
                            manager.get_index_to_remove()
                        )
                    except (LA.LinAlgError, ZeroDivisionError) as e:
                        exit_info, continue_flag = check_and_try_to_restart(
                            ExitInfo(
                                flag=ExitStatus.LINALG_ERROR,
                                msg="Singular matrix when choosing point to replace for "
                                + "geometry-improving procedure (short step). ",
                                exception=e,
                                traceback=traceback.format_exc(),
                            )
                        )
                        if exit_info:
                            break
                        elif continue_flag:
                            continue

                    want_improve_geometry = [
                        dist_new_eles[ele_idx]
                        > max(manager.radii[ele_idx], 2.0 * manager.resolution)
                        for ele_idx in ele_idxs
                    ]
                    n_want_GI = np.array(want_improve_geometry).sum()
                    want_reduce_resolution = True if (n_want_GI == 0) else False
            else:
                # If the norm of trust region step is big enough
                n_short_steps, n_very_short_steps = 0, 0
                try:
                    fval_eles, extra_fval = fun_eles_eval(x_hold)
                    fval = manager.build_fval(fval_eles, extra_fval)
                except MaxEvalNumReached as e:
                    exit_info = ExitInfo(
                        flag=ExitStatus.SUCCESS,
                        msg=f"(element {e.ele_name}) Objective has been called MAXFUN times. ",
                    )
                    break

                decrease_extra_fval = extra_fval_k - extra_fval

                # Calculate decreases of element models, and decreases of the value of
                # `weight * xform(element model or element obj-func value)`
                decrease_eles, decrease_wx_m_eles, decrease_wx_f_eles, fval_wx_eles = (
                    manager.calc_detailed_decrease(xk, x_hold, ele_fvals_k, fval_eles)
                )

                # Calculate the reduction ratio
                ratio, ratios, exit_info = manager.get_reduction_ratio(
                    fval, fval_k, decrease, fval_eles, ele_fvals_k, decrease_eles
                )

                exit_info, continue_flag = check_and_try_to_restart(exit_info)
                if exit_info:
                    break
                elif continue_flag:
                    continue

                # Decide which interpolation point to delete
                try:
                    idx_to_replaced_eles, _, cached_kkt_info_eles = (
                        manager.get_index_to_remove(x=x_hold, center=xk)
                    )
                except (LA.LinAlgError, ZeroDivisionError) as e:
                    exit_info, continue_flag = check_and_try_to_restart(
                        ExitInfo(
                            flag=ExitStatus.LINALG_ERROR,
                            msg="Singular matrix when choosing point to replace. ",
                            exception=e,
                            traceback=traceback.format_exc(),
                        )
                    )
                    if exit_info:
                        break
                    elif continue_flag:
                        continue

                # Check slow iteration
                if (
                    manager.params("slow.terminate_when_slow")
                    and ratio > 0
                    and manager.check_slow_iteration()[1]
                ):
                    exit_info, continue_flag = check_and_try_to_restart(
                        ExitInfo(
                            flag=ExitStatus.SLOW_WARNING,
                            msg="Maximum slow iterations reached. ",
                        )
                    )
                    if exit_info:
                        break
                    elif continue_flag:
                        continue

                # For each elemental surrogate model, check whether or not to accept the
                # new point `x_hold` as a new interpolation point.
                need_update, exit_info = manager.get_update_requests(
                    idx_to_replaced_eles,
                    tr_vios,
                    cached_kkt_info_eles,
                )
                if exit_info:
                    break

                if disp_level >= 3:
                    for ele_idx in ele_idxs:
                        cached_kkt_info = cached_kkt_info_eles[ele_idx]
                        if need_update[ele_idx]:
                            print(
                                f"  (element {ele_names[ele_idx]}) Determinant ratio ="
                                f" {cached_kkt_info[3][idx_to_replaced_eles[ele_idx]]:.4e}, "
                                f"beta = {cached_kkt_info[1]:.4e}"
                            )
                        else:
                            print(
                                f"  (element {ele_names[ele_idx]}) [filtered out]"
                                " Determinant ratio ="
                                f" {cached_kkt_info[3][idx_to_replaced_eles[ele_idx]]:.4e}, "
                                f"beta = {cached_kkt_info[1]:.4e}"
                            )

                # Update the interpolation set and the surrogate model
                overall_interp_set.update_point_on_idx(
                    x_hold,
                    idx_to_replaced_eles,
                    fval_eles,
                    fval,
                    extra_fval,
                    need_update,
                )
                try:
                    overall_surrogate.update(need_update, cached_kkt_info_eles)
                except SurrogateLinAlgError as e:
                    exit_info, continue_flag = check_and_try_to_restart(
                        ExitInfo(
                            flag=ExitStatus.LINALG_ERROR,
                            msg=(
                                f"(element {e.ele_name}) Singular matrix when "
                                "updating surrogate model. "
                            ),
                            exception=e,
                            traceback=traceback.format_exc(),
                        )
                    )
                    if exit_info:
                        break
                    elif continue_flag:
                        continue

                # Update useful tracking information for the auto detection of restart
                manager.update_restart_track_info(xk, old_grad, old_hess)
                old_grad, old_hess = overall_surrogate.grad_eval(
                    xk
                ), overall_surrogate.hess_eval(xk)
                grad_k = overall_surrogate.grad_eval(overall_interp_set.get_opt()[0])

                # Print information of current iteration
                if disp_level >= 2:
                    iter_log = ""
                    if disp_level == 3:
                        iter_log += f"    point change indices: \n      {dict(zip(ele_names, idx_to_replaced_eles))}\n"
                    iter_log += (
                        f"    nfev:                 \n      {dict(zip(ele_names, manager.nfev))}\n"
                        f"    reduc-ratio:          {ratio:.12}\n"
                        f"    step size:            {s_norm:.12}\n"
                        f"    radii:               "
                    )
                    if tr_shape == "structured":
                        iter_log += f"\n      {dict(zip(ele_names, [float(x) for x in manager.radii]))}\n"
                    else:
                        iter_log += f"{manager.radii[0]:.2}\n"
                    iter_log += (
                        f"    resolution:           {manager.resolution:.12}\n"
                        f"    obj change:           {fval_k:.12}  -->  {fval:.12}\n"
                    )
                    if extra_fun_enabled:
                        iter_log += f"    extra-fun change:     {extra_fval_k:.12}  -->  {extra_fval:.12}\n"
                    iter_log += f"    value of element:     \n      {dict(zip(ele_names, [float(x) for x in fval_eles]))}\n"
                    if weights_xforms_enabled:
                        iter_log += (
                            f"    value of weighted and transformed element on the new point:\n"
                            f"      {dict(zip(ele_names, [float(x) for x in fval_wx_eles]))}\n"
                        )
                    if disp_level == 3:
                        iter_log += (
                            f"    norm(gk):             {LA.norm(grad_k):.12}\n"
                            f"    tr_max_vio(sk):       {tr_max_vio:.12}\n"
                            f"    ele-wise reduc-ratio: \n      {dict(zip(ele_names, ratios))}\n"
                        )
                        if weights_xforms_enabled:
                            iter_log += (
                                f"    model decrease (with weights and transformations):\n"
                                f"      {dict(zip(ele_names, [float(x) for x in decrease_wx_f_eles]))}\n"
                            )
                        iter_log += (
                            f"    tr_vios:              \n      {dict(zip(ele_names, [float(x) for x in tr_vios]))}\n"
                            f"    n_alt_models:         \n      {dict(zip(ele_names, n_alt_models))}\n"
                            f"    n_short_steps:        {n_short_steps}\n"
                            f"    n_very_short_steps:   {n_very_short_steps}\n"
                        )
                    print(iter_log, end="")

                xk, fval_k = overall_interp_set.get_opt()

                # Update the trust region radii
                manager.update_radius(
                    tr_vios,
                    ratio,
                    ratios,
                    decrease,
                    decrease_extra_fval,
                    decrease_wx_m_eles,
                    decrease_wx_f_eles,
                    s_norm / min(manager.radii),
                    (tr_shape == "spherical"),
                )

                if disp_level == 1:
                    print(
                        "{:^5}{:^7}{:^10.2e}{:^10.2e}{:^11.2e}{:^12.2e}{:^7}".format(
                            manager.nrun + 1,
                            nit_in_a_run,
                            fval_k,
                            LA.norm(grad_k),
                            manager.average_radius,
                            manager.resolution,
                            max(manager.nfev),
                        )
                    )

                # Consider reinitializing the surrogate model if the difference of model grad at xk
                # between current model and the alternative model grows too large.
                for ele_idx in ele_idxs:
                    if manager.radii[ele_idx] <= manager.resolution and ratios[
                        ele_idx
                    ] < params("general.alt_model_check_thresh"):
                        surrogate = ele_models[ele_idx]
                        n_alt_models[ele_idx] += 1
                        xk_ele = proj_onto_ele(xk, ele_idx)
                        grad_norm, grad_alt_norm = (
                            LA.norm(surrogate.grad_eval(xk_ele)),
                            LA.norm(surrogate.alt_grad_eval(xk_ele)),
                        )
                        if disp_level == 3:
                            print(
                                f"  (element {ele_names[ele_idx]}) Model Grad Norm ="
                                f" {grad_norm}, Alt Model Grad Norm = {grad_alt_norm}."
                            )
                        if (
                            grad_norm
                            < params("general.alt_model_thresh") * grad_alt_norm
                        ):
                            n_alt_models[ele_idx] = 0
                        if n_alt_models[ele_idx] >= params(
                            "general.max_alt_model_steps"
                        ):
                            surrogate.reinit()
                            n_alt_models[ele_idx] = 0
                            if disp_level >= 2:
                                print(
                                    "  Reinitialize the surrogate model "
                                    f"(element {ele_names[ele_idx]})."
                                )
                    else:
                        n_alt_models[ele_idx] = 0

                try:
                    idx_to_replaced_eles, dist_new_eles, _ = (
                        manager.get_index_to_remove()
                    )
                except (LA.LinAlgError, ZeroDivisionError) as e:
                    exit_info, continue_flag = check_and_try_to_restart(
                        ExitInfo(
                            flag=ExitStatus.LINALG_ERROR,
                            msg=(
                                "Singular matrix when choosing point to replace "
                                "for geometry-improving procedure. "
                            ),
                            exception=e,
                            traceback=traceback.format_exc(),
                        )
                    )
                    if exit_info:
                        break
                    elif continue_flag:
                        continue

                # Check whether or not to launch a geometry-improving step
                if ratio <= params("tr_radius.eta1"):
                    want_improve_geometry = [
                        bool(
                            dist_new_eles[ele_idx]
                            > max(manager.radii[ele_idx], 2.0 * manager.resolution)
                        )
                        for ele_idx in ele_idxs
                    ]
                    n_want_GI = np.array(want_improve_geometry).sum()
                    want_reduce_resolution = (n_want_GI == 0) and (
                        max(last_radii) <= manager.resolution
                    )

            # Reduce the trust region resolution
            if want_reduce_resolution:
                if disp_level >= 2:
                    print(f"  Reduce resolution from {manager.resolution:.3}", end="")
                reach_minimum_resolution_flag = manager.reduce_resolution()
                if disp_level >= 2:
                    print(f" to {manager.resolution:.3}.")

            # Has reduced the trust region resolution to the minimum?
            if reach_minimum_resolution_flag:
                exit_info = ExitInfo(
                    flag=ExitStatus.SUCCESS,
                    msg="The resolution has reached its minimum. ",
                )
            exit_info, continue_flag = check_and_try_to_restart(exit_info)
            if exit_info:
                break
            elif continue_flag:
                continue

            # Improve the geometry of the interpolation set if necessary.
            if n_want_GI > 0:
                xk, fval_k, ele_fvals_k, extra_fval_k = overall_interp_set.get_opt(
                    verbose=True
                )
                exit_info = update_x_best_in_history(
                    xk, fval_k, ele_fvals_k, extra_fval_k
                )
                if exit_info:
                    break
                try:
                    step_eles, cached_kkt_info_eles = manager.get_geometry_step(
                        idx_to_replaced_eles, want_improve_geometry
                    )
                except (LA.LinAlgError, ZeroDivisionError) as e:
                    exit_info, continue_flag = check_and_try_to_restart(
                        ExitInfo(
                            flag=ExitStatus.LINALG_ERROR,
                            msg="Singular matrix encountered in geometry-improving step. ",
                            exception=e,
                            traceback=traceback.format_exc(),
                        )
                    )
                    if exit_info:
                        break
                    elif continue_flag:
                        continue
                except SurrogateLinAlgError as e:
                    exit_info = ExitInfo(
                        flag=ExitStatus.LINALG_ERROR,
                        msg=f"(element {e.ele_name}) " + str(e),
                        exception=e,
                        traceback=traceback.format_exc(),
                    )
                    break

                break_flag = False
                for ele_idx in ele_idxs:
                    interp_set = ele_interp_sets[ele_idx]
                    if want_improve_geometry[ele_idx]:
                        x_ele_GI = proj_onto_ele(xk, ele_idx) + step_eles[ele_idx]
                        try:
                            ele_fval = element_funcs[ele_idx](x_ele_GI)
                            interp_set.update_point_on_idx(
                                x_ele_GI,
                                idx_to_replaced_eles[ele_idx],
                                ele_fval,
                            )
                        except MaxEvalNumReached:
                            exit_info = ExitInfo(
                                flag=ExitStatus.SUCCESS,
                                msg=(
                                    f"(element {ele_names[ele_idx]}) Objective"
                                    " has been called MAXFUN times. "
                                ),
                            )
                            break_flag = True
                            break

                if break_flag:
                    break

                try:
                    overall_surrogate.update(
                        want_improve_geometry, cached_kkt_info_eles
                    )
                except SurrogateLinAlgError as e:
                    exit_info, continue_flag = check_and_try_to_restart(
                        ExitInfo(
                            flag=ExitStatus.LINALG_ERROR,
                            msg=f"Singular matrix when updating surrogate model"
                            + f"(element {e.ele_name}) in geometry-improving step. ",
                            exception=e,
                            traceback=traceback.format_exc(),
                        )
                    )
                    if exit_info:
                        break
                    elif continue_flag:
                        continue

                if disp_level >= 2:
                    print(f"  Improving Geometry ...")
                    if disp_level == 3:
                        print(
                            f"    point change indices: \n      {dict(zip(ele_names, idx_to_replaced_eles))}"
                        )
                    print(
                        f"    element improved:     \n      {dict(zip(ele_names, want_improve_geometry))}"
                    )

                # Delete the worst points, so that the number of maintained points does not
                # exceed `overall_interp_set.max_size``.
                overall_interp_set.clear_invalid_point()

        exit_info = exit_info or ExitInfo(
            flag=ExitStatus.SUCCESS, msg="Maximum iterations reached."
        )
        update_x_best_in_history(*(overall_interp_set.get_opt(verbose=True)))

        if disp_level >= 1:
            print(TERMINATION_MSG.format(msg=exit_info.message()))

        # Prepare the final result
        result = dict(
            x=x_best,
            fun=f_best,
            funs=dict(zip(ele_names, ele_fs_best)),
            extra_fun=extra_f_best,
            jac=overall_surrogate.grad_eval(x_best),
            hess=overall_surrogate.hess_eval(x_best),
            success=True if exit_info.flag >= 0 else False,
            message=exit_info.message(with_stem=True),
            nit=nit + 1,
            nfev=dict(zip(ele_names, manager.nfev)),
            avg_nfev=np.mean(manager.nfev),
            max_nfev=max(manager.nfev),
            nrun=manager.nrun + 1,
        )
        if not is_dict_input:
            result.update(
                dict(
                    funs=list(result["funs"].values()),
                    nfev=list(result["nfev"].values()),
                )
            )
        if debug or return_internals:
            result.update(
                dict(
                    manager=manager,
                    interp_set=overall_interp_set,
                    model=overall_surrogate,
                )
            )
        if exit_info.exception:
            # If there is an exception, add it to the result.
            result.update(
                dict(exception=exit_info.exception, traceback=exit_info.traceback)
            )
        return OptimizeResult(**result)

    except Exception as e:
        tr_e = traceback.format_exc()
        flag = (
            ExitStatus.INVALID_EVAL_ERROR
            if isinstance(e, ValueError)
            else ExitStatus.UNKNOWN_ERROR
        )
        exit_info = ExitInfo(flag, str(e), e, tr_e)
        if "disp_level" in vars() and disp_level >= 1:
            print(TERMINATION_MSG.format(msg=exit_info.message()))

        try:
            res_jac = (
                overall_surrogate.grad_eval(x_best)
                if ("x_best" in vars() and "overall_surrogate" in vars())
                else None
            )
        except:
            res_jac = None

        try:
            res_hess = (
                overall_surrogate.hess_eval(x_best)
                if ("x_best" in vars() and "overall_surrogate" in vars())
                else None
            )
        except:
            res_hess = None

        try:
            result = dict(
                x=x_best if "x_best" in vars() else None,
                fun=f_best if "f_best" in vars() else None,
                funs=(
                    dict(zip(ele_names, ele_fs_best))
                    if ("ele_names" in vars() and "ele_fs_best" in vars())
                    else None
                ),
                extra_fun=(
                    extra_f_best
                    if ("x_best" in vars() and "extra_f_best" in vars())
                    else None
                ),
                jac=res_jac,
                hess=res_hess,
                success=False,
                message=exit_info.message(),
                exception=exit_info.exception,
                traceback=exit_info.traceback,
                nit=nit + 1 if "nit" in vars() else None,
                nrun=manager.nrun + 1 if "manager" in vars() else None,
                nfev=(
                    dict(zip(ele_names, manager.nfev))
                    if ("ele_names" in vars() and "manager" in vars())
                    else None
                ),
                avg_nfev=np.mean(manager.nfev) if "manager" in vars() else None,
                max_nfev=max(manager.nfev) if "manager" in vars() else None,
            )
            if not is_dict_input:
                result.update(
                    dict(
                        funs=(
                            list(result["funs"].values())
                            if isinstance(result["funs"], dict)
                            else None
                        ),
                        nfev=(
                            list(result["nfev"].values())
                            if isinstance(result["nfev"], dict)
                            else None
                        ),
                    )
                )
            if debug or return_internals:
                result.update(
                    dict(
                        manager=manager if "manager" in vars() else None,
                        interp_set=(
                            overall_interp_set
                            if "overall_interp_set" in vars()
                            else None
                        ),
                        model=(
                            overall_surrogate if "overall_surrogate" in vars() else None
                        ),
                    )
                )
            return OptimizeResult(**result)
        except Exception as e2:
            warnings.warn(
                f"During handling the unknown exception: {e}\n\n{tr_e}\n\n"
                f"Another exception is raised: {e2}\n\n{traceback.format_exc()}\n\n"
                "Return empty optimization result."
            )
            return OptimizeResult()


def _get_result_of_a_bad_run(
    exit_info: ExitInfo,
    manager: Optional[UPOQAManager] = None,
    disp: int = 1,
    return_internals: bool = False,
    is_dict_input: bool = True,
) -> OptimizeResult:
    """
    Return an empty optimization result when the optimization process is terminated abnormally.
    """
    if disp >= 1:
        print(TERMINATION_MSG.format(msg=exit_info.message()))
    result = dict(
        x=None,
        fun=None,
        funs=None,
        extra_fun=None,
        nit=0,
        nrun=manager.nrun + 1 if manager and hasattr(manager, "nrun") else 0,
        nfev=(
            dict(zip(manager.ele_names, manager.nfev))
            if manager and hasattr(manager, "nfev")
            else None
        ),
        avg_nfev=np.mean(manager.nfev) if manager and hasattr(manager, "nfev") else 0,
        max_nfev=max(manager.nfev) if manager and hasattr(manager, "nfev") else 0,
        jac=None,
        hess=None,
        success=False,
        message=exit_info.message(),
    )
    if not is_dict_input:
        result.update(
            dict(
                nfev=(
                    list(result["nfev"].values())
                    if isinstance(result["nfev"], dict)
                    else None
                )
            )
        )
    if return_internals and manager:
        result.update(
            dict(manager=manager, interp_set=manager.interp_set, model=manager.model)
        )
    if exit_info.exception:
        result.update(
            dict(
                exception=exit_info.exception,
                traceback=exit_info.traceback,
            )
        )
    return OptimizeResult(**result)


def _build_coords_mask(coords: List[np.ndarray], n: int):
    q = len(coords)
    result = np.zeros((q, n), dtype=bool)
    for i in range(q):
        result[i][coords[i]] = np.True_
    return result
