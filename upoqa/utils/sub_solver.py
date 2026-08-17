# This file contains modified code from COBYQA (https://github.com/cobyqa/cobyqa)
# which is licensed under the BSD 3-Clause License:
#
# Copyright (c) 2021-2025, Tom M. Ragonneau and Zaikun Zhang
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The modifications to this file and the overall project are licensed under the
# GNU General Public License v3.0 (GPLv3). See the project's LICENSE file for
# the full GPLv3 terms.

"""
Trust Region Subproblem Solver
==============================

Implement several sub-routines for approximately solving trust-region sub-problems 
that arise in derivative-free optimization with partially separable structures:

* ``cauchy_geometry`` / ``spider_geometry`` : geometry-improving steps along
  Cauchy / straight-line directions.
* ``tangential_byrd_omojokun`` : truncated conjugate-gradient step on the
  tangential space.
* ``conjugate_gradient_proj_steinmetz`` : modified projected CG for structured
  trust-regions using steinmetz projections.

The ``cauchy_geometry``, ``spider_geometry``, and ``tangential_byrd_omojokun`` solvers 
are drawn from the `COBYQA <https://github.com/cobyqa/cobyqa/>`_ library; see the 
original implementation at 
`here <https://github.com/cobyqa/cobyqa/blob/main/cobyqa/subsolvers/>`_.
"""

import numpy as np
import numpy.linalg as LA
from typing import Optional, Callable
from .projection import *

__all__ = [
    "cauchy_geometry",
    "spider_geometry",
    "tangential_byrd_omojokun",
    "conjugate_gradient_proj_steinmetz",
]

float_tiny = np.finfo(float).tiny
float_eps = np.finfo(float).eps


def cauchy_geometry(const, grad, curv, delta):
    r"""
    Maximize approximately the absolute value of a quadratic function subject
    to bound constraints in a trust region.

    This function solves approximately

    .. math::

        \max_{s \in \mathbb{R}^n} \quad \bigg\lvert c + g^{\mathsf{T}} s +
        \frac{1}{2} s^{\mathsf{T}} H s \bigg\rvert \quad \text{s.t.} \quad
        \left\{ \begin{array}{l}
            l \le s \le u,\\
            \lVert s \rVert \le \Delta,
        \end{array} \right.

    by maximizing the objective function along the constrained Cauchy
    direction.

    Parameters
    ----------
    const : float
        Constant :math:`c` as shown above.
    grad : ndarray, shape (n,)
        Gradient :math:`g` as shown above.
    curv : callable
        Curvature of :math:`H` along any vector.

            ``curv(s) -> float``

        returns :math:`s^{\mathsf{T}} H s`.
    delta : float
        Trust-region radius :math:`\Delta` as shown above.

    Returns
    -------
    ndarray, shape (n,)
        Approximate solution :math:`s`.

    Notes
    -----
    This function is described as the first alternative in Section 6.5 of [1]_.
    It is assumed that the origin is feasible with respect to the bound
    constraints and that ``delta`` is finite and positive.

    References
    ----------
    .. [1] Tom M. Ragonneau. *Model-Based Derivative-Free Optimization Methods
       and Software*. PhD thesis, Department of Applied Mathematics, The Hong
       Kong Polytechnic University, Hong Kong, China, 2022. URL:
       https://theses.lib.polyu.edu.hk/handle/200/12294.
    """
    # To maximize the absolute value of a quadratic function, we maximize the
    # function itself or its negative, and we choose the solution that provides
    # the largest function value.
    step1, q_val1 = _cauchy_geom(const, grad, curv, delta)
    step2, q_val2 = _cauchy_geom(-const, -grad, lambda x: -curv(x), delta)
    return step1 if abs(q_val1) >= abs(q_val2) else step2


def spider_geometry(const, grad, curv, xpt, delta):
    r"""
    Maximize approximately the absolute value of a quadratic function subject
    to bound constraints in a trust region.

    This function solves approximately

    .. math::

        \max_{s \in \mathbb{R}^n} \quad \bigg\lvert c + g^{\mathsf{T}} s +
        \frac{1}{2} s^{\mathsf{T}} H s \bigg\rvert \quad \text{s.t.} \quad
        \left\{ \begin{array}{l}
            l \le s \le u,\\
            \lVert s \rVert \le \Delta,
        \end{array} \right.

    by maximizing the objective function along given straight lines.

    Parameters
    ----------
    const : float
        Constant :math:`c` as shown above.
    grad : ndarray, shape (n,)
        Gradient :math:`g` as shown above.
    curv : callable
        Curvature of :math:`H` along any vector.

            ``curv(s) -> float``

        returns :math:`s^{\mathsf{T}} H s`.
    xpt : ndarray, shape (npt, n)
        Points defining the straight lines. The straight lines considered are
        the ones passing through the origin and the points in ``xpt``.
    delta : float
        Trust-region radius :math:`\Delta` as shown above.

    Returns
    -------
    ndarray, shape (n,)
        Approximate solution :math:`s`.

    Notes
    -----
    This function is described as the second alternative in Section 6.5 of
    [1]_. It is assumed that the origin is feasible with respect to the bound
    constraints and that ``delta`` is finite and positive.

    References
    ----------
    .. [1] Tom M. Ragonneau. *Model-Based Derivative-Free Optimization Methods
       and Software*. PhD thesis, Department of Applied Mathematics, The Hong
       Kong Polytechnic University, Hong Kong, China, 2022. URL:
       https://theses.lib.polyu.edu.hk/handle/200/12294.
    """
    # Iterate through the straight lines.
    step = np.zeros_like(grad)
    q_val = const
    xpt_norms = LA.norm(xpt, axis=1)
    npt = xpt.shape[0]
    for k in range(npt):
        # Set alpha_tr to the step size for the trust-region constraint.
        if xpt_norms[k] > float_tiny * delta:
            alpha_tr = max(delta / xpt_norms[k], 0.0)
        else:
            # The current straight line is basically zero.
            continue

        # Set alpha_quad_pos and alpha_quad_neg to the step size to the extrema
        # of the quadratic function along the positive and negative directions.
        grad_step = grad @ xpt[k]
        curv_step = curv(xpt[k])
        tiny_grad = float_tiny * grad_step
        if (
            grad_step >= 0.0
            and curv_step < -tiny_grad
            or grad_step <= 0.0
            and curv_step > -tiny_grad
        ):
            alpha_quad_pos = max(-grad_step / curv_step, 0.0)
        else:
            alpha_quad_pos = np.inf
        if (
            grad_step >= 0.0
            and curv_step > tiny_grad
            or grad_step <= 0.0
            and curv_step < tiny_grad
        ):
            alpha_quad_neg = min(-grad_step / curv_step, 0.0)
        else:
            alpha_quad_neg = -np.inf

        # Select the step that provides the largest value of the objective
        # function if it improves the current best. The best positive step is
        # either the one that reaches the constraints or the one that reaches
        # the extremum of the objective function along the current direction
        # (only possible if the resulting step is feasible). We test both, and
        # we perform similar calculations along the negative step.
        # N.B.: we select the largest possible step among all the ones that
        # maximize the objective function. This is to avoid returning the zero
        # step in some extreme cases.
        alpha_pos, alpha_neg = alpha_tr, -alpha_tr
        alpha_tr_pow_curv = 0.5 * alpha_tr**2.0 * curv_step
        q_val_pos = const + alpha_pos * grad_step + alpha_tr_pow_curv
        q_val_neg = const + alpha_neg * grad_step + alpha_tr_pow_curv
        if alpha_quad_pos < alpha_pos:
            q_val_quad_pos = (
                const
                + alpha_quad_pos * grad_step
                + 0.5 * alpha_quad_pos**2.0 * curv_step
            )
            if abs(q_val_quad_pos) > abs(q_val_pos):
                alpha_pos = alpha_quad_pos
                q_val_pos = q_val_quad_pos
        if alpha_quad_neg > alpha_neg:
            q_val_quad_neg = (
                const
                + alpha_quad_neg * grad_step
                + 0.5 * alpha_quad_neg**2.0 * curv_step
            )
            if abs(q_val_quad_neg) > abs(q_val_neg):
                alpha_neg = alpha_quad_neg
                q_val_neg = q_val_quad_neg
        abs_q_val_pos, abs_q_val_neg = abs(q_val_pos), abs(q_val_neg)
        if abs_q_val_pos >= abs_q_val_neg and abs_q_val_pos > abs(q_val):
            step = alpha_pos * xpt[k]
            q_val = q_val_pos
        elif abs_q_val_neg > abs_q_val_pos and abs_q_val_neg > abs(q_val):
            step = alpha_neg * xpt[k]
            q_val = q_val_neg

    return step


def _cauchy_geom(const, grad, curv, delta):
    """
    Same as ``bound_constrained_cauchy_step`` without the absolute value.
    """
    # Calculate the initial active set.
    fixed_xl = grad > 0.0
    fixed_xu = grad < 0.0

    # Calculate the Cauchy step.
    cauchy_step = np.zeros_like(grad)
    cauchy_step[fixed_xl] = -np.inf
    cauchy_step[fixed_xu] = np.inf
    if LA.norm(cauchy_step) > delta:
        working = fixed_xl | fixed_xu
        # Calculate the Cauchy step for the directions in the working set.
        g_norm = LA.norm(grad[working])
        delta_reduced = np.sqrt(
            delta**2.0 - np.inner(cauchy_step[~working], cauchy_step[~working])
        )
        if g_norm > np.finfo(float).tiny * abs(delta_reduced):
            mu = max(delta_reduced / g_norm, 0.0)
            cauchy_step[working] = mu * grad[working]

    # Calculate the step that maximizes the quadratic along the Cauchy step.
    grad_step = grad @ cauchy_step
    if grad_step >= 0.0:
        # Set alpha_tr to the step size for the trust-region constraint.
        s_norm = LA.norm(cauchy_step)
        if s_norm > float_tiny * delta:
            alpha_tr = max(delta / s_norm, 0.0)
        else:
            # The Cauchy step is basically zero.
            alpha_tr = 0.0

        # Set alpha_quad to the step size for the maximization problem.
        curv_step = curv(cauchy_step)
        if curv_step < -float_tiny * grad_step:
            alpha_quad = max(-grad_step / curv_step, 0.0)
        else:
            alpha_quad = np.inf

        # Calculate the solution and the corresponding function value.
        alpha = min(alpha_tr, alpha_quad)
        step = alpha * cauchy_step
        q_val = const + alpha * grad_step + 0.5 * alpha**2.0 * curv_step
    else:
        # This case is never reached in exact arithmetic. It prevents this
        # function to return a step that decreases the objective function.
        step = np.zeros_like(grad)
        q_val = const

    return step, q_val


def tangential_byrd_omojokun(fun, grad, hess_prod, delta, n, **kwargs):
    r"""
    Minimize approximately a quadratic function subject to bound constraints in
    a trust region.

    This function solves approximately

    .. math::
        \min_{s\in\mathbb{R}^n}\quad f(s) \quad \text{s.t.} \quad
        \lVert s \rVert \le \Delta.
        
    using a variation of the truncated conjugate gradient method.

    Parameters
    ----------
    fun : callable
        Overall surrogate model function :math:`f` as shown above.
    grad : callable
        Gradient :math:`\nabla f` of the model function:

            ``grad(x) -> ndarray, shape (n,)``

        returns the gradient at ``x``.
    hess_prod : callable
        Product of the Hessian matrix :math:`\nabla^2 f(x)` with any vector :math:`v`:

            ``hess_prod(x, v) -> ndarray, shape (n,)``

        returns the product :math:`\nabla^2 f(x) v`.
    delta : float
        Trust-region radius :math:`\Delta` as shown above.
    n : int
        Dimension of the function.

    Returns
    -------
    ndarray, shape (n,)
        Approximate solution :math:`s`.

    Other Parameters
    ----------------
    improve_tcg : bool, optional
        If True, a solution generated by the truncated conjugate gradient
        method that is on the boundary of the trust region is improved by
        moving around the trust-region boundary on the two-dimensional space
        spanned by the solution and the gradient of the quadratic function at
        the solution (default is True).

    Notes
    -----
    This function originally implements Algorithm 6.2 of [1]_ and is modified 
    to adapt to the UPOQA solver. It is assumed that the origin is feasible 
    with respect to the bound constraints and that ``delta`` is finite and 
    positive.
    
    References
    ----------
    .. [1] Tom M. Ragonneau. *Model-Based Derivative-Free Optimization Methods
       and Software*. PhD thesis, Department of Applied Mathematics, The Hong
       Kong Polytechnic University, Hong Kong, China, 2022. URL:
       https://theses.lib.polyu.edu.hk/handle/200/12294.
    """
    # Copy the arrays that may be modified by the code below.-
    g = grad(np.zeros((n,)))

    # Set the initial iterate and the initial search direction.
    step, sd = np.zeros_like(g), -g
    # objective value may not be monotonic if xform is enabled in upoqa.
    best_step, best_decrease = step, 0.0

    k, reduct = 0, 0.0
    boundary_reached = False
    while k < n:
        # Stop the computations if sd is not a descent direction.
        g_sd = g @ sd
        if g_sd >= -10.0 * float_eps * n * max(1.0, LA.norm(g)):
            break

        # Set alpha_tr to the step size for the trust-region constraint.
        try:
            alpha_tr = _alpha_tr(step, sd, delta)
        except ZeroDivisionError:
            break

        # Stop the computations if a step along sd is expected to give a
        # relatively small reduction in the objective function.
        if -alpha_tr * g_sd <= 1e-8 * reduct:
            break

        # Set alpha_quad to the step size for the minimization problem.
        hess_sd = hess_prod(step, sd)
        curv_sd = sd @ hess_sd
        if curv_sd > float_tiny * abs(g_sd):
            alpha_quad = max(-g_sd / curv_sd, 0.0)
        else:
            alpha_quad = np.inf

        # Stop the computations if the reduction in the objective function
        # provided by an unconstrained step is small.
        alpha = min(alpha_tr, alpha_quad)
        if -alpha * (g_sd + 0.5 * alpha * curv_sd) <= 1e-8 * reduct:
            break

        # Update the iterate.
        if alpha > 0.0:
            step = step + alpha * sd
            g = grad(step)
            reduct = -fun(step)

            if reduct > best_decrease:
                best_step, best_decrease = step, reduct

        if alpha < alpha_tr:
            # The current iteration is a conjugate gradient iteration. Update
            # the search direction so that it is conjugate (with respect to H)
            # to all the previous search directions.
            beta = (g @ hess_sd) / curv_sd
            sd = beta * sd - g
            k += 1
        else:
            # The current iterate is on the trust-region boundary. Add all the
            # active bounds to the working set to prepare for the improvement
            # of the solution, and stop the iterations.
            boundary_reached = True
            break

    # Attempt to improve the solution on the trust-region boundary.
    if kwargs.get("improve_tcg", True) and boundary_reached:
        # Check whether a substantial reduction in the objective function
        # is possible, and set the search direction.
        step_sq = step.dot(step)
        g_sq = g.dot(g)
        g_step = g.dot(step)
        g_sd = -np.sqrt(max(step_sq * g_sq - g_step**2, 0.0))
        sd = g_step * step - step_sq * g
        if (g_sd < -1e-8 * reduct) and np.all(g_sd < -float_tiny * np.abs(sd)):
            sd /= -g_sd

            # Calculate some curvature information.
            hess_sd = hess_prod(step, sd)
            curv_sd = sd @ hess_sd

            # For a range of equally spaced values of tan(0.5 * theta),
            # calculate the reduction in the objective function that would be
            # obtained by accepting the corresponding angle.
            n_samples = 20
            t_samples = np.linspace(1 / n_samples, 1, n_samples)
            sin_values = 2.0 * t_samples / (1.0 + t_samples**2.0)

            cos_values = (1.0 - t_samples**2.0) / (1.0 + t_samples**2.0)
            candidate_steps = np.outer(cos_values, step) + np.outer(sin_values, sd)

            all_reduct = np.empty((n_samples,))
            for i in range(n_samples):
                all_reduct[i] = -fun(candidate_steps[i])

            if np.any(all_reduct > 0.0):
                # No reduction in the objective function is obtained.
                # Accept the angle that provides the largest reduction in the
                # objective function, and update the iterate.
                i_max = np.argmax(all_reduct)
                step = cos_values[i_max] * step + sin_values[i_max] * sd
                g = grad(step)
                reduct = all_reduct[i_max]

                if reduct > best_decrease:
                    best_step, best_decrease = step, reduct

    return best_step


def conjugate_gradient_proj_steinmetz(
    fun: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    hess_prod: Callable[[np.ndarray, np.ndarray], np.ndarray],
    coords_mask: np.ndarray,
    n: int,
    deltas: np.ndarray,
    envelope_delta: float,
    maxiter: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    r"""
    Modified projected conjugate gradient method for a quadratic trust-region
    sub-problem with a structured trust-region.

    The feasible region is the intersection of :math:`q` cylinders

    .. math::
        \mathcal{S} = \bigcap_{i=1}^{q}
        \left\{ s\in\mathbb{R}^{n} : \lVert s^{\mathcal{I}_i} \rVert_{2}
        \le \Delta_i \right\}.

    This function solves approximately

    .. math::
        \min_{s\in\mathcal{S}}\quad f(s).

    Parameters
    ----------
    fun : callable
        Overall surrogate model function :math:`f` as shown above.
    grad : callable
        Gradient :math:`\nabla f` of the model function:

            ``grad(x) -> ndarray, shape (n,)``

        returns the gradient at ``x``.
    hess_prod : callable
        Product of the Hessian matrix :math:`\nabla^2 f(x)` with any vector :math:`v`:

            ``hess_prod(x, v) -> ndarray, shape (n,)``

        returns the product :math:`\nabla^2 f(x) v`.
    coords_mask : ndarray, shape (q, n)
        Bool mask indicating variable indices :math:`\mathcal{I}_i` for each element.
    n : int
        Problem dimension
    deltas : ndarray
        Elemental trust-region radii :math:`\{\Delta_i\}_{i=1}^q` as shown above.
    envelope_delta : float
        A sufficiently large radius ensuring that the structured trust-region
        lies entirely within the ball centered at the origin with this radius.
    maxiter : int, optional
        Maximum number of iterations. Default is None, which sets it to
        :math:`\max(20, 3n)` in which :math:`2n` steps are reserved for projection
        steps.

    Returns
    -------
    ndarray
        Approximate solution to the trust-region subproblem.

    Notes
    -----
    This method uses Steinmetz projection (:func:`~upoqa.utils.projection.steinmetz_proj`) 
    and combined projection (:func:`~upoqa.utils.projection.steinmetz_comb_proj`) to 
    maintain feasibility with respect to the cylindrical constraints.
    """
    s0 = np.zeros(n)
    g = grad(s0)

    # Set the initial iterate and the initial search direction.
    step, sd = np.zeros(n), -g
    fval0 = fun(s0)
    reduct = 0.0
    proj_step, max_proj_step = 0, max(10, 2 * n)
    maxiter = max(10, n) if maxiter is None else int(maxiter)
    maxiter += max_proj_step
    best_step, best_decrease = (
        step,
        0.0,
    )  # objective value may not be monotonic if xform is enabled.

    for it in range(maxiter):
        # Stop the computations if sd is not a descent direction.
        g_sd = g @ sd
        if g_sd >= -10.0 * float_eps * n * max(1.0, LA.norm(g)):
            break

        hess_sd = hess_prod(step, sd)
        curv_sd = sd @ hess_sd

        try:
            # Set alpha_quad to the st+ep size for the minimization problem.
            if curv_sd > float_tiny * abs(g_sd):
                alpha_quad = max(-g_sd / curv_sd, 0.0)
                alpha_tr = _alpha_tr(step, sd, envelope_delta)
            else:
                alpha_quad = np.inf
                alpha_tr = _alpha_tr(step, sd, 1e3 * envelope_delta)
        except ZeroDivisionError:
            break
        alpha = min(alpha_quad, alpha_tr)

        # Stop the computations if the reduction in the objective function
        # provided by an unconstrained step is small.
        if -alpha * (g_sd + 0.5 * alpha * curv_sd) <= 1e-8 * reduct:
            break

        # Update the iterate.
        if alpha > 0.0:
            # Though Dykstra's projection method is an exact projection,
            # it shows limited experimental improvements and is slower.
            # skp1, did_project = dykstra_proj(step + alpha * sd, coords_mask, deltas)

            if it % 2 == 0:
                skp1, did_project = steinmetz_proj(
                    step + alpha * sd, coords_mask, deltas
                )
            else:
                skp1, did_project = steinmetz_comb_proj(
                    step + alpha * sd, coords_mask, deltas
                )

            if did_project:
                # line search along skp1 - step
                proj_step += 1

                if proj_step > max_proj_step:
                    break

                pj_sd = skp1 - step
                g_pj_sd = g @ pj_sd
                hess_pj_sd = hess_prod(step, pj_sd)
                curv_pj_sd = pj_sd @ hess_pj_sd
                if curv_pj_sd > float_tiny * abs(g_pj_sd):
                    alpha_ls = min(1.0, max(-g_pj_sd / curv_pj_sd, 0.0))
                else:
                    alpha_ls = 1.0

                if -alpha_ls * (g_pj_sd + 0.5 * alpha_ls * curv_pj_sd) <= 1e-8 * reduct:
                    break

                skp1 = step + alpha_ls * pj_sd

            reduct = fval0 - fun(skp1)
            step = skp1
            g = grad(step)

            if reduct > best_decrease:
                best_step, best_decrease = step, reduct

        else:
            break

        if did_project:
            sd = -g
        elif curv_sd > float_tiny * abs(g_sd):
            # The current iteration is a conjugate gradient iteration. Update
            # the search direction so that it is conjugate (with respect to H)
            # to all the previous search directions.
            beta = (g @ hess_sd) / curv_sd
            sd = beta * sd - g
        else:
            # The curvature along sd is nonpositive or negligible, so the
            # conjugacy update is unsafe. Restart with steepest descent.
            sd = -g

    return best_step


def get_arrays_tol(*arrays):
    """
    Get a relative tolerance for a set of arrays.

    Parameters
    ----------
    *arrays: tuple
        Set of ndarray to get the tolerance for.

    Returns
    -------
    float
        Relative tolerance for the set of arrays.

    Raises
    ------
    ValueError
        If no array is provided.
    """
    if len(arrays) == 0:
        raise ValueError("At least one array must be provided.")
    size = max(array.size for array in arrays)
    weight = max(
        np.max(np.abs(array[np.isfinite(array)]), initial=1.0) for array in arrays
    )
    return 10.0 * float_eps * max(size, 1.0) * weight


def _alpha_tr(step, sd, delta):
    step_sd = step.dot(sd)
    sd_sq = sd.dot(sd)
    dist_tr_sq = delta**2.0 - step.dot(step)
    temp = np.sqrt(max(step_sd**2.0 + sd_sq * dist_tr_sq, 0.0))
    if step_sd <= 0.0 and sd_sq > float_tiny * abs(temp - step_sd):
        alpha_tr = max((temp - step_sd) / sd_sq, 0.0)
    elif abs(temp + step_sd) > float_tiny * dist_tr_sq:
        alpha_tr = max(dist_tr_sq / (temp + step_sd), 0.0)
    else:
        raise ZeroDivisionError
    return alpha_tr
