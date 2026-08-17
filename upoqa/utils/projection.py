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
Projection Operator
===================

Implement several projection operators for the use of the projected 
gradient method.
"""

import numpy as np
import numpy.linalg as LA
from typing import List, Optional, Union


def get_violation_ratio(
    s: np.ndarray, coords: Union[List[np.ndarray], np.ndarray], radii: np.ndarray
) -> np.ndarray:
    """
    Compute the ratio ‖s[coord]‖₂ / radius for each element.

    Parameters
    ----------
    s : ndarray, shape (n,)
        Trial point.
    coords : ndarray, shape (ele_num, n) or list of ndarray
        List of index arrays, one for each element, or a mask array
        of which each line is a mask vector indicating the indices.
    radii : ndarray, shape (ele_num,)
        Trust-region radii for each element.

    Returns
    -------
    ndarray, shape (ele_num,)
        Violation ratios (>1 means infeasible).
    """
    s_pow = np.power(s, 2)
    return np.array([np.sqrt(s_pow[coord].sum()) for coord in coords]) / radii


def _shrink(
    s: np.ndarray,
    coords_mask: np.ndarray,
    shrink_group: np.ndarray,
    negligible_group: np.ndarray,
    shrink_coord_mask: np.ndarray,
    max_violation: float,
    radii: np.ndarray,
) -> np.ndarray:
    """
    Perform one shrinking step for Steinmetz projection.

    The current iterate ``s`` is scaled along the subspace defined by
    ``shrink_coord_mask`` so that the new iterate becomes feasible, or
    a new element is pulled into the shrinking set.

    Parameters
    ----------
    s : ndarray, shape (n,)
        Current iterate.
    coords_mask : ndarray, shape (ele_num, n)
        Boolean mask; ``coords_mask[i]`` indicates the variables belonging to
        element ``i``.
    shrink_group : ndarray, shape (ele_num,)
        Mask indicating which elements are already in the shrinking set.
    negligible_group : ndarray, shape (ele_num,)
        Mask indicating elements that will never become active.
    shrink_coord_mask : ndarray, shape (n,)
        Mask indicating the union of variables spanned by the current active set.
    max_violation : float
        Current maximum violation (≥ 1).
    radii : ndarray, shape (ele_num,)
        Trust-region radii.

    Returns
    -------
    new_max_violation : float
        Scaled violation (should be <= 1 unless numerical issues).
    """
    max_vio_pow = max_violation**2
    shrink_ratio, shrink_ratio_idx = 1 / max_violation, None
    for idx in range(len(radii)):
        if negligible_group[idx]:
            continue
        coord = coords_mask[idx]
        s_inter, s_diff = s[coord & shrink_coord_mask], s[coord & (~shrink_coord_mask)]
        inter_norm_pow = np.einsum("i,i->", s_inter, s_inter)
        diff_norm_pow = np.einsum("i,i->", s_diff, s_diff)
        radius_pow = radii[idx] ** 2
        if inter_norm_pow + diff_norm_pow <= radius_pow:
            negligible_group[idx] = np.True_
            continue
        denominator = radius_pow * max_vio_pow - inter_norm_pow
        if denominator <= 0.0:
            continue
        ratio = (diff_norm_pow / denominator) ** 0.5
        if ratio > shrink_ratio:
            shrink_ratio, shrink_ratio_idx = ratio, idx
    s[shrink_coord_mask] *= shrink_ratio
    if shrink_ratio_idx is not None:
        shrink_group[shrink_ratio_idx] = np.True_
        negligible_group[shrink_ratio_idx] = np.True_
        shrink_coord_mask |= coords_mask[shrink_ratio_idx]
    return max_violation * shrink_ratio


def steinmetz_comb_proj(
    s: np.ndarray,
    coords_mask: np.ndarray,
    radii: np.ndarray,
    pretol: float = 1e-8,
    preit: int = 4,
):
    """
    Combined averaged and Steinmetz projection onto the intersection of cylinders.

    First applies a small number of averaged-projection iterations, then
    switches to the Steinmetz shrinking procedure.

    Parameters
    ----------
    s : ndarray, shape (n,)
        Point to project.
    coords_mask : ndarray, shape (ele_num, n)
        Boolean mask; ``coords_mask[i]`` indicates the variables belonging to
        element ``i``.
    radii : ndarray, shape (ele_num,)
        Trust-region radii.
    pretol : float
        Tolerance for the initial averaged projection.
    preit : int
        Maximum iterations for the initial averaged projection.

    Returns
    -------
    ndarray, shape (n,)
        Feasible point.
    bool
        True if projection actually moved the point.
    """
    sk, did_project = average_proj(s, coords_mask, radii, tol=pretol, maxit=preit)
    if not did_project and preit > 0:
        return sk, False
    it = 0
    violation = get_violation_ratio(sk, coords_mask, radii)
    max_vio_idx = np.argmax(violation)
    max_violation = violation[max_vio_idx]
    if max_violation <= 1.0:
        return sk, did_project
    shrink_group = np.zeros(radii.size, dtype=bool)
    shrink_coord_mask = coords_mask[max_vio_idx].copy()
    shrink_group[max_vio_idx] = np.True_
    negligible_group = (violation <= 1) | shrink_group
    # Each _shrink call absorbs at least one new element into the shrink
    # group, so the loop needs at most ele_num iterations in exact
    # arithmetic. The cap and the finiteness check turn a non-converging
    # loop (e.g. one fed with NaN) into a controlled LinAlgError, which the
    # trust-region solver catches and turns into a restart.
    max_shrink_iter = radii.size + 5
    while True:
        if max_violation <= 1.0 + 1e-12:
            break
        if not np.isfinite(max_violation) or it >= max_shrink_iter:
            raise LA.LinAlgError(
                "Steinmetz projection failed to converge "
                f"(max violation = {max_violation})."
            )
        max_violation = _shrink(
            sk,
            coords_mask,
            shrink_group,
            negligible_group,
            shrink_coord_mask,
            max_violation,
            radii,
        )
        it += 1
    return sk, it >= 1


def steinmetz_proj(s: np.ndarray, coords_mask: np.ndarray, radii: np.ndarray):
    """
    Approximate projection onto the intersection of cylinders (which forms a 
    Steinmetz solid in the special case of two cylinders) using a shrinking strategy.

    Parameters
    ----------
    s : ndarray, shape (n,)
        Point to project.
    coords_mask : ndarray, shape (ele_num, n)
        Boolean masks for each element's variables.
    radii : ndarray, shape (ele_num,)
        Trust-region radii.

    Returns
    -------
    ndarray, shape (n,)
        Feasible point.
    bool
        True if projection actually moved the point.
    """
    sk = s.copy()
    it = 0
    violation = get_violation_ratio(sk, coords_mask, radii)
    max_vio_idx = np.argmax(violation)
    max_violation = violation[max_vio_idx]
    if max_violation <= 1.0:
        return sk, False
    shrink_group = np.zeros(radii.size, dtype=bool)
    shrink_coord_mask = coords_mask[max_vio_idx].copy()
    shrink_group[max_vio_idx] = np.True_
    negligible_group = (violation <= 1) | shrink_group
    # Each _shrink call absorbs at least one new element into the shrink
    # group, so the loop needs at most ele_num iterations in exact
    # arithmetic. The cap and the finiteness check turn a non-converging
    # loop (e.g. one fed with NaN) into a controlled LinAlgError, which the
    # trust-region solver catches and turns into a restart.
    max_shrink_iter = radii.size + 5
    while True:
        if max_violation <= 1.0 + 1e-12:
            break
        if not np.isfinite(max_violation) or it >= max_shrink_iter:
            raise LA.LinAlgError(
                "Steinmetz projection failed to converge "
                f"(max violation = {max_violation})."
            )
        max_violation = _shrink(
            sk,
            coords_mask,
            shrink_group,
            negligible_group,
            shrink_coord_mask,
            max_violation,
            radii,
        )
        it += 1
    return sk, it >= 1


def average_proj(
    s: np.ndarray,
    coords: Union[List[np.ndarray], np.ndarray],
    radii: np.ndarray,
    tol: float = 1e-8,
    maxit: Optional[int] = None,
):
    """
    Averaged projection onto the intersection of cylinders, see [1]_ for more
    details about its convergence.

    Parameters
    ----------
    s : ndarray, shape (n,)
        Initial point.
    coords : ndarray or list of 1D array
        Boolean masks or index arrays for each element's variables.
    radii : ndarray, shape (ele_num,)
        Trust-region radii.
    tol : float, default=1e-8
        Tolerance for convergence.
    maxit : int, optional
        Maximum iterations (default: 10*n).

    Returns
    -------
    ndarray, shape (n,)
        Feasible point.
    bool
        True if projection actually moved the point.

    References
    ----------
    .. [1] Lewis, A.S., Luke, D.R. & Malick, J. Local Linear Convergence for
        Alternating and Averaged Nonconvex Projections. *Found Comput Math* 9,
        485-513 (2009). https://doi.org/10.1007/s10208-008-9036-y
    """
    sk = s.copy()
    ele_num = len(coords)
    act_eles = np.ones(ele_num, dtype=bool)
    maxit = 10 * s.size if maxit is None else maxit
    did_project = False
    for _ in range(maxit):
        s_avg = np.zeros_like(sk)
        sk_pow = np.power(sk, 2)
        proj_ele_num = 0
        max_vio = 0.0
        for i in range(ele_num):
            if not act_eles[i]:
                continue
            coord = coords[i]
            vio = np.sqrt(sk_pow[coord].sum()) / radii[i]
            max_vio = max(max_vio, vio)
            if vio > 1.0:
                s_avg[coord] -= (
                    sk[coord] * (vio - 1) / vio
                )  # * (1 / vio) = * (1 - (vio - 1) / vio)
                proj_ele_num += 1
            else:
                act_eles[i] = False
        if proj_ele_num == 0:
            break
        sk += s_avg / proj_ele_num
        did_project = True
        if max_vio < 1.0 + tol:
            break
    return sk, did_project


def dykstra_proj(
    s: np.ndarray,
    coords_mask: np.ndarray,
    radii: np.ndarray,
    tol: float = 1e-6,
    maxit: Optional[int] = None,
):
    """
    Dykstra's projection method onto the intersection of cylinders, see [1]_.

    Parameters
    ----------
    s : ndarray, shape (n,)
        Initial point.
    coords_mask : ndarray, shape (ele_num, n)
        Boolean mask; ``coords_mask[i]`` indicates the variables belonging to
        element ``i``.
    radii : ndarray, shape (ele_num,)
        Trust-region radii.
    tol : float, default=1e-6
        Tolerance for convergence.
    maxit : int, optional
        Maximum iterations (default: 10*n).

    Returns
    -------
    ndarray, shape (n,)
        Feasible point.
    bool
        True if projection actually moved the point.

    References
    ----------
    .. [1] Boyle, J. P.; Dykstr, R. L. (1986). "A Method for Finding Projections
        onto the Intersection of Convex Sets in Hilbert Spaces". Advances in Order
        Restricted Statistical Inference. Lecture Notes in Statistics. Vol. 37.
        pp. 28-47. doi:10.1007/978-1-4613-9940-7_3. ISBN 978-0-387-96419-5.
    """
    sk = s.copy()
    max_radius = radii.max()
    ele_num = len(coords_mask)
    maxit = 10 * s.size if maxit is None else maxit
    did_project = False
    y = [np.zeros(coords_mask[i].sum()) for i in range(ele_num)]

    for _ in range(maxit):
        do_proj = False
        cI = 0.0
        for i in range(ele_num):
            coord = coords_mask[i]
            sk_sub_y = sk[coord] - y[i]
            vio = np.sqrt(np.einsum("i,i->", sk_sub_y, sk_sub_y)) / radii[i]
            # Update increment
            prev_y = y[i].copy()
            if vio > 1:
                y[i] = sk_sub_y * (1 / vio - 1)
                sk_sub_y /= vio
                do_proj = True
            else:
                y[i].fill(0.0)

            # Update iterate
            sk[coord] = sk_sub_y

            # Stop condition
            y_diff = prev_y - y[i]
            cI += np.einsum("i,i->", y_diff, y_diff)
        if not do_proj:
            break
        did_project = True
        if cI < (tol * max_radius) ** 2:
            break

    return sk, did_project