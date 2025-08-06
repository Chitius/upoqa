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
Interpolation Set
=================

Maintain two classes which represent the elemental and overall interpolation 
sets respectively.
"""

import numpy as np
import warnings
from typing import Any, Callable, Optional, Union, Tuple, List

__all__ = ["InterpSet", "OverallInterpSet"]


class InterpSet:
    """
    The interpolation set for a quadratic surrogate model.

    Parameters
    ----------
    n : int
        Problem dimension
    npt : int, optional
        Number of interpolation points. By default, it is set to ``2 * n + 1``.
        If the user provides a custom value, it should fall within the range
        ``[2 * n + 1, (n + 1) * (n + 2) / 2]``.

    Attributes
    ----------
    n : int
        Problem dimension
    npt : int
        Number of interpolation points. By default, it is set to ``2 * n + 1``.
    interp_set_Y : ndarray, shape (npt, n)
        Interpolation points
    interp_set_fval : ndarray, shape (npt,)
        Objective function values at the interpolation points
    x_opt_idx : int
        Index of the interpolation point with the lowest objective function value.
    x_anchor_idx : int
        Index of the interpolation point used as the center for solving the trust region
        subproblem.
    x_opt
        See :attr:`~InterpSet.x_opt` property
    x_anchor
        See :attr:`~InterpSet.x_anchor` property
    f_opt
        See :attr:`~InterpSet.f_opt` property
    f_anchor
        See :attr:`~InterpSet.f_anchor` property
    """

    def __init__(self, n: int, npt: Optional[int] = None) -> None:
        self.n = n
        self.npt = npt or (2 * self.n + 1)
        self.npt = max(self.n + 2, min(int((self.n + 1) * (self.n + 2) / 2), self.npt))
        self.interp_set_Y = np.empty((self.npt, self.n))  # interpolation set
        self.interp_set_fval = np.empty((self.npt,))  # obj values
        # the point with the lowest obj value
        self.x_opt_idx: Optional[int] = None
        # the point that we actually use as a start point for geometry-improving
        # and calculating trust region step
        self.x_anchor_idx: Optional[int] = None
        self.last_update_idx: Optional[int] = None
        self.just_deleted_point: Optional[np.ndarray] = None
        self.last_init_step_size: Optional[float] = None

    def get_opt(self) -> Tuple[np.ndarray, float]:
        """
        Retrieve the interpolation point with the lowest objective function value
        from the interpolation set.

        Returns
        -------
        ndarray, shape (n,)
            Interpolation point with the lowest objective function value.
        float
            The lowest objective function value found in the interpolation set.
        """
        return self.interp_set_Y[self.x_opt_idx].copy(), float(
            self.interp_set_fval[self.x_opt_idx]
        )

    def get_anchor(self) -> Tuple[np.ndarray, float]:
        """
        Retrieve the interpolation point used as the center for solving
        the trust region subproblem.

        Returns
        -------
        ndarray, shape (n,)
            Interpolation point serving as the center for the trust region subproblem
        float
            Value of objective function at the anchor point
        """
        return (
            self.interp_set_Y[self.x_anchor_idx].copy(),
            float(self.interp_set_fval[self.x_anchor_idx]),
        )

    @property
    def x_opt(self) -> np.ndarray:
        """
        The interpolation point with the lowest objective function value
        """
        return self.interp_set_Y[self.x_opt_idx]

    @property
    def x_anchor(self) -> np.ndarray:
        """
        The interpolation point serving as the center for the trust region subproblem
        """
        return self.interp_set_Y[self.x_anchor_idx]

    @property
    def f_opt(self) -> float:
        """
        The lowest objective function value in the interpolation set
        """
        return float(self.interp_set_fval[self.x_opt_idx])

    @property
    def f_anchor(self) -> float:
        """
        value of objective function at the anchor point (``self.x_anchor``)
        """
        return float(self.interp_set_fval[self.x_anchor_idx])

    def update_x_opt(self) -> None:
        """
        Update the optimal interpolation point (``self.x_opt``) based on the current
        objective function values in the interpolation set.

        If ``self.x_anchor`` is None, it will be updated to ``self.x_opt``.
        """
        self.x_opt_idx = np.argsort(self.interp_set_fval)[0]
        if self.x_anchor_idx is None:
            self.x_anchor_idx = self.x_opt_idx

    def get_interp_set(
        self, idx: Optional[int] = None
    ) -> Tuple[np.ndarray, Union[np.ndarray, float]]:
        """
        Retrieve the interpolation points or a single point at a specified index.

        Parameters
        ----------
        idx : int, optional
            Index of the interpolation point to retrieve. If not provided, return the
            entire interpolation set.

        Returns
        -------
        ndarray, shape (npt, n) or ndarray, shape (n,)
            The interpolation set if ``idx`` is None, or the interpolation point at
            the specified index ``idx``.
        ndarray, shape (npt,) or float
            Values of objective function for the interpolation set if ``idx`` is None,
            or objective function value at the interpolation point with index ``idx``.
        """
        if idx is None:
            return self.interp_set_Y, self.interp_set_fval
        else:
            return self.interp_set_Y[idx], self.interp_set_fval[idx]

    def update_point_on_idx(
        self, x: np.ndarray, idx: int, fval: float, update_x_anchor: bool = True
    ) -> None:
        """
        Replace the interpolation point at index ``idx`` with a new point ``x`` and
        update its corresponding function value ``fval``.

        Parameters
        ----------
        x : ndarray, shape (n,)
            The new point to be added to the interpolation set.
        idx : int
            The index of the point to be updated.
        fval : float
            The function value at ``x``.
        """
        if update_x_anchor and fval < self.f_anchor:
            self.x_anchor_idx = idx
        self.just_deleted_point = self.interp_set_Y[idx].copy()
        self.interp_set_Y[idx] = x
        self.interp_set_fval[idx] = fval
        self.last_update_idx = idx
        self.update_x_opt()

    def init_interp_set(
        self,
        fun: Callable[[np.ndarray], float],
        center: np.ndarray,
        step_size: float = 1.0,
    ):
        """
        Initialize the interpolation set and computes the corresponding objective 
        function values.

        Parameters
        ----------
        fun : callable
            A callable that takes a point in the interpolation set and returns its
            objective function value.

                ``fun(x: np.ndarray) -> (ndarray | float)``

            It should internally track its evaluation count and raise 
            :class:`~upoqa.utils.manager.MaxEvalNumReached` if the maximum number of 
            evaluations is exceeded, or ``ValueError`` if the function returns an 
            invalid value.
        center: ndarray, shape (n,)
            Center point around which the interpolation set is generated.
        step_size: float, default=1.0
            The step size used to generate interpolation points around the center point.
        """
        self.x_anchor_idx = None
        self.x_opt_idx = None
        self.last_update_idx = None
        self.just_deleted_point = None
        self.last_init_step_size = step_size
        self.init_interp_set_Y(center, step_size)
        self._update_interp_set_fval(fun)
        self.update_x_opt()
        return self

    def _update_interp_set_fval(self, fun: Callable[[np.ndarray], float]) -> None:
        """
        Update objective function values for all points in the interpolation set.
        If ``run_in_parallel`` is True, the function evaluations will be performed in 
        parallel.

        Parameters
        ----------
        fun: callable
            The objective function used to compute the function values.
        """
        for i in range(self.npt):
            self.interp_set_fval[i] = fun(self.interp_set_Y[i])

    def set_interp_set_fval(self, fvals: np.ndarray) -> None:
        """
        Set objective function values of the interpolation set to the provided values
        ``fvals``.

        Parameters
        ----------
        fval : nd.array, shape (npt,)
            The user-defined objective function values for the interpolation points.
        """
        fvals = np.asarray(fvals).reshape((-1,))
        assert (
            fvals.size == self.npt
        ), f"`fvals` should have the shape ({self.npt},) instead of {fvals.shape}."
        self.interp_set_fval = fvals
        self.x_anchor_idx = None
        self.update_x_opt()

    def init_interp_set_Y(self, center: np.ndarray, step_size: float = 1.0) -> None:
        """
        Initialize the interpolation set.

        The center point is set to ``center``, the next ``2n`` points are constructed
        as a CFD (central finite difference) stencil, and the remaining points are
        generated by combining previous points.

        Parameters
        ----------
        center : ndarray of shape (n,)
            Center point of the interpolation set. Must have size equal to the
            problem dimension.
        step_size : float, default=1.0
            Step size used to generate interpolation points around the center.

        Raises
        ------
        AssertionError
            If ``center`` size does not match problem dimension

        References
        ----------
        .. [1] Tom M. Ragonneau. *Model-Based Derivative-Free Optimization Methods
            and Software*. PhD thesis, Department of Applied Mathematics, The Hong
            Kong Polytechnic University, Hong Kong, China, 2022. URL:
            https://theses.lib.polyu.edu.hk/handle/200/12294.

        Notes
        -----
        The interpolation set is constructed as follows:
        
        - Point 0: ``center``
        - Points 1 to n: ``center + step_size * e_i``
        - Points n+1 to 2*n: ``center - step_size * e_i``
        - Points > 2*n: combinations of previous points
        """
        self.init_step_size = float(step_size)
        assert (
            center.size == self.n
        ), f"`center` should have the size {self.x_opt.size} instead of {center.size}."
        center = center.reshape((self.n,))
        interp_stencil_set = np.zeros((self.npt, self.n), dtype=np.float64)

        for k in range(self.npt):
            if 1 <= k <= self.n:
                interp_stencil_set[k, k - 1] = self.init_step_size
            elif self.n < k <= 2 * self.n:
                interp_stencil_set[k, k - self.n - 1] = -self.init_step_size
            elif k > 2 * self.n:
                shift = (k - self.n - 1) // self.n
                i = k - (1 + shift) * self.n - 1
                j = (i + shift) % self.n  # j < self.n
                interp_stencil_set[k, i] = interp_stencil_set[i + 1, i]
                interp_stencil_set[k, j] = interp_stencil_set[j + 1, j]

        self.interp_set_Y = interp_stencil_set + center
        self.x_anchor_idx, self.x_opt_idx = None, None


class OverallInterpSet:
    r"""
    The interpolation set for an overall surrogate model of the partially separable form
    :math:`m(x) = \sum_i m_i(x)`, where each :math:`m_i` is an element model.

    Note that this interpolation set does not directly use the stored points to construct
    the interpolation system; its primary functionality is to manage the elemental
    interpolation sets.

    Parameters
    ----------
    n : int
        Problem dimension
    set_size : int
        The initial number of points in the interpolation set
    max_size : int
        Maximum number of points that the interpolation set can hold
    ele_interp_sets : dict
        A dictionary of interpolation sets, one for each elemental surrogate model.
    proj_onto_ele : callable
        Projection function mapping full-space points to element subspaces:

        .. code-block:: python

            (full_point: ndarray, element_name: Any) -> element_point: ndarray

        For example, ``proj_onto_ele([1.0, 2.0], "f_1")`` yields ``[1.0,]``, where
        ``"f_1"`` denotes a function that depends only on the first variable of ``x``.

    Attributes
    ----------
    n : int
        Problem dimension
    ele_interp_sets : dict
        A dictionary of interpolation sets, one for each element model.
    proj_onto_ele : callable
        Projection function mapping full-space points to element subspaces:

        .. code-block:: python

            (full_point: ndarray, element_name: Any) -> element_point: ndarray

        For example, ``proj_onto_ele([1.0, 2.0], "f_1")`` yields ``[1.0,]``, where
        ``"f_1"`` denotes a function that depends only on the first variable of ``x``.
    ele_names : list
        List of names for each element function
    ele_idxs : list of int
        List of indices corresponding to the element functions
    npts : np.ndarray, shape (ele_num,)
        Number of points in each elemental interpolation set
    interp_set_Y : ndarray, shape (set_size, n)
        Interpolation points for the overall surrogate model
    interp_set_fval : ndarray, shape (set_size,)
        Objective function values at the interpolation points for the overall surrogate model
    interp_set_fval_eles : list of list of float
        Values of element functions at the interpolation points for the overall surrogate model
    interp_set_extra_fval : ndarray
        Values of the extra objective function at the interpolation points for the overall
        surrogate model. The "extra function" represents the differentiable white-box
        component of the objective.
    x_opt_idx : int
        Index of the point with the lowest objective function value in the overall
        interpolation set
    enable : np.ndarray, shape (set_size,)
        Boolean array indicating whether each point in the interpolation set is enabled
        (True) or disabled (False).
    ele_num
        See :attr:`~OverallInterpSet.ele_num` property
    set_size
        See :attr:`~OverallInterpSet.set_size` property
    disable
        See :attr:`~OverallInterpSet.disable` property
    capacity
        See :attr:`~OverallInterpSet.capacity` property
    unallocated_idx
        See :attr:`~OverallInterpSet.unallocated_idx` property
    allocated_idx
        See :attr:`~OverallInterpSet.allocated_idx` property
    x_opt
        See :attr:`~OverallInterpSet.x_opt` property
    f_opt
        See :attr:`~OverallInterpSet.f_opt` property
    """

    def __init__(
        self,
        n: int,
        set_size: int,
        max_size: int,
        ele_interp_sets: List[InterpSet],
        proj_onto_ele: Callable[[np.ndarray, Any], np.ndarray],
        ele_names: List[Any],
    ) -> None:
        self.n = n
        self.ele_interp_sets = ele_interp_sets

        # Store the names and indices of all element functions
        self.ele_names = ele_names
        self.ele_idxs = list(range(len(ele_names)))

        # Number of points in each elemental interpolation set
        self.npts = np.asarray(
            [ele_interp_sets[ele_idx].npt for ele_idx in self.ele_idxs], dtype=int
        )

        # Initialize the interpolation set with placeholder values
        self.interp_set_Y = np.empty((set_size, self.n))
        self.interp_set_fval = np.ones((set_size,)) * np.inf
        # Values of element functions at the interpolation points
        self.interp_set_fval_eles: List[Union[List[float], List[None]]] = [
            [None for _ in self.ele_idxs]
        ] * set_size
        self.interp_set_extra_fval = np.zeros(
            set_size,
        )
        # Flags indicating whether each point is enabled in the interpolation set
        self.enable = np.zeros((set_size,), dtype=bool)

        # Index of the point with the lowest objective function value
        self.x_opt_idx = None
        # Function to project a point onto the subspace of a specified element function
        self.proj_onto_ele = proj_onto_ele
        assert callable(self.proj_onto_ele)
        # Ensure the maximum size of the interpolation set is at least twice the initial size
        self.max_size = max(max_size, 2 * self.set_size)

    @property
    def ele_num(self) -> int:
        """Number of elements"""
        return len(self.ele_idxs)

    @property
    def set_size(self) -> int:
        """Number of currently enabled points in the interpolation set"""
        return int(np.sum(self.enable))

    @property
    def disable(self) -> np.ndarray:
        """
        Boolean array indicating which points are disabled in the interpolation set
        """
        return ~self.enable

    @property
    def capacity(self) -> int:
        """
        Total capacity of the interpolation set, including enabled and disabled
        points.
        """
        return len(self.enable)

    @property
    def unallocated_idx(self) -> np.ndarray:
        """Indices of unallocated (disabled) points in the interpolation set"""
        return np.where(self.enable == False)[0]

    @property
    def allocated_idx(self) -> np.ndarray:
        """Indices of allocated (enabled) points in the interpolation set"""
        return np.where(self.enable)[0]

    def _enlarge_capacity(self) -> None:
        """
        Double the capacity of the overall interpolation set by appending new points
        with placeholder values. This is used to dynamically expand the interpolation set
        when the current capacity is insufficient.
        """
        enlarge_size = self.capacity
        # Append new points to the interpolation set with placeholder values
        appended_interp_set_Y = np.empty((enlarge_size, self.n))
        self.interp_set_Y = np.append(self.interp_set_Y, appended_interp_set_Y, axis=0)
        appended_interp_set_fval = np.ones((enlarge_size,)) * np.inf
        self.interp_set_fval = np.append(
            self.interp_set_fval, appended_interp_set_fval, axis=0
        )
        self.interp_set_fval_eles = (
            self.interp_set_fval_eles
            + [
                [None for _ in self.ele_idxs],
            ]
            * enlarge_size
        )
        self.interp_set_extra_fval = np.append(
            self.interp_set_extra_fval, np.zeros(enlarge_size)
        )
        self.enable = np.append(self.enable, np.zeros((enlarge_size), dtype=bool))

    def append(
        self,
        x: np.ndarray,
        fval: float,
        fval_eles: List[float],
        extra_fval: float = 0.0,
    ) -> np.ndarray:
        """
        Add a new point and its associated function values to the interpolation set.

        Parameters
        ----------
        x : ndarray, shape (n,)
            The point to be added to the interpolation set
        fval : float
            Value of objective function at ``x``
        fval_eles : list of float
            Values of element functions at ``x``
        extra_fval : float, default=0.0
            Value of extra objective function at ``x``.
            The "extra function" represents the differentiable white-box component
            of the objective.

        Returns
        -------
        ndarray
            The index (or indices) where the point and its values were stored in the
            interpolation set.
        """
        # Find indices of unallocated (disabled) points in the interpolation set
        ind = self.unallocated_idx
        # If no unallocated points are available, enlarge the capacity of the interpolation set
        if ind.size == 0:
            self._enlarge_capacity()
            ind = self.unallocated_idx
        # Save the point and its function values at the first available index
        ind = ind[:1]
        self.interp_set_Y[ind] = x
        self.interp_set_fval[ind] = fval
        self.interp_set_fval_eles[ind[0]] = fval_eles
        self.interp_set_extra_fval[ind[0]] = extra_fval
        self.enable[ind] = True
        # Update the index of the point with the lowest objective function value
        self.update_x_opt()
        return ind

    def get_ele_fvals(
        self, idx: int
    ) -> Tuple[Optional[Union[List[float], List[None]]], Optional[float]]:
        """
        Retrieve the element function values and the extra function value at a specific
        index in the interpolation set.

        Parameters
        ----------
        idx : int
            Index of the point in the interpolation set

        Returns
        -------
        list of float or None
            Values of element functions at the interpolation point of the specified index.
            If the point does not exist, return None.
        float or None
            Value of the extra function at the interpolation point of the specified index.
            If the point does not exist, return None.
        """
        if self.enable[idx]:
            return self.interp_set_fval_eles[idx], float(
                self.interp_set_extra_fval[idx]
            )
        else:
            return None, None

    def get_opt(self, verbose: bool = False) -> Union[
        Tuple[None, None],
        Tuple[None, None, None, None],
        Tuple[np.ndarray, float],
        Tuple[np.ndarray, float, List[float], float],
    ]:
        """
        Retrieve the interpolation point with the lowest objective function value and
        its associated data.

        Parameters
        ----------
        verbose : bool, default=False
            If True, return additional information including the element function values
            and the extra function value.

        Returns
        -------
        x_opt : ndarray, shape (n,)
            Interpolation point with the lowest objective function value
        f_opt : float
            The lowest objective function value found in the interpolation set
        fval_eles : list of float, optional
            Values of element functions at the optimal point, if ``verbose=True``.
        extra_fval : float, optional
            Value of the extra function at the optimal point, if ``verbose=True``.
        """
        if self.x_opt_idx is None:
            if verbose:
                return None, None, None, None
            else:
                return None, None

        if not verbose:
            return (
                self.interp_set_Y[self.x_opt_idx].copy(),
                float(self.interp_set_fval[self.x_opt_idx]),
            )
        else:
            fvals, extra_fval = self.get_ele_fvals(self.x_opt_idx)
            return (
                self.interp_set_Y[self.x_opt_idx].copy(),
                float(self.interp_set_fval[self.x_opt_idx]),
                fvals,
                extra_fval,
            )

    @property
    def x_opt(self) -> np.ndarray:
        """
        Interpolation point with the lowest objective function value
        """
        return self.interp_set_Y[self.x_opt_idx]

    @property
    def f_opt(self) -> float:
        """
        Lowest objective function value in the interpolation set
        """
        return float(self.interp_set_fval[self.x_opt_idx])

    def update_x_opt(self) -> None:
        """
        Update the optimal interpolation point (``self.x_opt``) based on the current
        objective function values in the interpolation set.
        """
        self.x_opt_idx = np.argsort(self.interp_set_fval)[0]

    def update_fval_with_new_weights_and_xforms(
        self,
        weights: np.ndarray,
        xforms: List[Optional[List[Callable[[np.ndarray], Union[np.ndarray, float]]]]],
        extra_fun_eval: Callable[[np.ndarray], float],
    ) -> bool:
        """
        Update all objective function values in the interpolation set based on new weights
        and transformations.

        Parameters
        ----------
        weights : ndarray, shape (ele_num,)
            Weights applied to the element function values
        xforms : list of list of callables
            A list of transformations applied to the element function values. Each
            transformation is a callable function or None if no transformation is applied.
        extra_fun_eval : callable
            The "extra function" included in the objective function, representing the
            differentiable white-box component of the objective:

                ``extra_fun_eval(x: ndarray) -> float``

            where ``x`` is a 1-D array with shape (n,).

        Returns
        -------
        bool
            True if the optimal point (``x_opt``) changes after the update, otherwise False.
        """
        for idx in self.allocated_idx:
            value = 0.0
            for ele_idx in self.ele_idxs:
                if xforms[ele_idx] is not None:
                    value_ele = (
                        xforms[ele_idx][0](self.interp_set_fval_eles[idx][ele_idx])
                        * weights[ele_idx]
                    )
                else:
                    value_ele = (
                        weights[ele_idx] * self.interp_set_fval_eles[idx][ele_idx]
                    )
                value += value_ele
            self.interp_set_extra_fval[idx] = extra_fun_eval(self.interp_set_Y[idx])
            self.interp_set_fval[idx] = value + self.interp_set_extra_fval[idx]
        old_x_opt_idx = self.x_opt_idx
        self.update_x_opt()
        return old_x_opt_idx != self.x_opt_idx

    def get_interp_set(
        self, idx: Optional[int] = None
    ) -> Tuple[np.ndarray, Union[np.ndarray, float]]:
        """
        Retrieve the interpolation points or a single point at a specific index.

        Parameters
        ----------
        idx : int, optional
            Index of the interpolation point to retrieve. If None, return the entire
            interpolation set.

        Returns
        -------
        ndarray, shape (set_size, n) or ndarray, shape (n,)
            The interpolation point set if ``idx`` is None, or the interpolation point at
            the specified index ``idx``.
        ndarray, shape (set_size,) or float
            Objective function values at the interpolation points if ``idx`` is None,
            or objective function value at the interpolation point with index ``idx``.
        """
        if idx is None:
            return self.interp_set_Y, self.interp_set_fval
        else:
            return self.interp_set_Y[idx], self.interp_set_fval[idx]

    def delete_points(self, delete_idx: Union[np.ndarray, list, int]) -> None:
        """
        Delete specified points from the interpolation set.

        Parameters
        ----------
        delete_idx : array_like or int
            Indices (or a single index) of the points to delete from the interpolation set.

        Raises
        ------
        RuntimeWarning
            If attempting to delete a point that does not exist in the interpolation set.
        """
        if np.any(self.disable[delete_idx]):
            msg = "(Overall Interp Set) You're trying to delete a point that does not exist!"
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        if isinstance(delete_idx, int):
            delete_idx = np.array(
                [
                    delete_idx,
                ]
            )
        else:
            delete_idx = np.array(delete_idx)
        deleted_num = len(delete_idx)
        self.interp_set_Y[delete_idx] = np.empty((deleted_num, self.n))
        self.interp_set_fval[delete_idx] = np.inf
        for idx in range(delete_idx.size):
            self.interp_set_fval_eles[delete_idx[idx]] = [None for _ in self.ele_idxs]
            self.interp_set_extra_fval[delete_idx[idx]] = 0.0
        self.enable[delete_idx] = False
        self.update_x_opt()

    def clear(self) -> None:
        """
        Clear the entire interpolation set, resetting all points and associated data to
        their default values.
        """
        set_size = self.interp_set_Y.shape[0]
        self.interp_set_Y = np.empty((set_size, self.n))
        self.interp_set_fval = np.ones((set_size,)) * np.inf
        self.interp_set_fval_eles = [
            [None for _ in self.ele_idxs],
        ] * set_size
        self.interp_set_extra_fval = np.zeros(set_size)
        self.enable = np.zeros((set_size,), dtype=bool)
        self.x_opt_idx = None

    def clear_invalid_point(self) -> None:
        """
        Delete the worst points from the interpolation set to ensure the number of
        maintained points does not exceed ``self.max_size``.

        Notes
        -----
        Points are deleted based on their objective function values, with the worst
        (highest) values being removed first.
        """
        if self.set_size > self.max_size:
            to_be_deleted_idx = np.argsort(
                self.interp_set_fval,
            )[
                0 : self.set_size
            ][self.max_size - self.set_size :]
            self.delete_points(to_be_deleted_idx)

    def clear_with_only_one_left(self) -> None:
        """
        Delete all but the optimal point from the interpolation set, leaving only one point.

        Notes
        -----
        The optimal point is determined by the lowest objective function value.
        """
        if self.set_size > 1:
            to_be_deleted_idx = np.argsort(
                self.interp_set_fval,
            )[
                0 : self.set_size
            ][0 : self.set_size - 1]
            self.delete_points(to_be_deleted_idx)

    def update_point_on_idx(
        self,
        x: np.ndarray,
        idx_eles: List[int],
        fval_eles: List[float],
        fval: float,
        extra_fval: float = 0.0,
        need_update: Optional[List[bool]] = None,
    ) -> int:
        """
        Update the interpolation set with a new point, including its element function
        values and overall objective function value.

        Parameters
        ----------
        x : ndarray
            New point to be added to the interpolation set
        idx_eles : list of int
            Point indices in the elemental interpolation sets corresponding to the new point.
        fval_eles : list of float
            Element function values at the new point
        fval : float
            Overall objective function value at the new point
        extra_fval : float, default=0.0
            Extra objective function value at the new point.
            The "extra function" represents the differentiable white-box component
            of the objective.
        need_update : list of bool, optional
            List of boolean flags indicating which elemental interpolation sets should
            be updated. If None, all elemental sets are updated. Default is None.

        Returns
        -------
        int
            The index of the updated point in the overall interpolation set.
        """
        if need_update is None:
            need_update = [True for _ in self.ele_idxs]
        for ele_idx in self.ele_idxs:
            interp_set = self.ele_interp_sets[ele_idx]
            if need_update[ele_idx]:
                interp_set.update_point_on_idx(
                    self.proj_onto_ele(x, ele_idx),
                    idx_eles[ele_idx],
                    fval_eles[ele_idx],
                )
        return self.append(x, fval, fval_eles, extra_fval)[0]
