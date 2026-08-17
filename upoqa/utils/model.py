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
Interpolation Surrogate Model
=============================

Maintain two core classes:

1. :class:`~upoqa.utils.model.QuadSurrogate`: Quadratic interpolation surrogate 
   model using the derivative-free symmetric Broyden update [1]_. Serves as the 
   element model in UPOQA.

2. :class:`~upoqa.utils.model.OverallSurrogate`: Overall surrogate model formed by 
   assembling multiple element models.

References
----------
.. [1] Michael J. D. Powell. 2004. Least Frobenius norm updating of quadratic 
    models that satisfy interpolation conditions. *Math. Program.* 100, 1 
    (May 2004), 183-215.
"""

import numpy as np
import numpy.linalg as LA
from copy import deepcopy
from .interp_set import InterpSet, OverallInterpSet
from typing import Any, Callable, Optional, Union, Tuple, List, Dict
from .params import UPOQAParameterList

__all__ = ["SurrogateLinAlgError", "QuadSurrogate", "OverallSurrogate"]


class SurrogateLinAlgError(Exception):
    """
    Exception raised for linear algebra errors in the surrogate model

    Attributes
    ----------
    ele_idx : int or None
        Index of the element function that caused the error.
        If the error is not related to a specific element, this is None.
    ele_name : Any or None
        Name of the element function that caused the error.
        If the error is not related to a specific element, this is None.
    """

    def __init__(
        self,
        message: str,
        ele_idx: Optional[int] = None,
        ele_name: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.ele_idx = ele_idx
        self.ele_name = ele_name


class QuadSurrogate:
    """
    Quadratic interpolation surrogate model using the *derivative-free symmetric
    Broyden update* [1]_. Serves as the elemental surrogate model in UPOQA.

    Parameters
    ----------
    interp_set: :class:`~upoqa.utils.interp_set.InterpSet`
        The interpolation set. It must be provided in order for the interpolation
        surrogate model to be built based on the interpolation points.
    center: ndarray, optional
        The initial center of the surrogate model. If provided, it overrides the ``n``
        argument. Otherwise, the center will be set as a zero vector.
    n : int, optional
        Problem dimension. Required if ``center`` is not provided.
        Note that either ``n`` or ``center`` must be provided.
    ref_surrogate : :class:`~upoqa.utils.model.QuadSurrogate`, optional
        A reference surrogate model to help initialize the KKT coefficients
        and other cached information.

    Attributes
    ----------
    n : int
        Problem dimension
    interp_set : :class:`~upoqa.utils.interp_set.InterpSet`
        The interpolation set used to build the surrogate model
    npt : int
        Number of interpolation points in the interpolation set
    model_center : ndarray, shape (n,)
        Center of the surrogate model
    model_cons : float
        Constant term of the model at the center
    model_grad : ndarray, shape (n,)
        Gradient of the model at the center
    model_hess_explicit : ndarray, shape (n, n)
        Explicit Hessian of the model
    model_hess_implicit : ndarray, shape (npt,)
        Implicit Hessian of the model

    References
    ----------
    .. [1] Michael J. D. Powell. 2004. Least Frobenius norm updating of quadratic models that
        satisfy interpolation conditions. *Math. Program.* 100, 1 (May 2004), 183-215.
    """

    def __init__(
        self,
        interp_set: InterpSet,
        center: Optional[np.ndarray] = None,
        n: Optional[int] = None,
        ref_surrogate: Optional["QuadSurrogate"] = None,
    ) -> None:
        if center is None:
            assert n is not None
            center = np.zeros(n)
        n = center.size
        assert (
            interp_set.npt <= (n + 1) * (n + 2) // 2
        ), "QuadSurrogate requires npt to be <= (n + 1) * (n + 2) / 2."
        assert interp_set.npt >= n + 1, "QuadSurrogate requires npt to be >= n + 1."
        self.reset(interp_set, center, ref_surrogate)

    def reset(
        self,
        interp_set: InterpSet,
        center: np.ndarray,
        ref_surrogate: Optional["QuadSurrogate"] = None,
    ) -> None:
        """
        Reset the model with the given interpolation set, center point and dimension,
        and optionally a reference model.

        Parameters
        ----------
        interp_set: :class:`~upoqa.utils.interp_set.InterpSet`
            The interpolation set. It must be provided in order for the interpolation
            surrogate model to be built based on the interpolation points.
        center: ndarray
            The initial center of the surrogate model.
        ref_surrogate : :class:`~upoqa.utils.model.QuadSurrogate`, optional
            A reference surrogate model to help initialize the KKT coefficients
            and other cached information.
        """
        n = center.size
        self.n = n
        self.interp_set = interp_set
        self.npt = self.interp_set.npt

        # surrogate model coeff
        self.model_hess_explicit = np.zeros((n, n), dtype=np.float64)
        self.model_hess_implicit = np.zeros(interp_set.npt, dtype=np.float64)
        self.model_grad = np.zeros(n, dtype=np.float64)
        self.model_cons = 0.0
        self.model_center = (
            center.reshape((n,))
            if center is not None
            else np.zeros(n, dtype=np.float64)
        )

        if ref_surrogate is not None:
            # if ref_surrogate is provided, initialize KKT coefficients to match ref_surrogate
            self._KKT_R: np.ndarray = ref_surrogate._KKT_R.copy()
            self._KKT_B: np.ndarray = ref_surrogate._KKT_B.copy()
            self.cached_Y_shift: np.ndarray = (
                ref_surrogate.cached_Y_shift.copy()
            )  # interp set points - center
            self.cached_x_anchor: np.ndarray = ref_surrogate.cached_x_anchor.copy()
            self.cached_x_anchor_idx: int = ref_surrogate.cached_x_anchor_idx
            self._negative_s_idx: int = ref_surrogate._negative_s_idx
            self._update_surrogate_coeff()
        else:
            # otherwise, initialize all KKT coefficients to zero and update them later
            self._KKT_R = np.zeros((self.npt - self.n - 1, self.npt), dtype=np.float64)
            self._KKT_B = np.zeros((self.npt + self.n, self.n), dtype=np.float64)
            self.cached_Y_shift = (
                self.interp_set.get_interp_set()[0] - self.model_center
            )
            self.cached_x_anchor = self.interp_set.get_anchor()[0].copy()
            self.cached_x_anchor_idx = self.interp_set.x_anchor_idx
            self._negative_s_idx = 0
            self._init_KKT_and_model_coeff(self.interp_set.last_init_step_size)

    def _init_KKT_and_model_coeff(self, step_size: Optional[float] = None) -> None:
        """
        Initialize the KKT and model coefficients based on the interpolation set and
        the given step size. Detailed explanations can be found around formula (2.10)
        of [1]_.

        Parameters
        ----------
        step_size : float, optional
            The step size used for initializing the KKT and model coefficients. If not
            provided, the initialization step size from the interpolation set is used.

        References
        ----------
        .. [1] Michael J. D. Powell. 2009. The BOBYQA algorithm for bound constrained
            optimization without derivatives. *Cambridge NA Report NA2009/06,
            University of Cambridge, Cambridge* 26 (2009): 26-46.
        """
        init_step_size = step_size or self.interp_set.init_step_size

        self._negative_s_idx = 0
        xpt = np.zeros((self.npt, self.n))
        for k in range(self.npt):
            if 1 <= k <= self.n:
                xpt[k, k - 1] = init_step_size
                if self.npt <= k + self.n:
                    self._KKT_B[0, k - 1] = -1.0 / init_step_size
                    self._KKT_B[k, k - 1] = 1.0 / init_step_size
                    self._KKT_B[self.npt + k - 1, k - 1] = -0.5 * init_step_size**2.0
            elif self.n < k <= 2 * self.n:
                xpt[k, k - self.n - 1] = -init_step_size
                self._KKT_B[k, k - self.n - 1] = -0.5 / xpt[k - self.n, k - self.n - 1]
                self._KKT_B[k - self.n, k - self.n - 1] = (
                    -self._KKT_B[0, k - self.n - 1] - self._KKT_B[k, k - self.n - 1]
                )
                self._KKT_R[k - self.n - 1, 0] = -np.sqrt(2.0) / (init_step_size**2.0)
                self._KKT_R[k - self.n - 1, k] = np.sqrt(0.5) / init_step_size**2.0
                self._KKT_R[k - self.n - 1, k - self.n] = (
                    -self._KKT_R[k - self.n - 1, 0] - self._KKT_R[k - self.n - 1, k]
                )
            elif k > 2 * self.n:
                shift = (k - self.n - 1) // self.n
                i = k - (1 + shift) * self.n - 1
                j = (i + shift) % self.n
                self._KKT_R[k - self.n - 1, 0] = 1.0 / init_step_size**2.0
                self._KKT_R[k - self.n - 1, k] = 1.0 / init_step_size**2.0
                self._KKT_R[k - self.n - 1, i + 1] = -1.0 / init_step_size**2.0
                self._KKT_R[k - self.n - 1, j + 1] = -1.0 / init_step_size**2.0

        self._update_surrogate_coeff()

    def inv_kkt_matrix_dot(
        self, x: np.ndarray, w_star: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute inv(W) @ x, where W is the coefficient matrix of the KKT equation generated
        by the derivative-free symmetric Broyden update [1]_, and return the vector formed
        by the first ``npt`` and last ``n`` dimensions of the result.

        In the multiplication, we can safely ignore the ``(npt + 1)``-th dimension of ``x``.
        Therefore, the input vector ``x`` only needs to include the first ``npt`` and the last
        ``n`` elements, instead of the full ``(npt + 1 + n)``-dimensional vector.

        Parameters
        ----------
        x : ndarray, shape (npt + n,)
            The vector to be dotted with the inverse of the KKT matrix.
        w_star : ndarray, shape (npt + n,), optional
            The auxiliary vector to assist in the calculation. It is formed by

                ``[0.5 * ((yi - center) @ (Y - center).T) ** 2, yi - center]``,

            where ``Y`` is of shape ``(npt, n)`` which represents the interpolation points.
            ``yi`` can be any point in the interpolation set, and is ``x_anchor`` by default.

        Returns
        -------
        ndarray, shape (npt + n,)
            The vector formed by the first ``npt`` and last ``n`` dimensions of the result.

        References
        ----------
        .. [1] Michael J. D. Powell. 2004. Least Frobenius norm updating of quadratic
            models that satisfy interpolation conditions. *Math. Program.* 100, 1
            (May 2004), 183-215.
        """
        # assert x.size == self.npt + self.n
        w_star = self._get_w() if w_star is None else w_star

        x = x.reshape((-1,))
        Zx = np.zeros((self.npt + self.n,), dtype=np.float64)
        tmp = x[: self.npt] @ self._KKT_R.T
        tmp[: self._negative_s_idx] = -tmp[: self._negative_s_idx]
        Zx[: self.npt] = tmp @ self._KKT_R
        Zx[: self.npt] += x[self.npt :] @ (self._KKT_B[: self.npt]).T
        Zx[: self.npt] -= w_star[self.npt :] @ (self._KKT_B[: self.npt]).T
        tmp = w_star[: self.npt] @ self._KKT_R.T
        tmp[: self._negative_s_idx] = -tmp[: self._negative_s_idx]
        Zx[: self.npt] -= tmp @ self._KKT_R
        Zx[self.npt :] = (x - w_star) @ self._KKT_B
        Zx[self.cached_x_anchor_idx] += 1
        return Zx

    def inv_kkt_matrix_partial_dot(self, x: np.ndarray) -> np.ndarray:
        """
        Compute inv(W) @ x, where W is the coefficient matrix of the KKT equation
        generated by the derivative-free symmetric Broyden update [1]_, and return the
        vector formed by the first ``npt`` and last ``n`` dimensions of the result.

        Unlike ``self.inv_kkt_matrix_dot(x)``, this method assumes that ``x`` is zero for
        indices ``[npt:]``, allowing for a simplified calculation.

        Parameters
        ----------
        x : ndarray, shape (npt + n,)
            The vector to be dotted with the inverse of the KKT matrix.

        Returns
        -------
        ndarray, shape (npt + n,)
            The vector formed by the first ``npt`` and last ``n`` dimensions of the result.

        References
        ----------
        .. [1] Michael J. D. Powell. 2004. Least Frobenius norm updating of quadratic
            models that satisfy interpolation conditions. *Math. Program.* 100, 1
            (May 2004), 183-215.
        """
        x = x.reshape((-1,))
        tmp = x[: self.npt] @ self._KKT_R.T
        tmp[: self._negative_s_idx] = -tmp[: self._negative_s_idx]
        return np.hstack((tmp @ self._KKT_R, x[: self.npt] @ self._KKT_B[: self.npt]))

    def _get_alpha(self, k: Optional[int] = None) -> Union[np.ndarray, float]:
        if k is None:
            return (
                LA.norm(self._KKT_R[self._negative_s_idx :], axis=0) ** 2
                - LA.norm(self._KKT_R[: self._negative_s_idx], axis=0) ** 2
            )
        else:
            tmp = np.copy(self._KKT_R[:, k])
            tmp[: self._negative_s_idx] = -tmp[: self._negative_s_idx]
            return tmp.dot(self._KKT_R[:, k])

    def _get_beta(
        self,
        w: np.ndarray,
        Zw: np.ndarray,
        x_shift: np.ndarray,
        w_star: Optional[np.ndarray] = None,
    ) -> float:
        w_star = self._get_w() if w_star is None else w_star
        return (
            0.5 * (x_shift.dot(x_shift)) ** 2
            - w[self.cached_x_anchor_idx]
            - w.dot(Zw)
            + w_star.dot(Zw)
        )

    def _get_w(self, x_shift: Optional[np.ndarray] = None) -> np.ndarray:
        x_shift = (
            self.cached_x_anchor - self.model_center
            if x_shift is None
            else x_shift.copy()
        )
        w = np.zeros(self.n + self.npt, dtype=np.float64)
        w[: self.npt] = 0.5 * ((x_shift @ self.cached_Y_shift.T) ** 2)
        w[self.npt :] = x_shift
        return w.copy()

    def get_determinant_ratio(
        self, x_new: np.ndarray, idx: Optional[int] = None
    ) -> Tuple[
        Union[np.ndarray, float],
        float,
        Union[np.ndarray, float],
        Union[np.ndarray, float],
        np.ndarray,
    ]:
        """
        Calculate key intermediate quantities (``alpha``, ``beta``, ``tau``, ``sigma``) for
        updating the inverse KKT matrix in the derivative-free symmetric Broyden
        update [1]_. These quantities are defined in [2]_, where ``sigma`` (the denominator
        in the update equation) represents the ratio of the determinant of the inverse
        KKT matrix before and after update.

        Parameters
        ----------
        x_new : ndarray, shape (n,)
            New point to be added to the interpolation set (not yet added).
        idx : int, optional
            Index of the interpolation point to be replaced by ``x_new``.
            If provided, returns scalar values at this index; otherwise returns full arrays.

        Returns
        -------
        alpha : ndarray or float
            Quantity ``alpha``. Returns ``alpha[idx]`` if ``idx`` is provided.
        beta : float
            Quantity ``beta``.
        tau : ndarray or float
            Quantity ``tau``. Returns ``tau[idx]`` if ``idx`` is provided.
        sigma : ndarray or float
            Quantity ``sigma``. Returns ``sigma[idx]`` if ``idx`` is provided.
        ndarray
            Result of ``self.inv_kkt_matrix_dot(w)``, where:
            ``w = [0.5 * ((x_new - center) @ (Y - center).T) ** 2, x_new - center]``,
            where ``Y`` is of shape ``(npt, n)`` which represents the interpolation points.

        References
        ----------
        .. [1] Michael J. D. Powell. 2004. Least Frobenius norm updating of quadratic
            models that satisfy interpolation conditions. *Math. Program.* 100, 1
            (May 2004), 183-215.

        .. [2] Michael J. D. Powell. 2004. On updating the inverse of a KKT matrix.
            *Numerical Linear Algebra and Optimization, ed. Ya-xiang Yuan, Science Press
            (Beijing)* (2004): 56-78.
        """
        x_new_shift = x_new - self.model_center
        w = self._get_w(x_new_shift)
        w_star = self._get_w()

        Zw = self.inv_kkt_matrix_dot(w, w_star)
        beta = self._get_beta(w, Zw, x_new_shift, w_star=w_star)
        alpha = self._get_alpha(idx)
        tau = Zw[idx] if idx is not None else Zw[: self.npt]
        sigma = alpha * beta + tau**2
        return alpha, beta, tau, sigma, Zw

    def update(
        self,
        cached_kkt_info: Optional[
            Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]
        ] = None,
    ) -> None:
        """
        Update the KKT and model coefficients using the method detailed in [1]_, [2]_ and [3]_.

        Parameters
        ----------
        cached_kkt_info : tuple, optional
            If not None, it should be the cached output of ``self.get_determinant_ratio(x_new)``,
            where ``x_new`` denotes the new point just added into the interpolation set.

        References
        ----------
        .. [1] Michael J. D. Powell. 2004. Least Frobenius norm updating of quadratic models
            that satisfy interpolation conditions. *Math. Program.* 100, 1 (May 2004), 183-215.

        .. [2] Michael J. D. Powell. 2004. On updating the inverse of a KKT matrix. *Numerical
            Linear Algebra and Optimization, ed. Ya-xiang Yuan, Science Press (Beijing)*
            (2004): 56-78.

        .. [3] Michael J. D. Powell. 2006. *The NEWUOA Software for Unconstrained Optimization
            without Derivatives*. Springer US, Boston, MA, 255-297.
        """
        last_update_idx = self.interp_set.last_update_idx
        x_new = self.interp_set.get_interp_set(last_update_idx)[0]
        deleted_x = self.interp_set.just_deleted_point.copy()
        self._update_KKT_coeff(x_new, last_update_idx, cached_kkt_info)
        self._update_surrogate_coeff(deleted_x, last_update_idx)
        self.cached_x_anchor = self.interp_set.get_anchor()[0].copy()
        self.cached_x_anchor_idx = self.interp_set.x_anchor_idx

    def _update_KKT_coeff(
        self,
        x_new: np.ndarray,
        idx: int,
        cached_kkt_info: Optional[
            Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]
        ] = None,
    ) -> None:
        """
        Update the KKT coefficients using the method detailed in [1]_, [2]_ and [3]_.

        Parameters
        ----------
        x_new : ndarray, shape (n,)
            The new point which was just added into the interpolation set.
        idx : int
            The index in the interpolation set where ``x_new`` was added.
        cached_kkt_info : tuple, optional
            If not None, it should be the cached output of ``self.get_determinant_ratio(x_new)``,
            where ``x_new`` denotes the new point just added into the interpolation set.

        Notes
        -----
        The implementation here largely references that in COBYQA [4]_.

        References
        ----------
        .. [1] Michael J. D. Powell. 2004. Least Frobenius norm updating of quadratic
            models that satisfy interpolation conditions. *Math. Program.* 100, 1
            (May 2004), 183-215.

        .. [2] Michael J. D. Powell. 2004. On updating the inverse of a KKT matrix.
            *Numerical Linear Algebra and Optimization, ed. Ya-xiang Yuan, Science Press
            (Beijing)* (2004): 56-78.

        .. [3] Michael J. D. Powell. 2006. *The NEWUOA Software for Unconstrained Optimization
            without Derivatives*. Springer US, Boston, MA, 255-297.

        .. [4] Tom M. Ragonneau. *Model-Based Derivative-Free Optimization Methods
            and Software*. PhD thesis, Department of Applied Mathematics, The Hong
            Kong Polytechnic University, Hong Kong, China, 2022. URL:
            https://theses.lib.polyu.edu.hk/handle/200/12294.
        """

        def householder_transform(mat: np.ndarray, idx: int) -> None:
            """
            Perform an in-place Householder transformation on ``mat`` to make the ``idx``-th
            column align with the first standard basis vector ``e_1``.
            """
            vec = mat[:, idx].reshape((-1,)).copy()
            vec_norm = LA.norm(vec)
            if np.abs(vec_norm) > 0.0:
                vec[0] += vec_norm if vec[0] >= 0 else -vec_norm
                mat -= 2 * np.outer(vec, vec @ mat) / vec.dot(vec)

        # Perform Householder transformation on _KKT_R
        if self._negative_s_idx > 1:
            householder_transform(self._KKT_R[: self._negative_s_idx], idx)
        if self._negative_s_idx < self._KKT_R.shape[0] - 1:
            householder_transform(self._KKT_R[self._negative_s_idx :], idx)

        # Prepare alpha, beta, tau, sigma, and Zw
        if cached_kkt_info is None:
            alpha, beta, tau, sigma, Zw = self.get_determinant_ratio(x_new, idx)
        else:
            alpha, beta, tau, sigma, Zw = cached_kkt_info
            alpha, tau, sigma = alpha[idx], tau[idx], sigma[idx]
        b_max = np.max(np.abs(self._KKT_B), initial=1.0)
        z_max = np.max(np.abs(self._KKT_R), initial=1.0)
        if abs(sigma) < np.finfo(float).tiny * max(b_max, z_max):
            # The denominator of the updating formula is too small to safely
            # divide the coefficients of the KKT matrix of interpolation.
            # Theoretically, the value of abs(sigma) is always positive, and
            # becomes small only for ill-conditioned problems.
            raise ZeroDivisionError("The denominator of the updating formula is zero")

        # Update _KKT_B first
        _KKT_B_on_idx = self._KKT_B[idx].copy()
        if self._negative_s_idx == 0:
            Ze_ell_R = self._KKT_R[0] * self._KKT_R[0, idx]
        else:
            # With negative signs on the first _negative_s_idx rows of the S
            # matrix, the Householder pre-rotation leaves nonzeros only in
            # rows 0 and _negative_s_idx of column idx, and the W vector of
            # the updating formula reads (NEWUOA update.f, lines 38-44):
            # W = -Z(KNEW,1)*Z(:,1) + Z(KNEW,IDZ)*Z(:,IDZ).
            Ze_ell_R = -self._KKT_R[0] * self._KKT_R[0, idx]
            if self._negative_s_idx < self._KKT_R.shape[0]:
                # The second term vanishes when all rows carry a negative
                # sign (S = -I), which is out of NEWUOA's IDZ range but
                # reachable here via the `sigma < 0` branch below.
                Ze_ell_R = Ze_ell_R + (
                    self._KKT_R[self._negative_s_idx]
                    * self._KKT_R[self._negative_s_idx, idx]
                )
        Ze_ell = np.hstack((Ze_ell_R, _KKT_B_on_idx))
        Zw_minus_e_ell = Zw.copy()
        Zw_minus_e_ell[idx] -= 1
        temp_vec1 = (alpha * Zw[self.npt :] - tau * _KKT_B_on_idx) / sigma
        temp_vec2 = (tau * Zw[self.npt :] + beta * _KKT_B_on_idx) / sigma
        self._KKT_B[: self.npt] += np.outer(
            Zw_minus_e_ell[: self.npt], temp_vec1
        ) - np.outer(Ze_ell[: self.npt], temp_vec2)
        self._KKT_B[self.npt :] += np.outer(Zw_minus_e_ell[self.npt :], temp_vec1)
        self._KKT_B[self.npt :] -= np.outer(Ze_ell[self.npt :], temp_vec2)

        # Then update _KKT_R
        jdz = (
            self._negative_s_idx if self._negative_s_idx < self.npt - self.n - 1 else 0
        )
        if jdz == 0:
            self._KKT_R[0] = (
                tau * self._KKT_R[0] - self._KKT_R[0, idx] * Zw_minus_e_ell[: self.npt]
            ) / (np.abs(sigma) ** 0.5)
            if sigma < 0.0:
                self._negative_s_idx = 1
        else:
            scala = self._KKT_R[0, idx] if jdz == 0 else -self._KKT_R[0, idx]
            scalb = 0.0 if jdz == 0 else self._KKT_R[jdz, idx]
            kdz = jdz if beta >= 0.0 else 0
            jdz -= kdz
            tempb = self._KKT_R[jdz, idx] * tau / sigma
            # NEWUOA update.f, lines 74-78: SCALA and the JA-column update
            # both use ZMAT(KNEW, JA), i.e. row kdz here (not row jdz).
            tmp1 = 1.0 / np.sqrt(abs(beta) * self._KKT_R[kdz, idx] ** 2.0 + tau**2.0)
            self._KKT_R[kdz] = (
                tau * self._KKT_R[kdz]
                - self._KKT_R[kdz, idx] * Zw_minus_e_ell[: self.npt]
            )
            self._KKT_R[kdz] *= tmp1
            self._KKT_R[jdz] -= (self._KKT_R[jdz, idx] * beta / sigma) * (
                scala * self._KKT_R[0] + scalb * self._KKT_R[self._negative_s_idx]
            ) + tempb * Zw_minus_e_ell[: self.npt]
            self._KKT_R[jdz] *= tmp1 * np.sqrt(abs(sigma))
            if sigma <= 0.0:
                if beta < 0.0:
                    # If sigma <= 0 and beta < 0, the positive-definite property is broken.
                    self._negative_s_idx += 1
                else:
                    self._negative_s_idx -= 1
                    self._KKT_R[[0, self._negative_s_idx]] = self._KKT_R[
                        [self._negative_s_idx, 0]
                    ]

    def shift_center_to(self, x: np.ndarray) -> None:
        """
        Shift the model center to ``x``, then update the KKT and model coefficients.

        Parameters
        ----------
        x : ndarray, shape (n,)
            The new model center.

        Notes
        -----
        The computational cost is O(npt^3) with BLAS-3 operations.
        """
        x = x.copy()
        step = x - self.model_center

        # Prepare intermediate variables
        Phi_T = self.cached_Y_shift - 0.5 * step
        Psi_T = (step @ Phi_T.T).reshape((-1, 1)) * Phi_T + 0.25 * step.dot(step) * step
        R_T_Psi_T = self._KKT_R @ Psi_T
        Xi_Psi_T = self._KKT_B[: self.npt].T @ Psi_T
        S_R_T_Psi_T = np.copy(R_T_Psi_T)
        S_R_T_Psi_T[: self._negative_s_idx] = -S_R_T_Psi_T[: self._negative_s_idx]
        Omega_Psi_T = self._KKT_R.T @ S_R_T_Psi_T
        Psi_Omega_Psi_T = R_T_Psi_T.T @ S_R_T_Psi_T

        # Update _KKT_B
        # Note: _KKT_R remains invariant when shifting the model center.
        self._KKT_B[: self.npt] += Omega_Psi_T
        self._KKT_B[self.npt :] += Xi_Psi_T + Xi_Psi_T.T + Psi_Omega_Psi_T

        # Update surrogate coefficients
        center_shift = x - self.model_center
        Hess_center_shift = self.hess_operator(center_shift)
        self.model_cons = (
            self.model_cons
            + self.model_grad.dot(center_shift)
            + 0.5 * center_shift @ Hess_center_shift
        )
        self.model_grad = Hess_center_shift + self.model_grad
        self.model_center = x
        temp = (Phi_T.T * self.model_hess_implicit) @ np.tile(step, (self.npt, 1))
        self.model_hess_explicit += temp + temp.T
        self.cached_Y_shift = self.cached_Y_shift - step

    def _update_surrogate_coeff(
        self, deleted_x: Optional[np.ndarray] = None, idx: Optional[int] = None
    ) -> None:
        """
        Assuming the KKT coefficients have been updated, this function updates the model
        coefficients based on the updated KKT coefficients.

        Parameters
        ----------
        deleted_x : ndarray, shape (n,), optional
            The old interpolation point that was replaced with a new point at the index
            ``idx``. Both ``deleted_x`` and ``idx`` should be None if the model is being
            updated for the first time after being established.
        idx : int, optional
            The index where the old interpolation point ``deleted_x`` was replaced with a
            new point. Both ``deleted_x`` and ``idx`` should be None if the model is being
            updated for the first time after being established.
        """
        Y_new, f_Y_new = self.interp_set.get_interp_set()
        f_Y_shift = f_Y_new - self.fun_eval(Y_new)
        model_lambda, model_g = self.solve_interpolation_KKT(f_Y_shift=f_Y_shift)
        model_hess_explicit_p1 = self.model_hess_explicit.copy()
        model_hess_implicit_p1 = self.model_hess_implicit + model_lambda
        if idx is not None:
            # assert deleted_x is not None
            deleted_x = deleted_x - self.model_center
            model_hess_explicit_p1 += self.model_hess_implicit[idx] * np.outer(
                deleted_x, deleted_x
            )
            model_hess_implicit_p1[idx] = model_lambda[idx]
        model_grad_p1 = self.grad_eval(self.model_center) + model_g
        self.model_hess_explicit = model_hess_explicit_p1
        self.model_hess_implicit = model_hess_implicit_p1
        self.model_grad = model_grad_p1
        self.cached_Y_shift = Y_new - self.model_center
        self.model_cons += f_Y_new[self.interp_set.x_anchor_idx] - self.fun_eval(
            self.interp_set.x_anchor
        )

    def solve_interpolation_KKT(
        self, f_Y_shift=np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given the interpolation set, solves the first-order KKT optimality conditions
        for the derivative-free symmetric Broyden update subproblem and returns the solution:
        the Lagrange multipliers and the gradient update term.

        Parameters
        ----------
        f_Y_shift : ndarray, shape (npt,)
            Represents ``fun(Y) - surrogate(Y)``, where ``Y`` denotes the interpolation points.

        Returns
        -------
        lambda : ndarray, shape (npt,)
            The Lagrange multipliers.
        g : ndarray, shape (n,)
            The gradient update term.
        """
        coeff = self.inv_kkt_matrix_partial_dot(f_Y_shift)
        return coeff[0 : self.npt], coeff[self.npt :]

    def reset_surrogate_coeff(self) -> None:
        """
        Reset all surrogate model coefficients to zero values.
        """
        self.model_hess_explicit = np.zeros((self.n, self.n), dtype=np.float64)
        self.model_hess_implicit = np.zeros(self.interp_set.npt, dtype=np.float64)
        self.model_grad = np.zeros(self.n, dtype=np.float64)
        self.model_cons = 0.0

    def reinit(self) -> None:
        """
        Reinitialize the surrogate model by first resetting and then updating the coefficients.
        """
        self.reset_surrogate_coeff()
        self._update_surrogate_coeff()

    def rebuild(
        self, init_step_size: Optional[float] = None, max_trials: int = 5
    ) -> None:
        r"""
        Rebuild the inverse-KKT factors (``_KKT_R``/``_KKT_B``) from scratch for
        the *current* interpolation set, and recompute the model coefficients.

        The closed-form initialization used by ``_init_KKT_and_model_coeff`` is
        only valid for the initial CFD stencil. To rebuild the factors for an
        arbitrary point set, the interpolation set is temporarily reinitialized
        as a CFD stencil around the model center, and each stencil point is then
        morphed into the corresponding actual interpolation point via the
        standard rank-1 update (whose correctness only depends on the points,
        not on the function values). No objective function evaluation is
        performed. If a morphing update is near-singular, the target point is
        jittered by a tiny amount and retried; every jittered point is restored
        to its exact position in a final pass, once the set is non-degenerate.

        Parameters
        ----------
        init_step_size : float, optional
            Step size of the temporary CFD stencil. Default: the median distance
            from the model center to the current interpolation points.
        max_trials : int, default=5
            Maximum number of attempts (including jittered retries) per point.

        Raises
        ------
        SurrogateLinAlgError
            If a point cannot be morphed even after jittered retries. In that
            case the interpolation set and the factors may be left in an
            intermediate (stencil-morphed) state, and the caller should treat
            the model as unusable.
        """
        iset = self.interp_set
        npt, n = iset.npt, self.n
        Y_target = iset.interp_set_Y.copy()
        f_target = iset.interp_set_fval.copy()
        x_opt_idx, x_anchor_idx = iset.x_opt_idx, iset.x_anchor_idx
        center = self.model_center.copy()

        if init_step_size is None:
            dists = LA.norm(Y_target - center, axis=1)
            positive_dists = dists[dists > 0.0]
            init_step_size = (
                float(np.median(positive_dists)) if positive_dists.size else 1.0
            )
        step = max(init_step_size, np.finfo(float).eps)
        rng = np.random.default_rng(0)  # deterministic, for reproducibility

        # Reinitialize as a fresh CFD stencil with closed-form factors. The
        # function values play no role in the factor updates; they are placed
        # positionally so that the final state is exactly consistent.
        iset.init_interp_set_Y(center, step)
        iset.set_interp_set_fval(f_target)
        self.reset_surrogate_coeff()
        self._KKT_R = np.zeros((npt - n - 1, npt), dtype=np.float64)
        self._KKT_B = np.zeros((npt + n, n), dtype=np.float64)
        self.cached_Y_shift = iset.interp_set_Y - center
        self.cached_x_anchor = iset.get_anchor()[0].copy()
        self.cached_x_anchor_idx = iset.x_anchor_idx
        self._negative_s_idx = 0
        self._init_KKT_and_model_coeff(step)

        # Morph each stencil point into its target point.
        jittered = []
        for k in range(npt):
            target = Y_target[k]
            if np.array_equal(target, iset.interp_set_Y[k]):
                continue
            scale = 0.0
            for trial in range(max_trials):
                x_try = (
                    target
                    if trial == 0
                    else target + scale * rng.standard_normal(n)
                )
                iset.update_point_on_idx(x_try, k, f_target[k])
                try:
                    self.update()
                    break
                except (LA.LinAlgError, ZeroDivisionError):
                    scale = 1e-8 * step * 10.0**trial
            else:
                raise SurrogateLinAlgError(
                    f"Failed to rebuild the KKT factors (point {k})."
                )
            if scale > 0.0:
                jittered.append(k)

        # Restore the exact positions of the jittered points.
        for k in jittered:
            iset.update_point_on_idx(Y_target[k], k, f_target[k])
            self.update()

        # Restore the bookkeeping and recompute the coefficients for the
        # final state.
        iset.x_opt_idx, iset.x_anchor_idx = x_opt_idx, x_anchor_idx
        self.cached_x_anchor = iset.get_anchor()[0].copy()
        self.cached_x_anchor_idx = x_anchor_idx
        self._update_surrogate_coeff()

    def alt_grad_eval(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate gradient of a newly-instantiated surrogate model at point ``x``.

        Conceptually equivalent to:
        
        1. Creating a new surrogate model with current interpolation set
        2. Updating its coefficients from zero
        3. Returning its gradient at ``x``

        Avoids actual model reconstruction by leveraging KKT solution.

        Parameters
        ----------
        x : ndarray, shape (n,)
            Evaluation point for gradient calculation.

        Returns
        -------
        g : ndarray
            Gradient vector of the conceptual surrogate model at ``x``.
        """
        _, f_Y = self.interp_set.get_interp_set()
        model_lambda, model_g = self.solve_interpolation_KKT(f_Y_shift=f_Y)
        return (
            ((x - self.model_center) @ self.cached_Y_shift.T) * model_lambda
        ) @ self.cached_Y_shift + model_g

    def fun_eval(self, X: np.ndarray) -> Union[np.ndarray, float]:
        """
        Evaluate the surrogate model at given point(s).

        Parameters
        ----------
        x : ndarray, shape (n,) or ndarray, shape (n_samples, n)
            Evaluation point(s). Can be a single point (1D) or batch of points (2D).

        Returns
        -------
        fval : float or ndarray, shape (n_samples,)
            Model value(s). Scalar for single point input, array of shape (n_samples,)
            for multiple points.
        """
        X_shift = X - self.model_center
        if X_shift.ndim == 1:
            return (
                X_shift.dot(0.5 * self.hess_operator(X_shift) + self.model_grad)
                + self.model_cons
            )
        else:
            value = (
                0.5 * np.diag(self.hess_operator(X_shift) @ X_shift.T)
                + X_shift @ self.model_grad
                + self.model_cons
            )
            return (
                value.item()
                if isinstance(value, np.ndarray) and value.size == 1
                else value
            )

    def grad_eval(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient vector(s) of the surrogate model at given point(s).

        Parameters
        ----------
        x : ndarray, shape (n,) or ndarray, shape (n_samples, n)
            Evaluation point(s). Can be a single point (1D) or batch of points (2D).

        Returns
        -------
        g : ndarray, shape (n, ) or ndarray, shape (n_samples, n)
            Gradient vector(s). Shape (n,) for single point, (n_samples, n) for
            multiple points.
        """
        return self.hess_operator(X - self.model_center) + self.model_grad

    @property
    def model_hess(self) -> np.ndarray:
        """Full Hessian matrix of the surrogate model"""
        return (
            self.cached_Y_shift.T * self.model_hess_implicit
        ) @ self.cached_Y_shift + self.model_hess_explicit

    def hess_eval(self, *arg, **kwargs) -> np.ndarray:
        """
        Return the full Hessian matrix of the surrogate model.

        Returns
        -------
        H : ndarray, shape (n, n)
            Hessian matrix (same as ``model_hess`` property).
        """
        return self.model_hess

    def hess_operator(self, V: np.ndarray) -> np.ndarray:
        """
        Compute Hessian-vector product(s) for the surrogate model.

        Parameters
        ----------
        V : ndarray, shape (n, ) or ndarray, shape (n_samples, n)
            Input vector(s). Single vector (1D) or batch of vectors (2D).

        Returns
        -------
        HV : ndarray, shape (n, ) or ndarray, shape (n_samples, n)
            Hessian-vector product(s). Shape (n,) for single input vector,
            (n_samples, n) for multiple vectors.
        """
        if V.ndim == 1:
            V = V[np.newaxis, :]
        result = (
            (V @ self.cached_Y_shift.T) * self.model_hess_implicit
        ) @ self.cached_Y_shift + V @ self.model_hess_explicit
        return result if V.shape[0] > 1 else result[0]


class OverallSurrogate:
    r"""
    Overall surrogate model formed by assembling multiple elemental surrogate models.

    Represent a full-system surrogate as a sum of component models, each defined on
    a subspace of the full design space.

    Attributes
    ----------
    n : int
        Problem dimension
    interp_set : :class:`~upoqa.utils.interp_set.OverallInterpSet`
        Interpolation set for the overall surrogate model
    proj_onto_ele : callable
        Projection function mapping full-space points to element subspaces:

        .. code-block:: python

            (full_point: ndarray, element_name: Any) -> element_point: ndarray

        For example, ``proj_onto_ele([1.0, 2.0], "f_1")`` yields ``[1.0,]``, where
        ``"f_1"`` denotes a function that depends only on the first variable of ``x``.
    coords : list of 1D arrays
        Variable indices for each element's subspace (parallel to ``ele_names``).
    ele_models : list
        List of elemental surrogate models
    xforms : list
        Transformations :math:`h_1, \ldots, h_q` applied to element function outputs
    weights : list
        Weights :math:`w_1, \ldots, w_q` for each element in the model
    extra_fun : list
        The white-box component :math:`f_0` of the objective function, which is a list of
        length 3 containing callables to evaluate the function value, gradient, and Hessian.
    params : :class:`~upoqa.utils.params.UPOQAParameterList`
        Parameter list used for the algorithm
    ele_names : list
        List containing the names of all elements in order
    model_center : ndarray, shape (n,)
        Center of the surrogate model
    """

    def __init__(
        self,
        n: int,
        interp_set: OverallInterpSet,
        proj_onto_ele: Callable[[np.ndarray, Any], np.ndarray],
        coords: List[Union[List, np.ndarray]],
        ele_models: List[QuadSurrogate],
        extra_fun: List[Callable[[np.ndarray], float]],
        ele_names: List[Any],
        params: UPOQAParameterList,
    ) -> None:
        self.n = n
        self._zero = np.zeros((n,))
        self.interp_set = interp_set
        self.ele_models = ele_models
        self.ele_names = ele_names
        self.ele_idxs = list(range(len(ele_names)))

        assert callable(proj_onto_ele)
        self.proj_onto_ele = proj_onto_ele

        self.coords = coords
        self.model_center = self.interp_set.x_opt

        self.extra_fun = extra_fun
        self.params = params
        self.xforms = None
        self.weights: Optional[np.ndarray] = None

    def __getitem__(self, ele_idx: int) -> QuadSurrogate:
        return self.ele_models[ele_idx]

    def set_weights(self, weights: np.ndarray) -> None:
        """
        Set the weights for the elements.

        Parameters
        ----------
        - weights : ndarray
            The weights to assign to each element.
        """
        self.weights = deepcopy(weights)

    def set_xforms(
        self,
        xforms: List[Optional[List[Callable[[np.ndarray], Union[np.ndarray, float]]]]],
    ) -> None:
        """
        Set transformation functions for the elements.

        Parameters
        ----------
        xforms : list
            Transformation functions to apply to each element.
        """
        self.xforms = deepcopy(xforms)

    def update_weights(
        self,
        weights: Union[
            Dict[Any, Union[int, float]], List[Union[int, float]], np.ndarray
        ],
    ) -> None:
        """
        Update the weights for the elements, handling both list and dictionary inputs.

        If ``weights`` is a dictionary, it maps element names to weight values.
        Missing elements retain their current weights.

        This method should only be called by 
        :meth:`~upoqa.utils.manager.UPOQAManager.update_weights()`.

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
            self.set_weights(weights_list)
        else:
            self.set_weights(weights)

    def update_xforms(
        self,
        xforms: List[List[Callable[[np.ndarray], Union[np.ndarray, float]]]],
    ) -> None:
        """
        Update transformation functions with input validation.
        This method should only be called by 
        :meth:`~upoqa.utils.manager.UPOQAManager.update_xforms()`.

        Parameters
        ----------
        xforms : list
            List of updated transformation lists for all elements. each transformation
            list should contain exactly 3 callables: ``[function, gradient, hessian]``.
        """
        if isinstance(xforms, dict):
            xforms_list = []
            for ele_idx in self.ele_idxs:
                ele_name = self.ele_names[ele_idx]
                if ele_name in xforms:
                    if xforms[ele_name] is not None:
                        assert isinstance(
                            xforms[ele_name], list
                        ), f"xforms[{ele_name}] should be a list, but got {type(xforms[ele_name])}"
                        assert (
                            len(xforms[ele_name]) == 3
                        ), f"xforms[{ele_name}] should be a list of length 3, but got length {len(xforms[ele_name])}"
                        assert all(
                            [isinstance(op, Callable) for op in xforms[ele_name]]
                        ), f"xforms[{ele_name}] should contain only callable functions."
                else:
                    xforms[ele_name] = self.xforms[ele_idx]
                xforms_list.append(xforms[ele_name])
            self.set_xforms(xforms_list)
        else:
            self.set_xforms(xforms)

    def shift_center_to(self, x: np.ndarray) -> None:
        """
        Shift the model center to a new point ``x``.
        Note that this does not change the centers of the element models.

        Parameters
        ----------
        x : ndarray
            The new center point for the overall surrogate model.
        """
        self.model_center = x

    def update(
        self,
        want_update: List[bool],
        cached_KKT_info_eles: List[
            Optional[Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]]
        ] = None,
    ) -> None:
        """
        Update the element models based on the provided update flags and cached KKT
        information.

        Parameters
        ----------
        want_update : list of bool
            A list of flags indicating whether each element model should be updated.
        cached_KKT_info_eles : list of tuples, optional
            List of cached outputs of ``ele_surrogate.get_determinant_ratio(x_ele_new)``
            for each element model ``ele_surrogate``, where ``x_ele_new`` denotes the new
            point just added into the corresponding elemental interpolation set.
        """
        for ele_idx in self.ele_idxs:
            ele_surrogate = self.ele_models[ele_idx]
            if want_update[ele_idx]:
                try:
                    ele_surrogate.update(
                        cached_kkt_info=(
                            cached_KKT_info_eles[ele_idx]
                            if cached_KKT_info_eles
                            else None
                        )
                    )
                except (LA.LinAlgError, ZeroDivisionError) as e:
                    # The interpolation set of this element has already been
                    # modified. Rebuild its KKT factors from the current point
                    # set, so that the model stays consistent with the set
                    # instead of leaving the two permanently desynced.
                    try:
                        ele_surrogate.rebuild()
                    except SurrogateLinAlgError:
                        raise SurrogateLinAlgError(
                            str(e), ele_name=self.ele_names[ele_idx], ele_idx=ele_idx
                        )

    def fun_eval(self, x: np.ndarray) -> float:
        """
        Evaluate the overall surrogate model at given point.

        Parameters
        ----------
        x : ndarray, shape (n,)
            Evaluation point.

        Returns
        -------
        fval : float
            The model value of the overall surrogate model at the point ``x``.
        """
        value = float(self.extra_fun[0](x))
        for ele_idx in self.ele_idxs:
            ele_surrogate = self.ele_models[ele_idx]
            x_ele = self.proj_onto_ele(x, ele_idx)
            if self.xforms[ele_idx] is not None:
                value_ele = self.weights[ele_idx] * self.xforms[ele_idx][0](
                    ele_surrogate.fun_eval(x_ele)
                )
            else:
                value_ele = self.weights[ele_idx] * ele_surrogate.fun_eval(x_ele)
            if self.params("debug.check_nan_fval") and np.any(np.isnan(value_ele)):
                raise ValueError(
                    f"(element {self.ele_names[ele_idx]}) NaN encountered in the return value"
                    " of the surrogate function. "
                    "This may be caused by some numerical error or matrix singularity. "
                )
            value += value_ele
        return value

    def grad_eval(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient vector of the overall surrogate model at given point.

        Parameters
        ----------
        x : ndarray, shape (n,)
            Evaluation point.

        Returns
        -------
        g : ndarray, shape (n, )
            The gradient vector of the overall surrogate model at the point ``x``.
        """
        grad = np.array(self.extra_fun[1](x), dtype=np.float64)
        for ele_idx in self.ele_idxs:
            ele_surrogate = self.ele_models[ele_idx]
            x_ele = self.proj_onto_ele(x, ele_idx)
            if self.xforms[ele_idx] is not None:
                grad_ele = (
                    self.weights[ele_idx]
                    * self.xforms[ele_idx][1](ele_surrogate.fun_eval(x_ele))
                    * ele_surrogate.grad_eval(x_ele)
                )
            else:
                grad_ele = self.weights[ele_idx] * ele_surrogate.grad_eval(x_ele)
            if self.params("debug.check_nan_fval") and np.any(np.isnan(grad_ele)):
                raise ValueError(
                    f"(element {self.ele_names[ele_idx]}) NaN encountered in the return value"
                    " of the surrogate gradient. "
                    "This may be caused by some numerical error or matrix singularity. "
                )
            grad[self.coords[ele_idx]] += grad_ele
        return grad

    def hess_eval(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the full Hessian matrix of the overall surrogate model at given point.

        Parameters
        ----------
        x : ndarray, shape (n,)
            Evaluation point.

        Returns
        -------
        H : ndarray, shape (n, n)
            The Hessian matrix of the overall surrogate model at the point ``x``.
        """
        hess = np.array(self.extra_fun[2](x), dtype=np.float64)
        for ele_idx in self.ele_idxs:
            ele_surrogate = self.ele_models[ele_idx]
            model_dim_idx = np.asarray(self.coords[ele_idx]).reshape((1, -1))
            x_ele = self.proj_onto_ele(x, ele_idx)
            if self.xforms[ele_idx] is not None:
                ele_model_fval = ele_surrogate.fun_eval(x_ele)
                ele_model_grad = ele_surrogate.grad_eval(x_ele)
                xform_grad = self.xforms[ele_idx][1](ele_model_fval)
                xform_hess = self.xforms[ele_idx][2](ele_model_fval)
                hess_ele = self.weights[ele_idx] * (
                    xform_hess * np.outer(ele_model_grad, ele_model_grad)
                    + xform_grad * ele_surrogate.hess_eval(x_ele)
                )
            else:
                hess_ele = self.weights[ele_idx] * ele_surrogate.hess_eval(x_ele)
            if self.params("debug.check_nan_fval") and np.any(np.isnan(hess_ele)):
                raise ValueError(
                    f"(element {self.ele_names[ele_idx]}) NaN encountered in the return value"
                    " of the surrogate hessian. "
                    "This may be caused by some numerical error or matrix singularity. "
                )
            hess[model_dim_idx.T, model_dim_idx] += hess_ele
        return hess

    def hess_operator(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute Hessian-vector product for the overall surrogate model at given point.

        Parameters
        ----------
        x : ndarray, shape (n,)
            The point at which to evaluate the Hessian-vector product.
        v : ndarray, shape (n, )
            The vector to multiply with the Hessian matrix.

        Returns
        -------
        Hv : ndarray, shape (n, )
            The result of the Hessian-vector product.
        """
        result = v @ np.array(self.extra_fun[2](x), dtype=np.float64)
        for ele_idx in self.ele_idxs:
            ele_surrogate = self.ele_models[ele_idx]
            v_ele = self.proj_onto_ele(v, ele_idx)
            if self.xforms[ele_idx] is not None:
                x_ele = self.proj_onto_ele(x, ele_idx)
                ele_model_fval = ele_surrogate.fun_eval(x_ele)
                ele_model_grad = ele_surrogate.grad_eval(x_ele)
                xform_grad = self.xforms[ele_idx][1](ele_model_fval)
                xform_hess = self.xforms[ele_idx][2](ele_model_fval)
                hess_operator_ele = self.weights[ele_idx] * (
                    xform_hess
                    * np.outer(ele_model_grad, v_ele.dot(ele_model_grad)).squeeze()
                    + xform_grad * ele_surrogate.hess_operator(v_ele)
                )
            else:
                hess_operator_ele = self.weights[ele_idx] * ele_surrogate.hess_operator(
                    v_ele
                )
            if self.params("debug.check_nan_fval") and np.any(
                np.isnan(hess_operator_ele)
            ):
                raise ValueError(
                    f"(element {self.ele_names[ele_idx]}) NaN encountered in the return "
                    "value of the surrogate hessian operator. "
                    "This may be caused by some numerical error or matrix singularity. "
                )
            result[self.coords[ele_idx]] += hess_operator_ele
        return result
