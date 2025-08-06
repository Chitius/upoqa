# This file contains modified code from SciPy (https://github.com/scipy/scipy)
# which is licensed under the BSD 3-Clause License:
#
# Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
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
Optimization Result
===================

Adapted from ``scipy.optimize.OptimizeResult`` with minor modifications.
"""

import numpy as np
from typing import Iterable


class OptimizeResult(dict):
    """
    Represent the optimization result.

    Attributes
    ----------
    message : str
        Description of the cause of the termination.
    success : bool
        Whether the optimization procedure terminated successfully.
    exception : Exception, optional
        Exception raised during the optimization that caused the termination, if any.
    traceback : str, optional
        Traceback of the exception, if any.
    fun : float
        Value of objective function evaluated at the solution.
    funs : dict or list
        Values of the element functions evaluated at the solution.

        The type matches the input ``fun``:

        - If ``fun`` was a dictionary, return a dictionary mapping element names to
          their function values.
        - If ``fun`` was a list, return a list of function values in the same order
          as the input list.
    extra_fun : float
        Value of the extra function at the solution.

        The "extra function" represents the differentiable white-box component of the objective.
    x : ndarray
        The solution of the optimization.
    jac, hess : ndarray
        Values of objective function's Jacobian and its Hessian at ``x``.
        The Hessian is an approximation.
    nit : int
        Number of iterations performed by the optimizer
    nfev : dict or list
        Number of evaluations of element functions.

        The type matches the input ``fun``:

        - If ``fun`` was a dictionary, return a dictionary mapping element names to
          their numbers of function evaluations.
        - If ``fun`` was a list, return a list of evaluation numbers in the same order
          as the input list.
    max_nfev : int
        Maximum number of evaluations of element functions.
    avg_nfev : float
        Average number of evaluations of element functions.
    nrun : int
        Number of runs (when enabling restarts).
    manager : :class:`~upoqa.utils.manager.UPOQAManager`, optional
        Algorithm manager (debugging only).
        Included only when ``debug=True`` or ``return_internals=True``.
    interp_set : :class:`~upoqa.utils.interp_set.OverallInterpSet`, optional
        Interpolation point set (debugging only).
        Included only when ``debug=True`` or ``return_internals=True``.
    model : :class:`~upoqa.utils.model.OverallSurrogate`, optional
        Surrogate model (debugging only).
        Included only when ``debug=True`` or ``return_internals=True``.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        order_keys = [
            "message",
            "success",
            "exception",
            "traceback",
            "fun",
            "funs",
            "extra_fun",
            "x",
            "jac",
            "hess",
            "nit",
            "nfev",
            "max_nfev",
            "avg_nfev",
            "nrun",
            "manager",
            "interp_set",
            "model",
        ]
        order_keys = getattr(self, "_order_keys", order_keys)
        # 'slack', 'con' are redundant with residuals
        # 'crossover_nit' is probably not interesting to most users
        omit_keys = {"slack", "con", "crossover_nit", "_order_keys"}

        def key(item):
            try:
                return order_keys.index(item[0].lower())
            except ValueError:  # item not in list
                return np.inf

        def omit_redundant(items):
            for item in items:
                if item[0] in omit_keys:
                    continue
                yield item

        def item_sorter(d):
            return sorted(omit_redundant(d.items()), key=key)

        if self.keys():
            return _dict_formatter(self, sorter=item_sorter)
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def _dict_formatter(d, n=0, mplus=1, sorter=None):
    """
    Pretty printer for dictionaries

    ``n`` keeps track of the starting indentation;
    lines are indented by this much after a line break.
    ``mplus`` is additional left padding applied to keys
    """
    if isinstance(d, dict) and isinstance(next(iter(d)), Iterable):
        m = max(map(len, list(d.keys()))) + mplus  # width to print keys
        s = "\n".join(
            [
                k.rjust(m)
                + ": "  # right justified, width m
                + _indenter(_dict_formatter(v, m + n + 2, 0, sorter), m + 2)
                for k, v in sorter(d)
            ]
        )  # +2 for ': '
    else:
        # By default, NumPy arrays print with linewidth=76. `n` is
        # the indent at which a line begins printing, so it is subtracted
        # from the default to avoid exceeding 76 characters total.
        # `edgeitems` is the number of elements to include before and after
        # ellipses when arrays are not shown in full.
        # `threshold` is the maximum number of elements for which an
        # array is shown in full.
        # These values tend to work well for use with OptimizeResult.
        with np.printoptions(
            linewidth=76 - n,
            edgeitems=2,
            threshold=12,
            formatter={"float_kind": _float_formatter_10},
        ):
            s = str(d)
    return s


def _indenter(s, n=0):
    """
    Ensures that lines after the first are indented by the specified amount
    """
    split = s.split("\n")
    indent = " " * n
    return ("\n" + indent).join(split)


def _float_formatter_10(x):
    """
    Returns a string representation of a float with exactly ten characters
    """
    if np.isposinf(x):
        return "       inf"
    elif np.isneginf(x):
        return "      -inf"
    elif np.isnan(x):
        return "       nan"
    return np.format_float_scientific(x, precision=3, pad_left=2, unique=False)
