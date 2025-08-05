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

import sys

sys.path.append("../upoqa/problems/S2MPJ/")
import importlib
from .problem_base import PSProblem, default_noise_wrapper
import numpy as np
from . import S2MPJ
from .S2MPJ import s2mpjlib

S2MPJ_dir = "upoqa.problems.S2MPJ.python_problems"


def import_s2mpj_problem_class(name: str):
    try:
        return getattr(importlib.import_module('%s.%s' % (S2MPJ_dir, name)), name)
    except Exception as e:
        raise RuntimeError(f"Could not find problem {name}, error: {e}")


class S2MPJPSProblem(PSProblem):
    def __init__(self, name: str, n: int, noise_wrapper: callable = default_noise_wrapper):
        prob = import_s2mpj_problem_class(name)(n)
        self.s2mpj_prob = prob
        if hasattr(prob, "x0"):
            self.x0 = np.array(prob.x0).squeeze()
        dim = prob.n
        super().__init__(elements=dict(), coords=dict(), dim=dim, noise_wrapper=noise_wrapper)
        self.update_meta_info(dict(name=name))

        if hasattr(prob, "A"):
            sA1, sA2 = prob.A.shape
            self._has_A = True
        else:
            self._has_A = False

        prob.getglobs()
        self._has_grftype = hasattr(prob, "grftype")
        self._has_grelt = hasattr(prob, "grelt")
        self._has_gconst = hasattr(prob, 'gconst')
        self._has_A = hasattr(prob, 'A')
        self._has_gscale = hasattr(prob, 'gscale')
        self._has_grelw = hasattr(prob, "grelw")

        glist = prob.objgrps
        for iig in range(len(glist)):
            ig = int(glist[iig])
            if self._has_A and ig < sA1:
                coord = np.where(prob.A[ig, :sA2].toarray().squeeze() != 0.0)[0].tolist()
            else:
                coord = []

            if self._has_grelt and ig < len(prob.grelt) and not prob.grelt[ig] is None:
                for iiel in range(len(prob.grelt[ig])):  #  loop on elements
                    iel = prob.grelt[ig][iiel]  #  the element's index
                    coord.extend(
                        [iv for iv in prob.elvar[iel]]
                    )  #  the elemental variable's indeces

            coord = list(set(coord))
            coord.sort()
            coord = np.array(coord)
            if coord.size > 0:
                self.append(iig, self.sub_fun_generator(iig), coord)

        #  Evaluate the quadratic term, if any.
        if hasattr(prob, "H"):

            def prob_quadratic_part(x):
                Htimesx = self.H.dot(x)
                return 0.5 * x.T.dot(Htimesx)

            self.append(
                "quadratic_part", prob_quadratic_part, np.array([int(x) for x in range(prob.n)])
            )

        # self check
        if hasattr(self, "x0") and self.x0 is not None:
            fval_raw_x0 = prob.fx(self.x0)
            fval_x0 = self.fun_eval(self.x0)
            assert abs(fval_x0 - fval_raw_x0) <= np.finfo(float).eps * max(
                1.0, abs(fval_raw_x0), abs(fval_x0)
            ) * len(self.coords)

    @property
    def objlower(self):
        return self.s2mpj_prob.objlower if hasattr(self.s2mpj_prob, "objlower") else -np.inf

    def is_unconstrained(self):
        cx = self.s2mpj_prob.cx(np.ones(self.dim))
        return (
            (cx is None)
            and (np.all(self.s2mpj_prob.xlower == -np.inf))
            and np.all(self.s2mpj_prob.xupper == np.inf)
        )

    def __repr__(self):
        return f"S2MPJPSProblem ({self.name}) with dim = {self.dim}, average elemental dim = {self.avg_ele_dim:.3f}"

    def sub_fun_generator(self, iig: int):
        def _eval_group_iig(x: np.ndarray):
            prob = self.s2mpj_prob
            # prob.getglobs()
            x_extend = np.zeros((self.dim))
            x_extend[self.coords[iig]] = x
            x = x_extend

            glist = prob.objgrps
            fx = 0.0
            fin = 0
            gsc = 1.0

            if self._has_A:
                sA1, sA2 = prob.A.shape

            ig = int(glist[iig])

            #  Find the group's scaling.
            if (
                self._has_gscale
                and ig < len(prob.gscale)
                and prob.gscale[ig] is not None
                and abs(prob.gscale[ig]) > 1e-15
            ):
                gsc = prob.gscale[ig]

            #  Evaluate the linear term, if any.
            if self._has_gconst and ig < len(prob.gconst) and prob.gconst[ig] is not None:
                fin = float(-prob.gconst[ig])

            if self._has_A and ig < sA1:
                fin += float(prob.A[ig].dot(x[:sA2]))

            if self._has_grelt and ig < len(prob.grelt) and prob.grelt[ig] is not None:
                for iiel in range(len(prob.grelt[ig])):  #  loop on elements
                    iel = prob.grelt[ig][iiel]  #  the element's index
                    efname = prob.elftype[iel]  #  the element's ftype
                    xiel = x[prob.elvar[iel].astype(int)]  #  the elemental variable's values

                    wiel = 1.0
                    if self._has_grelw and ig <= len(prob.grelw) and prob.grelw[ig] is not None:
                        wiel = prob.grelw[ig][iiel]

                    # Only the value is requested.
                    fin += wiel * getattr(prob, efname)(prob, 1, xiel, iel)

            #  Evaluate the group function.
            #  1) the non-TRIVIAL case
            if self._has_grftype and ig < len(prob.grftype) and prob.grftype[ig] is not None:
                egname = prob.grftype[ig]
            else:
                egname = "TRIVIAL"
            if egname != 'TRIVIAL' and egname is not None:
                fx += getattr(prob, egname)(prob, 1, fin, ig) / gsc
            #  2) the TRIVIAL case: the group function is the identity
            else:
                fx += fin / gsc

            return float(fx)

        return _eval_group_iig

    def grad_eval(self, x: np.ndarray):
        return self.s2mpj_prob.fgx(x)[1]

    def hess_eval(self, x: np.ndarray):
        return self.s2mpj_prob.fgHx(x)[2]
