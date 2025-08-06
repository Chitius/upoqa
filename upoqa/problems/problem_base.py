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

import numpy as np
from copy import deepcopy
from typing import Dict, List, Optional, Union, Any, Iterable, Callable, Literal
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage
import IPython
import traceback
from contextlib import contextmanager
from functools import reduce


class ProblemPlot:
    def __init__(self) -> None:
        self._eval_count = 0
        self.history_fun = (
            None  # example: [[1, 2, 3, 4, ...], [102, 56, 21, 1.9, 0.5, ...]]
        )
        self.history_plot_fig = None
        self.history_plot_ax = None
        self.record_history_fun_dict = True
        self.ax_suffix = ""
        self.y_shift = None
        self.default_plt_number = 23333
        self.plot_fig_shape = (14, 8)

    @property
    def eval_count(self) -> int:
        return self._eval_count

    def enable_history_fun_dict_record(self, flag: bool = True):
        self.record_history_fun_dict = flag

    def set_figure_size(self, new_shape: tuple = (10, 6)):
        """(14, 8) by default"""
        self.plot_fig_shape = new_shape

    def clear_fig(self) -> None:
        self.history_plot_ax = None
        self.history_plot_fig = None

    def clear(self) -> None:
        self._eval_count = 0
        self.history_fun = None

    def _update_history_fun(
        self, value: Union[int, float, np.float64, list, np.ndarray]
    ) -> None:
        if type(value) in [int, float, np.float64]:
            self.history_fun = (
                np.array([np.ceil(self._eval_count), value]).reshape(2, -1)
                if self.history_fun is None
                else np.hstack(
                    (
                        self.history_fun,
                        np.array([np.ceil(self._eval_count), value]).reshape(2, -1),
                    )
                )
            )
        elif type(value) in [list, np.ndarray]:
            self.history_fun = (
                np.vstack(
                    (
                        np.ones_like(value.squeeze()) * np.ceil(self._eval_count),
                        value.squeeze(),
                    )
                ).reshape(2, -1)
                if self.history_fun is None
                else np.hstack(
                    (
                        self.history_fun,
                        np.vstack(
                            (
                                np.ones_like(value.squeeze())
                                * np.ceil(self._eval_count),
                                value.squeeze(),
                            )
                        ).reshape(2, -1),
                    )
                )
            )

    def save_history(
        self,
        path: str,
        info: dict = dict(),
        name: str = None,
    ) -> bool:
        try:
            saved_info = dict()
            k_min = np.argmin(self.history_fun[-1])
            saved_info = {
                "optimize": {
                    "Optimal Objective Value in History": self.history_fun[-1, k_min],
                    "Number of Evaluations": self._eval_count,
                    "Theoretical Optimal Objective Value": (
                        -self.y_shift if self.y_shift is not None else 0.0
                    ),
                    "History of Objective Function Values": self.history_fun,
                },
                "meta_info": {},
            }
            if info:
                if "meta_info" in info:
                    saved_info["meta_info"].update(info.get("meta_info"))
                if "optimize" in info:
                    saved_info["optimize"].update(info.get("optimize"))
            if name:
                if not path.endswith("/"):
                    path = path + "/"
                path = path + name
            np.save(path, saved_info)
            return True
        except Exception as e:
            print(
                f"Fail to save optimization history information! Error: {e}\n\n{traceback.format_exc()}"
            )
            return False

    def load_history(self, path: str, name=None) -> Optional[dict]:
        try:
            if name:
                if not path.endswith("/"):
                    path = path + "/"
                path = path + name
            problem_optimization_info = np.load(path, allow_pickle=True).item()
            self._eval_count = problem_optimization_info.get("Number of Evaluations", 0)
            self.y_shift = -problem_optimization_info.get(
                "Theoretical Optimal Objective Value", 0.0
            )
            self.history_fun = problem_optimization_info.get(
                "History of Objective Function Values", None
            )
            return problem_optimization_info
        except Exception as e:
            print(
                f"Fail to load optimization history information! Error: {e}\n\n{traceback.format_exc()}"
            )
            return None

    def _get_disp_fun_history_data(
        self,
        x_shift: Union[int, float] = 0.0,
        upper_truncation: Union[int, float] = 1e8,
    ):
        plot_data = self.history_fun[-1].squeeze()
        if self.y_shift is not None:
            plot_data = (plot_data + self.y_shift) / abs(self.y_shift)
        return (self.history_fun[0] + x_shift), np.minimum(plot_data, upper_truncation)

    def disp_fun_history(
        self,
        title_suffix: str = "",
        clear: bool = False,
        label: str = "Unassigned",
        fmt: str = "-",
        alpha: float = 1.0,
        x_shift: Union[int, float] = 0.0,
        show_fig: bool = True,
    ) -> Line2D:
        plt.figure(self.default_plt_number)
        if clear:
            self.history_plot_ax = None
            self.history_plot_fig = None
        if self.history_plot_fig is None or self.history_plot_ax is None:
            plt.style.use("default")
            self.history_plot_fig, self.history_plot_ax = plt.subplots()
            self.history_plot_fig.set_size_inches(*self.plot_fig_shape)
            self.history_plot_ax.set_xlabel("Number of Evaluations")
            if self.y_shift is not None:
                self.history_plot_ax.set_ylabel(
                    f"Objective Relative Error (Optimal Value = {-self.y_shift:.3f})"
                )
            else:
                self.history_plot_ax.set_ylabel("Objective Value")

            suffix = " ".join([title_suffix.strip(), self.ax_suffix.strip()])
            if len(suffix) > 0:
                suffix = "\n" + suffix

            if self.y_shift is None:
                plt.title("Objective Value vs Number of Evaluations" + suffix)
            else:
                # if y_shift is set, then we are plotting relative error
                plt.title("Objective Relative Error vs Number of Evaluations" + suffix)

            self.history_plot_ax.grid(True)

        self.history_plot_ax.set_yscale("log")
        eval_data, plot_data = self._get_disp_fun_history_data(x_shift)
        (line,) = self.history_plot_ax.plot(
            eval_data,
            plot_data,
            fmt,
            label=label,
        )

        line.set_alpha(alpha)
        self.history_plot_ax.legend()
        if show_fig:
            IPython.display.display(self.history_plot_fig)
        plt.close(self.history_plot_fig)  # prevent displaying twice in jupyter notebook
        return line


def default_noise_wrapper(x: np.ndarray, *args, **kwargs) -> np.ndarray:
    return x


class PSProblem(ProblemPlot):
    def __init__(
        self,
        elements: Dict[
            Any, Callable[[np.ndarray], Union[np.ndarray, list, float]]
        ] = dict(),
        coords: Dict[Any, Union[List, np.ndarray]] = dict(),
        weights: Dict[Any, Union[int, float]] = dict(),
        xforms: Dict[Any, List[callable]] = dict(),
        dim: int = None,
        noise_wrapper: callable = default_noise_wrapper,
        nfev_mode: Literal["wst", "avg"] = "wst",
        # "wst": nfev = max(nfev of all elements), "avg": nfev = avg(nfev of all elements)
    ) -> None:
        super().__init__()
        self._eval_count = {"wst": 0, "avg": 0}  # override father's _eval_count
        self.elements = elements
        self.coords = coords
        self.noise_wrapper = noise_wrapper or default_noise_wrapper
        self.eval_count_dict = dict()
        self.history_fun_dict = dict()
        self.xforms = xforms if xforms else dict()
        self.default_xform = None
        self.weights = weights if weights else dict()
        for key in self.elements.keys():
            self.weights[key] = self.weights.get(key, 1.0)
            self.xforms[key] = self.xforms.get(key, self.default_xform)
        if dim is None:
            union_coord = reduce(np.union1d, self.coords.values())
            dim = len(union_coord)
        self.dim = dim
        self.sol_info = {"xopt": None, "fopt": None}
        self.meta_info = dict()
        self._debug_model_incre_nfev = None
        self.set_nfev_mode(nfev_mode)

    def __str__(self) -> str:
        if "name" in self.meta_info:
            msg = f"The {self.name} Problem"
        else:
            msg = "A Partially-Separable Problem"
        msg += f" with {len(self.elements)} Elements and {self.dim} Dimensions."
        return msg

    @property
    def eval_count(self) -> int:
        return self._eval_count[self.nfev_mode]

    def clear(self) -> None:
        super().clear()
        self._eval_count = {"wst": 0, "avg": 0}
        self.eval_count_dict = dict()
        self.history_fun_dict = dict()

    def set_nfev_mode(self, mode: Literal["wst", "avg"]):
        mode = mode.lower()
        assert mode in ["wst", "avg"]
        self.nfev_mode = mode

    def update_meta_info(self, info: dict) -> None:
        self.meta_info.update(info)

    def update_sol_info(self, xopt: np.ndarray = None, fopt: float = None) -> None:
        if xopt is not None:
            self.sol_info["xopt"] = xopt
        if fopt is not None:
            self.sol_info["fopt"] = fopt

    def set_noise_wrapper(self, noise_wrapper: callable) -> None:
        self.noise_wrapper = noise_wrapper

    @property
    def xopt(self) -> str:
        if "xopt" not in self.sol_info or self.sol_info["xopt"] is None:
            return None
        else:
            return self.sol_info["xopt"]

    @property
    def fopt(self) -> str:
        if "fopt" not in self.sol_info or self.sol_info["fopt"] is None:
            return None
        else:
            return self.sol_info["fopt"]

    @property
    def name(self) -> str:
        if "name" in self.meta_info:
            return self.meta_info["name"]
        else:
            return self.__class__.__name__

    def inc_count(self, count: int = 1, eval_ele_name: Any = None) -> None:
        if eval_ele_name is None:
            for mod in self._eval_count:
                self._eval_count[mod] += count
            for ele_name in self.ele_names:
                self.inc_count(count, ele_name)
        else:
            if eval_ele_name in self.eval_count_dict:
                self.eval_count_dict[eval_ele_name] += count
            else:
                self.eval_count_dict[eval_ele_name] = count
            # set self.eval_count = max(self.eval_count_dict)
            self._eval_count = {
                "wst": int(max(list(self.eval_count_dict.values()))),
                "avg": float(np.mean(list(self.eval_count_dict.values()))),
            }

    def __getitem__(self, ele_name: Any) -> callable:
        return self.elements[ele_name]

    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self) -> Iterable:
        return iter(self.elements.keys())

    def get_local(self, x: np.ndarray, ele_name: Any) -> np.ndarray:
        return x[self.coords[ele_name]] if x.ndim == 1 else x[:, self.coords[ele_name]]

    @property
    def avg_ele_dim(self) -> int:
        return np.mean([len(c) for c in self.coords.values()])

    @property
    def ele_names(self) -> List[Any]:
        return list(self.elements.keys())

    def set_elements(
        self, elements: Dict[Any, callable], coords: Dict[Any, Union[List, np.ndarray]]
    ) -> None:
        assert isinstance(elements, dict)
        assert isinstance(coords, dict)
        self.elements = elements
        self.coords = coords

    def append(
        self,
        ele_name: Any,
        ele: callable,
        coord: Union[List, np.ndarray],
        weight: Optional[float] = None,
    ) -> None:
        self.elements[ele_name] = ele
        self.coords[ele_name] = coord
        if isinstance(weight, float):
            self.weights[ele_name] = weight

    @property
    def fun(self) -> Dict[Any, callable]:
        def _sub_fun(ele_name: Any) -> callable:
            return lambda x: self.ele_eval(x, ele_name)

        return {ele_name: _sub_fun(ele_name) for ele_name in self.elements}

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.fun_eval(*args, **kwds)

    def ele_eval(self, x: np.ndarray, ele_name: Any) -> float:
        assert ele_name in self.ele_names, "Invalid elemental name."
        x = np.atleast_1d(np.asarray(x).squeeze())
        if x.size == len(self.coords[ele_name]):
            res = self.noise_wrapper(self.elements[ele_name](x), ele_name)
        elif x.size == self.dim:
            res = self.noise_wrapper(
                self.elements[ele_name](x[self.coords[ele_name]]), ele_name
            )
        else:
            raise ValueError("Input dimension does not match the problem dimension.")
        self.inc_count(1, ele_name)
        self._update_history_fun_dict({ele_name: res})
        return res

    def fun_eval(
        self, x: np.ndarray, ele_name: Any = None, incre_nfev: bool = True
    ) -> float:
        if self._debug_model_incre_nfev is not None:
            incre_nfev = bool(self._debug_model_incre_nfev)  # override incre_nfev

        if ele_name is None:
            x = np.atleast_1d(np.asarray(x).squeeze())
            assert (
                x.size == self.dim
            ), "Input dimension does not match the problem dimension."
            res_nf, res = 0.0, 0.0
            for ele_name, ele in self.elements.items():
                xform = self.xforms.get(ele_name, self.default_xform)
                ele_val = ele(x[self.coords[ele_name]])
                if xform is not None:
                    res_nf += self.weights.get(ele_name, 1) * xform[0](ele_val)
                    if self.noise_wrapper is None:
                        res += self.weights.get(ele_name, 1) * xform[0](ele_val)
                    else:
                        res += self.weights.get(ele_name, 1) * xform[0](
                            self.noise_wrapper(ele_val, ele_name)
                        )
                else:
                    res_nf += self.weights.get(ele_name, 1) * ele_val
                    if self.noise_wrapper is None:
                        res += self.weights.get(ele_name, 1) * ele_val
                    else:
                        res += self.weights.get(ele_name, 1) * self.noise_wrapper(
                            ele_val, ele_name
                        )

            if incre_nfev:
                self.inc_count(1)
                self._update_history_fun(res_nf)

            return res
        else:
            assert ele_name in self.ele_names, "Invalid elemental name."
            x = np.atleast_1d(np.asarray(x).squeeze())
            xform = self.xforms.get(ele_name, self.default_xform)

            if x.size == len(self.coords[ele_name]):
                ele_val = self.elements[ele_name](x)
            elif x.size == self.dim:
                ele_val = self.elements[ele_name](x[self.coords[ele_name]])
            else:
                raise ValueError(
                    "Input dimension does not match the problem dimension."
                )

            if xform is not None:
                res_nf = self.weights.get(ele_name, 1) * xform[0](ele_val)
                if self.noise_wrapper is None:
                    res = res_nf
                else:
                    res = self.weights.get(ele_name, 1) * xform[0](
                        self.noise_wrapper(ele_val, ele_name)
                    )
            else:
                res_nf = self.weights.get(ele_name, 1) * ele_val
                if self.noise_wrapper is None:
                    res = res_nf
                else:
                    res = self.weights.get(ele_name, 1) * self.noise_wrapper(
                        ele_val, ele_name
                    )

            if incre_nfev:
                self.inc_count(1, ele_name)
                if self.record_history_fun_dict:
                    self._update_history_fun_dict(
                        {ele_name: res_nf}
                    )  # record noise-free value

            return res  # return noisy value

    def spy(self, x: Optional[np.ndarray] = None) -> AxesImage:
        if hasattr(self, "hess_eval"):
            if x is None:
                x = (
                    self.x0
                    if hasattr(self, "x0") and self.x0 is not None
                    else np.zeros(self.dim)
                )
            return plt.spy(self.hess_eval(x).toarray())
        else:
            raise NotImplementedError("spy() method is not available for this problem.")

    def _update_history_fun_dict(
        self, values: Dict[Any, Union[int, float, np.float64]]
    ) -> None:
        for ele_name, value in values.items():
            # save [[element function evaluation count], [obj wst evaluation count], [obj avg evaluation count], [element function value]]
            self.history_fun_dict[ele_name] = (
                np.array(
                    [
                        np.ceil(self.eval_count_dict[ele_name]),
                        np.ceil(self._eval_count["wst"]),
                        np.ceil(self._eval_count["avg"]),
                        value,
                    ]
                ).reshape(4, -1)
                if ele_name not in self.history_fun_dict
                else np.hstack(
                    (
                        self.history_fun_dict[ele_name],
                        np.array(
                            [
                                np.ceil(self.eval_count_dict[ele_name]),
                                np.ceil(self._eval_count["wst"]),
                                np.ceil(self._eval_count["avg"]),
                                value,
                            ]
                        ).reshape(4, -1),
                    )
                )
            )

    def _update_history_fun(
        self, value: Union[int, float, np.float64, list, np.ndarray]
    ) -> None:
        if type(value) in [int, float, np.float64]:
            self.history_fun = (
                np.array(
                    [
                        np.ceil(self._eval_count["wst"]),
                        np.ceil(self._eval_count["avg"]),
                        value,
                    ]
                ).reshape(3, -1)
                if self.history_fun is None
                else np.hstack(
                    (
                        self.history_fun,
                        np.array(
                            [
                                np.ceil(self._eval_count["wst"]),
                                np.ceil(self._eval_count["avg"]),
                                value,
                            ]
                        ).reshape(3, -1),
                    )
                )
            )
        elif type(value) in [list, np.ndarray]:
            self.history_fun = (
                np.vstack(
                    (
                        np.ones_like(value.squeeze())
                        * np.ceil(self._eval_count["wst"]),
                        np.ones_like(value.squeeze())
                        * np.ceil(self._eval_count["avg"]),
                        value.squeeze(),
                    )
                ).reshape(3, -1)
                if self.history_fun is None
                else np.hstack(
                    (
                        self.history_fun,
                        np.vstack(
                            (
                                np.ones_like(value.squeeze())
                                * np.ceil(self._eval_count["wst"]),
                                np.ones_like(value.squeeze())
                                * np.ceil(self._eval_count["avg"]),
                                value.squeeze(),
                            )
                        ).reshape(3, -1),
                    )
                )
            )

    def save_history(self, path: str, info: dict = dict(), name: str = None) -> bool:
        try:
            saved_info = dict()
            k_min = np.argmin(self.history_fun[-1])
            saved_info = {
                "optimize": {
                    "Optimal Objective Value in History": self.history_fun[-1, k_min],
                    "Number of Elemental Evaluations": (
                        self.eval_count_dict if self.history_fun_dict else dict()
                    ),
                    "Number of Evaluations": self._eval_count,  # set to be max(self.eval_count_dict.values())
                    "History of Element Function Values": self.history_fun_dict,
                    "History of Objective Function Values": self.history_fun,
                    "Theoretical Optimal Objective Value": (
                        -self.y_shift if self.y_shift is not None else 0.0
                    ),
                },
                "meta_info": {},
            }
            if info:
                if "meta_info" in info:
                    saved_info["meta_info"].update(info.get("meta_info"))
                if "optimize" in info:
                    saved_info["optimize"].update(info.get("optimize"))
            if name:
                if not path.endswith("/"):
                    path = path + "/"
                path = path + name
            np.save(path, saved_info)
            return True
        except Exception as e:
            print(
                f"Fail to save optimization history information! Error: {e}\n\n{traceback.format_exc()}"
            )
            return False

    def load_history(self, path: str, name: str = None) -> Optional[dict]:
        try:
            if name:
                if not path.endswith("/"):
                    path = path + "/"
                path = path + name
            problem_optimization_info = np.load(path, allow_pickle=True).item()
            self._eval_count = problem_optimization_info["optimize"].get(
                "Number of Evaluations", 0
            )
            self.eval_count_dict = problem_optimization_info["optimize"].get(
                "Number of Elemental Evaluations", dict()
            )
            self.y_shift = -problem_optimization_info["optimize"].get(
                "Theoretical Optimal Objective Value", 0.0
            )
            self.history_fun = problem_optimization_info["optimize"].get(
                "History of Objective Function Values", None
            )
            self.history_fun_dict = problem_optimization_info["optimize"].get(
                "History of Element Function Values", dict()
            )
            return problem_optimization_info
        except Exception as e:
            print(
                f"Fail to load optimization history information! Error: {e}\n\n{traceback.format_exc()}"
            )
            return None

    @contextmanager
    def debug_mode(self, incre_nfev: bool = False, noisy: bool = False):
        """
        example:

        >>> with prob.debug_mode(incre_nfev = False, noisy = False):
                print(prob.fun_eval(prob.x0))

        The result will be the noise-free function value, and the evaluation count will remain unchanged.
        """
        if not incre_nfev:
            self._debug_model_incre_nfev = False
        if not noisy:
            cached_noise_wrapper = self.noise_wrapper
            self.noise_wrapper = None

        yield

        if not incre_nfev:
            self._debug_model_incre_nfev = None
        if not noisy:
            self.noise_wrapper = cached_noise_wrapper

    def _get_disp_fun_history_data(
        self,
        x_shift: Union[int, float] = 0.0,
        upper_truncation: Union[int, float] = 1e8,
    ):
        plot_data = self.history_fun[-1].squeeze()
        if self.y_shift is not None:
            plot_data = (plot_data + self.y_shift) / abs(self.y_shift)
        eval_data = (
            self.history_fun[0] if self.nfev_mode == "wst" else self.history_fun[1]
        )
        return (eval_data + x_shift), np.minimum(plot_data, upper_truncation)
