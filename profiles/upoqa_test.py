import matplotlib.pyplot as plt
import traceback
from utils import (
    problem_set,
    ProblemSet,
    prob_idx_2_prob_info_and_source,
    get_noise_free_ps_problem,
)
from solvers import REGISTERED_SOLVERS

plt.style.use('default')

# choose the problem set
# NOTE: MAKE SURE YOU'VE CLONED THE SUBMODULE upoqa/problems/S2MPJ BEFORE TEST
problem_set.load_problem_set("s2mpj_prob_set_v1")

solver_label = "upoqa"
# label can be changed to test other solvers in REGISTERED_SOLVERS, available options are
# "upoqa", "upoqa (single)", "pybobyqa", "l-bfgs-b (ffd)", "l-bfgs-b (cfd)", "imfil", "spsa",
# "bobyqa", "newuoa", "cobyla", "cma-es", "cobyqa", "gsls", "umda"

for prob_idx in problem_set.valid_problem_idx():
    # create problem
    prob_name = prob_idx_2_prob_info_and_source(prob_idx)[0]["name"]
    n = ProblemSet.PROBLEM_SETS[problem_set.enabled_problem_set]['param_map'](prob_name)
    prob = get_noise_free_ps_problem(
        problem_idx=prob_idx,
        params={"n": n},
    )

    # the max number of function evaluations
    maxfev = max(int(1000 * prob.dim), 10000)

    print("# " + "-" * 60)
    print("# ", end="")
    print(prob)
    print("")

    # optimize. the result will be saved in ./results_scalable_good_ps_problem_v1/
    try:
        res, _ = REGISTERED_SOLVERS[solver_label](
            prob,
            maxfev=maxfev,
            # identifier = "upoqa (ver=1.0.0)",
            disp=1,
            show_fig=0,
            save_fig=1,
            save_log=1,
        )
    except Exception as e:
        print(str(e) + ", " + traceback.format_exc())
