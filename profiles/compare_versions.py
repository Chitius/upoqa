# Compare two upoqa versions on the S2MPJ problem set used by the old
# profiling scripts (profiles_old, "scalable_good_ps_problem_v1": 86 problems,
# dim ~50).
#
# Usage (run twice, once per version, selecting the version via PYTHONPATH):
#   PYTHONPATH=/tmp/upoqa_baseline python profiles/compare_versions.py out_old.json
#   PYTHONPATH=<repo root>        python profiles/compare_versions.py out_new.json
#
# Optional args: out.json [start_idx] [end_idx] [budget_mult]
# Each problem is run with maxfev = max(budget_mult * dim, 2000) (default
# budget_mult=100). The reference optimum fopt is 0 for problems flagged
# has_zero_fopt, otherwise computed by BFGS with exact gradients (same
# reference for both versions).

import json
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_REPO_ROOT, "upoqa", "problems", "S2MPJ"))

import upoqa  # noqa: E402
from upoqa.problems import S2MPJPSProblem  # noqa: E402
from scipy.optimize import minimize as sp_minimize  # noqa: E402

# dim_map_of_at_50 from profiles_old/profile_utils.py
_BIG_PROB = {
    "CYCLOOCFLS", "CYCLOOCTLS", "DIXMAANB", "DIXMAANC", "DIXMAANE1",
    "DIXMAANF", "DIXMAANJ", "DIXMAANK", "DIXMAANL", "DTOC2", "DTOC3",
    "HAGER2", "HAGER3", "RAYBENDL",
}
_TOO_BIG_PROB = {"DTOC2"}


def dim_map_of_at_50(name):
    if name not in _BIG_PROB:
        return 50
    return 15 if name in _TOO_BIG_PROB else 30


def get_fopt(prob, has_zero_fopt):
    if has_zero_fopt:
        return 0.0, "zero"
    # The exact-gradient path (fgx) is broken for some auto-generated problem
    # files under numpy>=2 (2D element arrays assigned into 1D gradient slots),
    # so try exact-gradient BFGS first and fall back to finite-difference BFGS.
    try:
        res = sp_minimize(
            prob.s2mpj_prob.fx,
            prob.x0,
            method="BFGS",
            jac=lambda x: prob.s2mpj_prob.fgx(x)[1].squeeze(),
            options={"maxiter": 100000},
            tol=1e-18,
        )
        return float(res.fun), "bfgs_exact"
    except Exception:
        res = sp_minimize(
            prob.s2mpj_prob.fx,
            prob.x0,
            method="BFGS",
            options={"maxiter": 10000},
            tol=1e-12,
        )
        return float(res.fun), "bfgs_fd"


def main():
    out_path = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    budget_mult = float(sys.argv[4]) if len(sys.argv) > 4 else 100

    print(f"upoqa loaded from: {upoqa.__file__}", flush=True)
    info = np.load(
        os.path.join(_REPO_ROOT, "profiles_old", "scalable_good_ps_problem.npy"),
        allow_pickle=True,
    )
    results = []
    for idx, entry in enumerate(info):
        if not (start <= idx < end):
            continue
        name = entry["name"]
        has_zero_fopt = bool(entry["has_zero_fopt"])
        n = dim_map_of_at_50(name)
        rec = {"idx": idx, "name": name}
        try:
            prob = S2MPJPSProblem(name, n=n)
            rec["dim"] = int(prob.dim)
            rec["fopt"], rec["fopt_src"] = get_fopt(prob, has_zero_fopt)
            maxfev = int(max(budget_mult * prob.dim, 2000))
            rec["maxfev"] = maxfev
            np.random.seed(0)
            t0 = time.time()
            res = upoqa.minimize(
                prob.fun, prob.x0, coords=prob.coords, weights=prob.weights,
                maxfev=maxfev, disp=0,
            )
            rec.update(
                fun=float(res.fun), nfev=int(res.max_nfev), nit=int(res.nit),
                success=bool(res.success), message=str(res.message)[:120],
                time=time.time() - t0,
            )
            rec["gap"] = rec["fun"] - rec["fopt"]
        except Exception as e:
            rec.update(fun=None, gap=None, nfev=-1, success=False,
                       message=f"CRASH: {type(e).__name__}: {e}"[:200])
        results.append(rec)
        print(f"[{idx}] {name} dim={rec.get('dim')} gap={rec['gap']} "
              f"nfev={rec.get('nfev')} success={rec.get('success')} "
              f"{rec.get('time', 0):.1f}s {rec['message'][:40]}", flush=True)
        with open(out_path, "w") as f:  # checkpoint after every problem
            json.dump(results, f, indent=1)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
