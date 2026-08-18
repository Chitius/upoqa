# Parameter sweep for `tr_radius.alpha3` x `general.center_shift_threshold`
# on the S2MPJ problem set (same 86 problems as compare_versions.py; only the
# 78 historically-comparable ones are run).
#
# Usage:
#   python profiles/param_sweep.py out.json alpha3 cst [start] [end] [budget_mult]
#
# fopt references come from /tmp/s2mpj_fopt_final.json (multi-source min over
# BFGS + bobyqa/cobyqa/newuoa + all upoqa versions), so no BFGS runs are
# needed here. All runs use seed=0 for reproducibility.

import json
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_REPO_ROOT, "upoqa", "problems", "S2MPJ"))

import upoqa  # noqa: E402
from upoqa.problems import S2MPJPSProblem  # noqa: E402

_BIG_PROB = {
    "CYCLOOCFLS", "CYCLOOCTLS", "DIXMAANB", "DIXMAANC", "DIXMAANE1",
    "DIXMAANF", "DIXMAANJ", "DIXMAANK", "DIXMAANL", "DTOC2", "DTOC3",
    "HAGER2", "HAGER3", "RAYBENDL",
}
_TOO_BIG_PROB = {"DTOC2"}

_FOPT_PATH = "/tmp/s2mpj_fopt_final.json"


def dim_map_of_at_50(name):
    if name not in _BIG_PROB:
        return 50
    return 15 if name in _TOO_BIG_PROB else 30


def main():
    out_path = sys.argv[1]
    alpha3 = float(sys.argv[2])
    cst = float(sys.argv[3])
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    end = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9
    budget_mult = float(sys.argv[6]) if len(sys.argv) > 6 else 50

    print(f"upoqa loaded from: {upoqa.__file__}", flush=True)
    print(f"alpha3={alpha3} center_shift_threshold={cst}", flush=True)
    ref = json.load(open(_FOPT_PATH))
    info = np.load(
        os.path.join(_REPO_ROOT, "profiles_old", "scalable_good_ps_problem.npy"),
        allow_pickle=True,
    )
    results = []
    for idx, entry in enumerate(info):
        if not (start <= idx < end):
            continue
        name = entry["name"]
        if name not in ref or not ref[name]["valid"]:
            continue
        fopt = ref[name]["fopt"]
        n = dim_map_of_at_50(name)
        rec = {"idx": idx, "name": name, "fopt": fopt}
        try:
            prob = S2MPJPSProblem(name, n=n)
            rec["dim"] = int(prob.dim)
            maxfev = int(max(budget_mult * prob.dim, 2000))
            rec["maxfev"] = maxfev
            t0 = time.time()
            res = upoqa.minimize(
                prob.fun, prob.x0, coords=prob.coords, weights=prob.weights,
                maxfev=maxfev, seed=0, disp=0,
                options={
                    "tr_radius.alpha3": alpha3,
                    "general.center_shift_threshold": cst,
                },
            )
            rec.update(
                fun=float(res.fun), nfev=int(res.max_nfev), nit=int(res.nit),
                success=bool(res.success), message=str(res.message)[:120],
                time=time.time() - t0,
            )
            rec["gap"] = rec["fun"] - fopt
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
