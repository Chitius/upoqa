# Benchmark script for validating stability fixes (#1 soft_restart desync,
# #3 clear_with_only_one_left inversion, #8 stale old_fval_eles).
#
# Run from the repository root:
#     python profiles/bench_fix_validation.py <output.json> [ABC]
#
# Part A (clean, heterogeneously-scaled Rosenbrock chain): regression check —
#   per-element radius adaptation (bug #8) should not hurt clean performance.
# Part B (noisy quad chain, noise_level=1): exercises the noise-restart path
#   (bugs #1, #3). Metric is the TRUE objective at the returned x.
# Part C (separable Rastrigin, seek_global_minimum=True): exercises the
#   global-search restart path (bugs #1, #3). Metrics: final fun and nfev.

import json
import sys
import time

import numpy as np

import upoqa


def make_quad_chain(seed, n=4, r=6, overlap=1, biquad_reg=0.0):
    rng = np.random.default_rng(seed)
    dim = (n - overlap) * (r - 1) + n
    center = rng.standard_normal(dim)
    fun, coords = {}, {}
    blocks = []
    for i in range(r):
        c = list(range((n - overlap) * i, (n - overlap) * i + n))
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        H = Q @ np.diag(np.linspace(0.1, 10, n)) @ Q.T
        ce = center[c]
        gamma = rng.standard_normal(n) / np.sqrt(n)
        blocks.append((c, H, ce, gamma))

        def make_ele(H, ce, gamma):
            def ele(x):
                q = float((x - ce) @ H @ (x - ce))
                if biquad_reg > 0:
                    q += biquad_reg * float(np.sum(np.power(x - gamma, 4)))
                return q

            return ele

        fun[f"blk{i}"] = make_ele(H, ce, gamma)
        coords[f"blk{i}"] = c

    def true_f(x):
        return sum(float((x[c] - ce) @ H @ (x[c] - ce)) +
                   (biquad_reg * float(np.sum(np.power(x[c] - gamma, 4)))
                    if biquad_reg > 0 else 0.0)
                   for c, H, ce, gamma in blocks)

    return fun, coords, dim, true_f


def make_scaled_ros_chain(dim, smax):
    # element i: 10^u_i * (100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2), u log-spaced
    fun, coords = {}, {}
    scales = np.logspace(-smax, smax, dim - 1)
    for i in range(dim - 1):
        s = scales[i]
        fun[f"ros{i}"] = lambda x, s=s: s * (
            100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
        )
        coords[f"ros{i}"] = [i, i + 1]
    return fun, coords


def make_rastrigin(dim):
    # separable Rastrigin: f_i(x_i) = x_i^2 - 10 cos(2 pi x_i) + 10, fopt=0 at x=0
    fun = {
        f"r{i}": (lambda x: float(x[0] ** 2 - 10 * np.cos(2 * np.pi * x[0]) + 10))
        for i in range(dim)
    }
    coords = {f"r{i}": [i] for i in range(dim)}
    return fun, coords


def run_one(fun, x0, coords, seed, **kwargs):
    np.random.seed(seed)  # solver-internal randomness (soft_restart etc.)
    t0 = time.time()
    try:
        res = upoqa.minimize(fun, x0, coords=coords, disp=0, **kwargs)
        return {
            "fun": float(res.fun),
            "x": res.x,
            "nfev": int(res.max_nfev),
            "nit": int(res.nit),
            "success": bool(res.success),
            "message": str(res.message),
            "time": time.time() - t0,
        }
    except Exception as e:  # crash = worst outcome
        return {
            "fun": np.inf,
            "x": None,
            "nfev": -1,
            "nit": -1,
            "success": False,
            "message": f"CRASH: {type(e).__name__}: {e}",
            "time": time.time() - t0,
        }


def main(out_path, parts="ABC"):
    print(f"upoqa loaded from: {upoqa.__file__}", flush=True)
    results = {"A_clean_ros": [], "B_noisy_quad": [], "C_global_rastrigin": []}

    # ---------- Part A: clean heterogeneously-scaled Rosenbrock chain ----------
    if "A" in parts:
        dim = 30
        fun, coords = make_scaled_ros_chain(dim, smax=3.0)
        for seed in range(4):
            r = run_one(fun, np.zeros(dim), coords, seed,
                        maxfev=2000, radius_final=1e-8)
            r["seed"] = seed
            results["A_clean_ros"].append(r)
            print(f"[A] seed={seed} fun={r['fun']:.4e} nfev={r['nfev']} "
                  f"success={r['success']} {r['time']:.1f}s", flush=True)

    # ---------- Part B: noisy quad chain, noise_level=1 ----------
    if "B" in parts:
        sigma = 1e-3
        for seed in range(10):
            fun, coords, dim, true_f = make_quad_chain(seed)
            noise_rng = np.random.default_rng(10_000 + seed)
            noisy_fun = {
                k: (lambda f: lambda x: f(x) + sigma * noise_rng.standard_normal())(f)
                for k, f in fun.items()
            }
            r = run_one(noisy_fun, np.zeros(dim), coords, seed,
                        maxfev=1500, noise_level=1)
            r["seed"] = seed
            r["true_fun"] = true_f(r["x"]) if r["x"] is not None else np.inf
            results["B_noisy_quad"].append(r)
            print(f"[B] seed={seed} true_fun={r['true_fun']:.4e} nfev={r['nfev']} "
                  f"nit={r['nit']} success={r['success']} {r['time']:.1f}s "
                  f"msg={r['message'][:50]}", flush=True)

    # ---------- Part C: separable Rastrigin with global restarts ----------
    if "C" in parts:
        dim = 10
        fun, coords = make_rastrigin(dim)
        for seed in range(8):
            rng = np.random.default_rng(seed)
            x0 = rng.uniform(2.0, 5.0, dim)
            r = run_one(fun, x0, coords, seed,
                        maxfev=2000, seek_global_minimum=True, radius_init=1.0)
            r["seed"] = seed
            results["C_global_rastrigin"].append(r)
            print(f"[C] seed={seed} fun={r['fun']:.4e} nfev={r['nfev']} "
                  f"success={r['success']} {r['time']:.1f}s "
                  f"msg={r['message'][:50]}", flush=True)

    for runs in results.values():
        for r in runs:
            r.pop("x", None)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench_result.json",
         sys.argv[2] if len(sys.argv) > 2 else "ABC")
