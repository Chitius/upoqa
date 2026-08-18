import sys

sys.path.append("..")
import numpy as np

import upoqa


def _rastrigin(dim):
    # separable Rastrigin: f_i(x_i) = x_i^2 - 10 cos(2 pi x_i) + 10, fopt=0 at x=0
    fun = {
        f"r{i}": (lambda x: float(x[0] ** 2 - 10 * np.cos(2 * np.pi * x[0]) + 10))
        for i in range(dim)
    }
    coords = {f"r{i}": [i] for i in range(dim)}
    return fun, coords


def _run(seed, dim=5, maxfev=1000):
    fun, coords = _rastrigin(dim)
    x0 = np.linspace(2.0, 4.0, dim)
    return upoqa.minimize(
        fun,
        x0,
        coords=coords,
        maxfev=maxfev,
        seek_global_minimum=True,
        seed=seed,
        disp=False,
    )


def test_same_seed_reproducible():
    """Two runs with the same seed must be bitwise identical."""
    r1 = _run(seed=123)
    r2 = _run(seed=123)
    assert r1.nrun >= 2  # make sure the restart path (the RNG consumer) fired
    assert r1.fun == r2.fun
    assert np.array_equal(r1.x, r2.x)
    assert r1.nit == r2.nit


def test_different_seeds_differ():
    """Different seeds should give different trajectories (per-element nfev)."""
    r1 = _run(seed=123)
    r2 = _run(seed=456)
    assert r1.nrun >= 2  # restarts fired, so the RNG was actually consumed
    assert dict(r1.nfev) != dict(r2.nfev)


def test_global_rng_untouched():
    """The solver must not consume the global numpy random state."""
    np.random.seed(588)
    before = np.random.randn(8)
    np.random.seed(588)
    _run(seed=123)
    _run(seed=None)
    after = np.random.randn(8)
    assert np.array_equal(before, after)
