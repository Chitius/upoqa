import sys

sys.path.append("..")
import numpy as np
import numpy.linalg as LA
import pytest

from upoqa.utils.interp_set import InterpSet
from upoqa.utils.model import QuadSurrogate


def _dense_kkt(Y, ref):
    """Dense KKT matrix of the min-Frobenius quadratic interpolation system."""
    npt, n = Y.shape
    D = Y - ref
    W = np.zeros((npt + n + 1, npt + n + 1))
    W[:npt, :npt] = 0.5 * (D @ D.T) ** 2
    P = np.hstack([D, np.ones((npt, 1))])
    W[:npt, npt:] = P
    W[npt:, :npt] = P.T
    return W


def _make_model(n, npt, seed=42):
    rng = np.random.default_rng(seed)
    Q, _ = LA.qr(rng.standard_normal((n, n)))
    H = Q @ np.diag(np.linspace(0.5, 5, n)) @ Q.T
    cc = rng.standard_normal(n)

    def fun(x):
        return float(x @ H @ x + cc @ x + 0.1 * np.sum(x**3))

    center = rng.standard_normal(n)
    iset = InterpSet(n, npt)
    iset.init_interp_set(fun, center, step_size=1.0)
    model = QuadSurrogate(iset, center)
    return model, iset, fun, rng


def _random_updates(model, iset, fun, rng, count):
    for _ in range(count):
        idx = int(rng.integers(0, iset.npt))
        x_new = model.model_center + 0.4 * rng.standard_normal(iset.n)
        iset.update_point_on_idx(x_new, idx, fun(x_new))
        model.update()


def _interp_err(model, iset):
    Y, fv = iset.get_interp_set()
    return max(abs(model.fun_eval(Y[k]) - fv[k]) for k in range(iset.npt))


def _det_rel_err(model, iset, rng, trials=8):
    Y, _ = iset.get_interp_set()
    err = 0.0
    for _ in range(trials):
        idx = int(rng.integers(0, iset.npt))
        x_new = model.model_center + 0.3 * rng.standard_normal(iset.n)
        W_old = _dense_kkt(Y, model.model_center)
        Y2 = Y.copy()
        Y2[idx] = x_new
        dr = LA.det(_dense_kkt(Y2, model.model_center)) / LA.det(W_old)
        sg = model.get_determinant_ratio(x_new, idx)[3]
        sg = sg[idx] if isinstance(sg, np.ndarray) else sg
        err = max(err, abs(sg - dr) / abs(dr))
    return err


@pytest.mark.parametrize("npt_mode", ["min", "default", "full"])
def test_rebuild_recovers_corrupted_factors(npt_mode):
    n = 4
    npt = {"min": n + 2, "default": 2 * n + 1, "full": (n + 1) * (n + 2) // 2}[
        npt_mode
    ]
    model, iset, fun, rng = _make_model(n, npt)
    _random_updates(model, iset, fun, rng, 40)
    det_before = _det_rel_err(model, iset, rng)
    assert det_before < 1e-8

    Y_before = iset.interp_set_Y.copy()
    opt_before, anch_before = iset.x_opt_idx, iset.x_anchor_idx

    # Corrupt the factors and rebuild
    model._KKT_R += 1e-2 * rng.standard_normal(model._KKT_R.shape)
    model._KKT_B += 1e-2 * rng.standard_normal(model._KKT_B.shape)
    assert _det_rel_err(model, iset, rng) > 1e-3  # corruption is effective

    model.rebuild()
    assert _interp_err(model, iset) < 1e-8
    # The determinant-ratio check involves dense determinants of the KKT
    # system, whose conditioning deteriorates sharply for larger npt; the
    # discriminating power against corruption (errors ~1e2) is retained.
    det_tol = {"min": 1e-8, "default": 1e-8, "full": 1e-3}[npt_mode]
    assert _det_rel_err(model, iset, rng) < det_tol
    assert np.array_equal(iset.interp_set_Y, Y_before)
    assert (iset.x_opt_idx, iset.x_anchor_idx) == (opt_before, anch_before)


def test_negative_sigma_update_self_heals():
    # A near-duplicate point makes sigma slightly negative, activating the
    # negative track of _update_KKT_coeff. The factors must stay consistent
    # and self-heal afterwards.
    n, npt = 4, 2 * 4 + 1
    model, iset, fun, rng = _make_model(n, npt)
    _random_updates(model, iset, fun, rng, 20)

    # A near-duplicate point makes sigma tiny; its sign after cancellation is
    # essentially random, so probe a few candidates until sigma < 0 actually
    # triggers. The cached quantities are computed BEFORE updating the set
    # and passed to update(), exactly as the production flow does.
    Y, _ = iset.get_interp_set()
    triggered = False
    for src, idx in ((3, 5), (1, 7), (6, 2), (2, 8), (4, 0)):
        for eps in (1e-13, 1e-14, 1e-12):
            x_dup = Y[src] + eps * rng.standard_normal(n)
            cached = model.get_determinant_ratio(x_dup)
            if cached[3][idx] < 0.0:
                triggered = True
                break
        if triggered:
            break
    assert triggered, "could not trigger a negative sigma event"

    iset.update_point_on_idx(x_dup, idx, fun(x_dup))
    model.update(cached_kkt_info=cached)
    assert model._negative_s_idx == 1

    _random_updates(model, iset, fun, rng, 60)
    assert _interp_err(model, iset) < 1e-8
    assert _det_rel_err(model, iset, rng) < 1e-8
    assert model._negative_s_idx == 0
