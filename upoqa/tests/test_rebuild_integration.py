import sys

sys.path.append("..")
import numpy as np
import pytest

import upoqa
import upoqa.utils.model as model_module
from upoqa.utils.model import QuadSurrogate


def test_failed_update_falls_back_to_rebuild_in_minimize():
    """
    Force QuadSurrogate.update to fail once during a real minimize() run.
    The run must continue (no restart, no crash) via the rebuild() fallback
    in OverallSurrogate.update, and still converge.
    """
    stats = {"fail_injected": 0, "rebuild_called": 0}
    orig_update = QuadSurrogate.update
    orig_rebuild = QuadSurrogate.rebuild

    def patched_update(self, cached_kkt_info=None):
        # Fail exactly once, on the 5th update call overall
        patched_update.calls += 1
        if patched_update.calls == 5:
            stats["fail_injected"] += 1
            raise ZeroDivisionError("injected failure for testing")
        return orig_update(self, cached_kkt_info)

    patched_update.calls = 0

    def patched_rebuild(self, *args, **kwargs):
        stats["rebuild_called"] += 1
        return orig_rebuild(self, *args, **kwargs)

    QuadSurrogate.update = patched_update
    QuadSurrogate.rebuild = patched_rebuild
    try:
        res = upoqa.minimize(
            # f = x^2 + 2y^2 + (y-0.5)^2 + z^2, minimum 1/6 at (0, 1/6, 0)
            {"a": lambda x: x[0] ** 2 + 2 * x[1] ** 2,
             "b": lambda x: (x[0] - 0.5) ** 2 + x[1] ** 2},
            x0=[1.0, 1.0, 1.0],
            coords={"a": [0, 1], "b": [1, 2]},
            disp=0,
        )
    finally:
        QuadSurrogate.update = orig_update
        QuadSurrogate.rebuild = orig_rebuild

    assert stats["fail_injected"] == 1
    assert stats["rebuild_called"] >= 1
    assert res.success, res.message
    assert res.nrun == 1  # rebuild avoided the restart
    assert abs(res.fun - 1.0 / 6.0) < 1e-8
