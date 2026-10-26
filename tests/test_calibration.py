import numpy as np
import torch

from roughvol.calibration import lbfgs_calibrate, lm_calibrate
from roughvol.surrogate.models import Surrogate, build_network

LO = np.array([0.025, 0.5, -0.95, 0.01])
HI = np.array([0.5, 4.0, -0.1, 0.16])


def _toy_surrogate(seed=0):
    """Random smooth network: calibrating to its own output must reach zero cost."""
    torch.manual_seed(seed)
    net = build_network({"kind": "mlp", "hidden": [32, 32], "activation": "tanh"}, 4, 12)
    s = Surrogate(
        net,
        {"kind": "mlp", "hidden": [32, 32], "activation": "tanh"},
        LO,
        HI,
        np.full(12, 0.2),
        np.full(12, 0.05),
        [0.1, 0.5, 1.0],
        [-0.1, 0.0, 0.1, 0.2],
    )
    return s.eval()


def test_lm_self_consistency():
    s = _toy_surrogate()
    rng = np.random.default_rng(0)
    theta = LO + (HI - LO) * (0.1 + 0.8 * rng.random((16, 4)))
    target = s.predict(theta)
    res = lm_calibrate(s, target, n_starts=6, max_iter=200)
    assert np.median(res["rmse"]) < 1e-5
    assert np.all((res["theta"] >= LO) & (res["theta"] <= HI))
    np.testing.assert_allclose(s.predict(res["theta"]), target, atol=1e-3)


def test_lm_respects_mask():
    s = _toy_surrogate(1)
    theta = 0.5 * (LO + HI)[None]
    target = s.predict(theta)
    target[0, 0, :] = 9.0  # garbage cells, masked out
    mask = np.ones_like(target, bool)
    mask[0, 0, :] = False
    res = lm_calibrate(s, target, mask=mask, n_starts=4, max_iter=200)
    assert res["rmse"][0] < 1e-5


def test_lbfgs_self_consistency():
    s = _toy_surrogate(2)
    theta = LO + 0.37 * (HI - LO)
    target = s.predict(theta[None])[0]
    res = lbfgs_calibrate(s, target, n_starts=4)
    assert res["rmse"] < 1e-4
    assert np.all((res["theta"] >= LO) & (res["theta"] <= HI))


def test_lm_fd64_jacobian_matches_autodiff():
    s = _toy_surrogate(3)
    rng = np.random.default_rng(1)
    theta = LO + (HI - LO) * (0.1 + 0.8 * rng.random((8, 4)))
    target = s.predict(theta)
    a = lm_calibrate(s, target, n_starts=3, max_iter=200, jacobian="autodiff")
    b = lm_calibrate(s, target, n_starts=3, max_iter=200, jacobian="fd64")
    assert np.median(b["rmse"]) < 1e-5
    np.testing.assert_allclose(a["rmse"], b["rmse"], atol=1e-5)
