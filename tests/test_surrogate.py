import numpy as np
import torch

from roughvol.surrogate.models import Surrogate, build_network
from roughvol.surrogate.train import masked_mse

LO = np.array([0.025, 0.5, -0.95, 0.01])
HI = np.array([0.5, 4.0, -0.1, 0.16])
SPEC = {"kind": "resmlp", "width": 16, "n_blocks": 2, "activation": "silu"}


def _model():
    torch.manual_seed(0)
    net = build_network(SPEC, 4, 6)
    return Surrogate(
        net, SPEC, LO, HI, np.full(6, 0.2), np.full(6, 0.05), [0.1, 0.5], [-0.1, 0, 0.1]
    )


def test_save_load_roundtrip(tmp_path):
    m = _model().eval()
    theta = LO + 0.3 * (HI - LO)
    m.save(tmp_path / "m.pt")
    m2 = Surrogate.load(tmp_path / "m.pt")
    np.testing.assert_allclose(m.predict(theta[None]), m2.predict(theta[None]), atol=1e-7)
    assert m2.predict(theta[None]).shape == (1, 2, 3)


def test_input_scaling_maps_box_to_unit_cube():
    m = _model()
    x = m.scale_x(torch.as_tensor(np.stack([LO, HI]), dtype=torch.float32))
    torch.testing.assert_close(x, torch.tensor([[-1.0] * 4, [1.0] * 4]))


def test_masked_mse_ignores_nan_cells():
    pred = torch.zeros(2, 3)
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[False, True, True], [True, True, True]])
    assert masked_mse(pred, target, mask).item() == 0.0


def test_network_fits_smooth_map():
    """The residual MLP should fit a smooth 4 -> 6 map to small error quickly."""
    torch.manual_seed(1)
    x = torch.rand(2048, 4) * 2 - 1
    y = torch.stack([torch.sin(x @ torch.randn(4)) for _ in range(6)], 1)
    net = build_network({"kind": "resmlp", "width": 64, "n_blocks": 2, "activation": "silu"}, 4, 6)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    for _ in range(400):
        loss = ((net(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-3
