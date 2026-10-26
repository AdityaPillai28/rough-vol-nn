import copy

import numpy as np
import pytest

from roughvol.config import load_config
from roughvol.data import generate_dataset, load_dataset


def _tiny_cfg(tmp_path, name):
    cfg = copy.deepcopy(load_config(quick=True))
    d = cfg["data"]
    d["grid"]["maturities"] = [0.05, 0.25]
    d["grid"]["z"] = [-0.2, 0.0, 0.2]
    d["mc"].update(steps_per_year=40, n_paths=400, path_batch=400)
    d["generation"].update(
        n_samples=10,
        chunk_size=4,
        raw_dir=str(tmp_path / name / "raw"),
        processed=str(tmp_path / name / "ds.npz"),
    )
    cfg["results_dir"] = str(tmp_path / name)
    return cfg


def test_resume_is_bitwise_identical(tmp_path):
    a = _tiny_cfg(tmp_path, "a")
    first = generate_dataset(a, device="cpu", max_chunks=1)
    assert not first["complete"]
    assert len(list((tmp_path / "a" / "raw").glob("chunk_*.npz"))) == 1
    done = generate_dataset(a, device="cpu")
    assert done["complete"]

    b = _tiny_cfg(tmp_path, "b")
    generate_dataset(b, device="cpu")
    da, db = load_dataset(a), load_dataset(b)
    for key in ("params", "iv", "iv_se", "price"):
        np.testing.assert_array_equal(da[key], db[key])
    assert da["iv"].shape == (10, 2, 3)
    assert set(np.unique(da["split"])) <= {0, 1, 2}


def test_resume_refuses_changed_config(tmp_path):
    a = _tiny_cfg(tmp_path, "a")
    generate_dataset(a, device="cpu", max_chunks=1)
    a["data"]["mc"]["n_paths"] = 800
    with pytest.raises(RuntimeError, match="different config"):
        generate_dataset(a, device="cpu")
