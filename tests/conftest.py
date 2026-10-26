# ruff: noqa: I001  (roughvol must be imported first: it loads pyarrow before torch)
import roughvol  # noqa: F401  (loads pyarrow before torch; see roughvol/__init__.py)
import pytest
import torch

from roughvol.utils import get_device, set_seed


@pytest.fixture(autouse=True)
def _seed():
    set_seed(0)


@pytest.fixture(scope="session")
def device() -> torch.device:
    return get_device("auto")
