import pytest
import torch

from roughvol.utils import set_seed


@pytest.fixture(autouse=True)
def _seed():
    set_seed(0)


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
