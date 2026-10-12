import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield


@pytest.fixture
def small_cfg():
    from soma.config import small_model_config
    return small_model_config()
