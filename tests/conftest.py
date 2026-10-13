import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Never talk to a real SOMA from tests: fakes are injected where needed.
os.environ["PETTE_SOMA_ENDPOINT"] = str(Path(__file__).resolve().parent / "_no_soma_endpoint.json")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])
