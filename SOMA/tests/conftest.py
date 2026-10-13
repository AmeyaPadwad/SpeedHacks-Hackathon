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


def pytest_sessionfinish(session, exitstatus):
    """Destroy leftover Qt widgets while the QApplication still exists.

    Tests create widgets without closing them; left to interpreter shutdown, Python can free the
    QApplication first and Qt then crashes (0xC0000005 / 0xC0000409 on Windows) after every test passed.
    """
    import gc
    try:
        from PySide6.QtCore import QCoreApplication, QEvent
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    app = QApplication.instance()
    if app is None:
        return
    for w in QApplication.topLevelWidgets():
        w.close()
        w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    gc.collect()
