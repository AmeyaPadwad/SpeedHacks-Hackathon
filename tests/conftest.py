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


def pytest_sessionfinish(session, exitstatus):
    """Destroy leftover Qt windows (overlays, speech-bubble bars, world panels) while the QApplication still
    exists; left to interpreter shutdown, Qt can crash after every test has passed."""
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
