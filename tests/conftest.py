import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Never read the developer's real VS Code snapshot: tests inject read_file= where they need one.
os.environ["FLOWSTATE_ACTIVE_FILE"] = str(Path(__file__).resolve().parent / "_no_active_file.json")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])
