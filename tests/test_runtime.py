import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_real_app_exits_when_signalled_while_idle(tmp_path, signum):
    ready = tmp_path / "ready"
    source = """
import sys
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from pet.runtime import TerminalShutdown
app = QApplication([])
app.setQuitOnLastWindowClosed(False)
shutdown = TerminalShutdown(app)
QTimer.singleShot(10, lambda: Path(sys.argv[1]).write_text('ready'))
result = app.exec()
assert not shutdown.timer.isActive()
sys.exit(result)
"""
    process = subprocess.Popen([sys.executable, "-c", source, str(ready)],
        cwd=Path(__file__).resolve().parents[1], env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), process.communicate(timeout=1)[1].decode()
        started = time.monotonic()
        process.send_signal(signum)
        _, errors = process.communicate(timeout=3)
        assert process.returncode == 0, errors.decode()
        assert time.monotonic() - started < 2
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)


def test_input_gathering_entrypoint_closes_pet_on_ctrl_c(tmp_path):
    ready = tmp_path / "ready"
    source = """
import sys
from pathlib import Path
from PySide6.QtCore import QTimer
import main
real_application = main.QApplication
project = Path(sys.argv[1])
marker = project / 'ready'
def create_application(args):
    app = real_application(args)
    QTimer.singleShot(100, lambda: marker.write_text('ready'))
    return app
main.QApplication = create_application
sys.argv = ['main.py', '--input-gathering', '--project', str(project)]
sys.exit(main.main())
"""
    process = subprocess.Popen([sys.executable, "-c", source, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 12
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), process.communicate(timeout=1)[1].decode()
        process.send_signal(signal.SIGINT)
        _, errors = process.communicate(timeout=5)
        assert process.returncode == 0, errors.decode()
        assert b"KeyboardInterrupt" not in errors
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
