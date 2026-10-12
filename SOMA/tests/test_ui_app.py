"""App orchestration: consent gates the scheduler; tray; launcher smoke; real runtime."""

import subprocess
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from soma.config import SomaConfig, load_config, small_model_config
from soma.consent import ConsentManager, Scope
from soma.ui.desktop.app import SomaDesktop, main, parse_args
from soma.ui.desktop.tray import SomaTray
from tests.test_ui_bridge import FakeRuntime, FakeScheduler

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


class Builder:
    def __init__(self):
        self.calls = 0
        self.runtime = None

    def __call__(self, cfg, consent, *, data_dir, checkpoint, synthetic):
        self.calls += 1
        self.runtime = FakeRuntime()
        return self.runtime


def make_desktop(tmp_path, consent=None, **kw):
    scheds = []

    def factory(rt, iv, cb):
        s = FakeScheduler(rt, iv, cb)
        scheds.append(s)
        return s

    builder = Builder()
    desk = SomaDesktop(SomaConfig(), consent or ConsentManager(), data_dir=tmp_path,
                       runtime_builder=builder, scheduler_factory=factory,
                       with_tray=False, **kw)
    return desk, builder, scheds


def test_parse_args_defaults():
    a = parse_args([])
    assert a.config == "configs/dev.yaml" and not a.synthetic and not a.smoke
    a = parse_args(["--synthetic", "--data-dir", "d", "--checkpoint", "c.pt"])
    assert a.synthetic and a.data_dir == "d" and a.checkpoint == "c.pt"


def test_nothing_starts_when_dialog_declined(qapp, tmp_path):
    desk, builder, scheds = make_desktop(tmp_path)
    running = desk.edit_permissions(exec_dialog=lambda d: 0)
    assert not running and scheds == [] and builder.calls == 0
    assert desk.window.state == "Off"


def test_nothing_starts_without_capture_grant(qapp, tmp_path):
    desk, builder, scheds = make_desktop(tmp_path)

    def accept_feedback_only(dlg):
        dlg.checkboxes[Scope.FEEDBACK_STORAGE].setChecked(True)
        dlg.accept()
        return 1
    assert not desk.edit_permissions(exec_dialog=accept_feedback_only)
    assert scheds == [] and desk.window.state == "Off"
    assert desk.consent.allowed(Scope.FEEDBACK_STORAGE)


def test_capture_grant_starts_scheduler(qapp, tmp_path):
    desk, builder, scheds = make_desktop(tmp_path)

    def accept_capture(dlg):
        dlg.checkboxes[Scope.SCREEN_CAPTURE].setChecked(True)
        dlg.accept()
        return 1
    assert desk.edit_permissions(exec_dialog=accept_capture)
    assert builder.calls == 1 and scheds[0].events == ["start"]
    assert desk.window.state == "Capturing"

    def revoke(dlg):
        dlg.checkboxes[Scope.SCREEN_CAPTURE].setChecked(False)
        dlg.accept()
        return 1
    assert not desk.edit_permissions(exec_dialog=revoke)
    assert "stop" in scheds[0].events and desk.window.state == "Off"


def test_persisted_pause_is_respected(qapp, tmp_path):
    consent = ConsentManager()
    consent.grant(Scope.SCREEN_CAPTURE)
    consent.pause()
    desk, _, scheds = make_desktop(tmp_path, consent=consent)
    desk.apply_consent()
    assert desk.window.state == "Paused" and "pause" in scheds[0].events


def test_tray_toggles_and_tooltip(qapp, tmp_path):
    desk, _, _ = make_desktop(tmp_path)
    tray = SomaTray(desk.bridge)
    assert tray.tooltip == "SOMA: Off" and not tray.toggle_action.isEnabled()
    desk.consent.grant(Scope.SCREEN_CAPTURE)
    desk.apply_consent()
    assert tray.tooltip == "SOMA: Capturing"
    tray.toggle_action.trigger()
    assert tray.tooltip == "SOMA: Paused" and tray.toggle_action.text() == "Resume"
    assert desk.window.state == "Paused"
    tray.toggle_action.trigger()
    assert desk.window.state == "Capturing"


def test_folder_choice_is_ingested(qapp, tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    desk, builder, _ = make_desktop(tmp_path, pick_folder=lambda _p: str(folder))
    seen = []

    class Ing:
        def ingest_folder(self, f):
            seen.append(f)

            class R:
                changed, errors = 1, []
            return R()

    def accept_folder(dlg):
        dlg.checkboxes[Scope.FOLDER_INDEXING].setChecked(True)
        dlg.choose_folder()
        dlg.accept()
        return 1
    desk._build = lambda *a, **k: type("RT", (FakeRuntime,), {"memory": type("M", (), {
        "ingestor": Ing(), "store": None, "index": None})()})()
    desk.edit_permissions(exec_dialog=accept_folder)
    for _ in range(50):
        QCoreApplication.processEvents()
        if seen and "Indexed" in desk.window.ingest_label.text():
            break
        import time
        time.sleep(0.02)
    assert seen == [str(folder.resolve())]
    assert not desk.bridge.running   # folder indexing alone does not start capture


def test_main_smoke_in_process(qapp, capsys, tmp_path):
    assert main(["--smoke", "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "NO_SUGGESTION=" in out and "SOMA smoke" in out


def test_run_desktop_script_smoke_subprocess(tmp_path):
    proc = subprocess.run([sys.executable, "-m", "scripts.run_desktop", "--smoke",
                           "--data-dir", str(tmp_path)],
                          cwd=ROOT, capture_output=True, text=True, timeout=600,
                          env={**__import__("os").environ, "QT_QPA_PLATFORM": "offscreen"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "NO_SUGGESTION=" in proc.stdout


def _small_cfg():
    cfg = load_config(ROOT / "configs" / "dev.yaml")
    cfg.model = small_model_config()
    cfg.device = "cpu"
    return cfg


def test_integration_with_real_runtime(qapp, tmp_path):
    pytest.importorskip("soma.runtime.factory")
    from soma.runtime.factory import build_runtime

    consent = ConsentManager()
    consent.grant(Scope.SCREEN_CAPTURE)
    cfg = _small_cfg()
    desk = SomaDesktop(cfg, consent, data_dir=tmp_path, synthetic=True, with_tray=False,
                       runtime_builder=lambda c, k, **kw: build_runtime(c, k, **kw))
    try:
        desk.ensure_runtime()
        res = desk.bridge.step_sync(force=True)
        QCoreApplication.processEvents()
        assert res is not None and res.skipped_reason is None
        win = desk.window
        assert win.ocr_table.rowCount() == len(res.observation.elements) > 0
        labels = dict(win.chart.items)
        assert "NO_SUGGESTION" in labels and win.chart.total() == pytest.approx(1.0, abs=1e-3)
        assert win.debug.has_image and "frame" in win.debug.caption.text()
        assert "P50" in win.timing_label.text()
        win.memory_view.refresh()   # real store opened by the factory
        assert win.memory_view.store is desk.runtime.memory.store

        # pause -> runtime and consent paused; debug frame dropped
        desk.start()
        desk.bridge.pause()
        assert desk.runtime.paused and consent.paused and not win.debug.has_image
        assert desk.frame_tap.latest() is None
    finally:
        desk.shutdown()


def test_real_scheduler_delivers_then_pause_drops(qapp, tmp_path):
    pytest.importorskip("soma.runtime.scheduler")
    import time

    from soma.runtime.factory import build_runtime

    consent = ConsentManager()
    consent.grant(Scope.SCREEN_CAPTURE)
    cfg = _small_cfg()
    cfg.capture.min_interval_s = 0.05
    desk = SomaDesktop(cfg, consent, data_dir=tmp_path, synthetic=True, with_tray=False,
                       runtime_builder=lambda c, k, **kw: build_runtime(c, k, **kw))
    got = []
    desk.bridge.result.connect(got.append)
    try:
        desk.start()
        deadline = time.monotonic() + 60
        while not got and time.monotonic() < deadline:
            QCoreApplication.processEvents()
            time.sleep(0.02)
        assert got, "no result from the real scheduler"
        desk.bridge.pause()
        n = len(got)
        end = time.monotonic() + 0.5
        while time.monotonic() < end:
            QCoreApplication.processEvents()
            time.sleep(0.02)
        assert len(got) == n
        assert desk.window.state == "Paused"
    finally:
        desk.shutdown()
