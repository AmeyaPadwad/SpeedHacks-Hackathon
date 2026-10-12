"""Consent-first desktop app: wires consent dialog, runtime, bridge, window and tray.

Order of events on launch:

1. The consent dialog opens (all scopes off on a fresh install).
2. The runtime is built (model + local memory); building captures nothing.
3. The scheduler starts only if screen capture was granted. Without it the
   window shows "Off" and the user can reopen Permissions at any time.
4. Folders picked in the dialog are indexed in the background.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

from soma.config import SOMA_ROOT, SomaConfig, load_config
from soma.consent import ConsentManager, Scope
from soma.ui.desktop.bridge import RuntimeBridge, SchedulerFactory
from soma.ui.desktop.consent_dialog import ConsentDialog, FolderPicker
from soma.ui.desktop.debug_overlay import FrameTap
from soma.ui.desktop.memory_viewer import ExportPicker

RuntimeBuilder = Callable[..., Any]
CONSENT_FILE = "consent.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m scripts.run_desktop",
                                description="SOMA consent-first desktop UI.")
    p.add_argument("--synthetic", action="store_true",
                   help="use synthetic screens instead of real screen capture")
    p.add_argument("--config", default="configs/dev.yaml", help="YAML config (default: dev)")
    p.add_argument("--data-dir", default=None,
                   help="where local memory and consent live (default: SOMA/data)")
    p.add_argument("--checkpoint", default=None, help="model checkpoint to load")
    p.add_argument("--smoke", action="store_true",
                   help="build the window, run one synthetic step, print a summary and exit")
    return p.parse_args(list(argv) if argv is not None else None)


def capture_granted(consent: ConsentManager) -> bool:
    """Screen capture granted, regardless of pause."""
    state = getattr(consent, "_state", None)
    if state is not None and hasattr(state, "granted"):
        return Scope.SCREEN_CAPTURE in state.granted
    return consent.allowed(Scope.SCREEN_CAPTURE)


def default_runtime_builder(cfg: SomaConfig, consent: ConsentManager, *, data_dir: Path,
                            checkpoint: Path | None, synthetic: bool) -> Any:
    from soma.runtime.factory import build_runtime
    return build_runtime(cfg, consent, data_dir=data_dir, checkpoint=checkpoint,
                         synthetic=synthetic, telemetry=True)


class SomaDesktop:
    """Owns the long-lived objects. Testable with injected builders and pickers."""

    def __init__(self, cfg: SomaConfig, consent: ConsentManager, *, data_dir: Path,
                 checkpoint: Path | None = None, synthetic: bool = False,
                 runtime_builder: RuntimeBuilder | None = None,
                 scheduler_factory: SchedulerFactory | None = None,
                 pick_folder: FolderPicker | None = None,
                 export_picker: ExportPicker | None = None,
                 with_tray: bool = True) -> None:
        from soma.ui.desktop.main_window import MainWindow

        self.cfg = cfg
        self.consent = consent
        self.data_dir = Path(data_dir)
        self.checkpoint = checkpoint
        self.synthetic = synthetic
        self._build = runtime_builder or default_runtime_builder
        self._pick_folder = pick_folder
        self.runtime: Any = None
        self.frame_tap: FrameTap | None = None
        self.bridge = RuntimeBridge(None, cfg.capture.min_interval_s,
                                    scheduler_factory=scheduler_factory)
        self.window = MainWindow(self.bridge, consent, export_picker=export_picker)
        self.window.permissions_requested.connect(self.edit_permissions)
        self.tray = None
        if with_tray:
            from soma.ui.desktop.tray import SomaTray
            self.tray = SomaTray(self.bridge)
            self.tray.show_action.triggered.connect(self._show_window)

    # -- consent
    def make_dialog(self) -> ConsentDialog:
        return ConsentDialog(self.consent, pick_folder=self._pick_folder, parent=self.window)

    def edit_permissions(self, exec_dialog: Callable[[ConsentDialog], int] | None = None) -> bool:
        """Show the consent dialog, then apply the result. Returns True if capture runs."""
        dlg = self.make_dialog()
        accepted = (exec_dialog or (lambda d: d.exec()))(dlg)
        folders = list(dlg.chosen_folders) if accepted else []
        dlg.deleteLater()
        return self.apply_consent(folders)

    def apply_consent(self, new_folders: Sequence[str] = ()) -> bool:
        if capture_granted(self.consent) or new_folders:
            self.ensure_runtime()
        if capture_granted(self.consent):
            self.start()
        else:
            self.bridge.stop()
        if new_folders and self.runtime is not None:
            ingestor = getattr(getattr(self.runtime, "memory", None), "ingestor", None)
            self.window.ingest_folders(list(new_folders), ingestor)
        return self.bridge.running

    # -- runtime
    def ensure_runtime(self) -> Any:
        if self.runtime is None:
            self.runtime = self._build(self.cfg, self.consent, data_dir=self.data_dir,
                                       checkpoint=self.checkpoint, synthetic=self.synthetic)
            capturer = getattr(self.runtime, "capturer", None)
            if capturer is not None and hasattr(capturer, "capture"):
                self.frame_tap = FrameTap(capturer)
                self.runtime.capturer = self.frame_tap
                self.window.frame_tap = self.frame_tap
            self.bridge.set_runtime(self.runtime)
            self.window.set_memory(getattr(self.runtime, "memory", None))
        return self.runtime

    def start(self) -> None:
        if not capture_granted(self.consent):
            return
        self.ensure_runtime()
        was_paused = self.consent.paused
        if not self.bridge.running:
            self.bridge.start()
            if was_paused:
                self.bridge.pause()

    def _show_window(self) -> None:
        self.window.showNormal()
        self.window.raise_()
        self.window.activateWindow()

    def shutdown(self) -> None:
        self.bridge.stop()
        if self.frame_tap is not None:
            self.frame_tap.clear()
        if self.tray is not None:
            self.tray.hide()
        mem = getattr(self.runtime, "memory", None)
        if mem is not None and hasattr(mem, "close"):
            try:
                mem.close()
            except Exception:
                pass


def resolve_data_dir(arg: str | None) -> Path:
    if arg is None:
        return SOMA_ROOT / "data"
    p = Path(arg)
    return p if p.is_absolute() else Path.cwd() / p


def smoke_error_frame(desk: "SomaDesktop", version: int) -> Any:
    """Feed one synthetic traceback screen so the smoke run shows real candidates."""
    from soma.perception import render_scenario
    from soma.types import ScreenFrame

    image, obs, _ = render_scenario("terminal_traceback", seed=0)
    res = desk.runtime.process_frame(ScreenFrame(image=image, window=obs.window, version=version))
    desk.bridge.result.emit(res)
    return res


def run_smoke(args: argparse.Namespace, cfg: SomaConfig) -> int:
    """Build the window, process one synthetic step, print a short summary."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    tmp = None
    if args.data_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="soma_smoke_")
        data_dir = Path(tmp.name)
    else:
        data_dir = resolve_data_dir(args.data_dir)
    consent = ConsentManager()          # in memory: smoke grants never persist
    consent.grant(Scope.SCREEN_CAPTURE)
    desk = SomaDesktop(cfg, consent, data_dir=data_dir,
                       checkpoint=Path(args.checkpoint) if args.checkpoint else None,
                       synthetic=True, with_tray=False)
    try:
        desk.ensure_runtime()
        desk.window.show()
        res = desk.bridge.step_sync(force=True)
        app.processEvents()
        if res is None or res.skipped_reason:
            print(f"SOMA smoke: step skipped ({getattr(res, 'skipped_reason', None)})")
            return 1
        if not res.candidates and hasattr(desk.runtime, "process_frame"):
            res = smoke_error_frame(desk, res.frame_version + 1)
            app.processEvents()
        dist = res.suggestion.distribution
        print(f"SOMA smoke: app={res.window.app_name} elements={len(res.observation.elements)} "
              f"candidates={len(res.candidates)} chosen={desk.window.chart.chosen}")
        print("SOMA smoke: scores " + ", ".join(f"{k}={v:.3f}" for k, v in dist.items()))
        print(f"SOMA smoke: {desk.window.timing_text()}")
        ok = "NO_SUGGESTION" in dist and abs(sum(dist.values()) - 1.0) < 1e-3
        return 0 if ok else 1
    finally:
        desk.window.close()
        desk.shutdown()
        if tmp is not None:
            tmp.cleanup()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    cfg = load_config(args.config)
    if args.smoke:
        return run_smoke(args, cfg)

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("SOMA")
    app.setQuitOnLastWindowClosed(True)
    data_dir = resolve_data_dir(args.data_dir)
    consent = ConsentManager(data_dir / CONSENT_FILE)
    desk = SomaDesktop(cfg, consent, data_dir=data_dir,
                       checkpoint=Path(args.checkpoint) if args.checkpoint else None,
                       synthetic=args.synthetic)
    desk.window.show()
    desk.edit_permissions()
    app.aboutToQuit.connect(desk.shutdown)
    return int(app.exec())
