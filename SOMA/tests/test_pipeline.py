"""SomaRuntime behaviour: privacy gates, candidates, silence, dismissal, invalidation."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
import torch

from soma.capture.screen import ScreenCapturer
from soma.capture.window import static_window
from soma.config import CaptureConfig, PolicyConfig, SomaConfig, small_model_config
from soma.consent import ConsentManager, Scope
from soma.perception.ocr import SyntheticOCREngine
from soma.perception.synthetic import render_scenario
from soma.runtime.factory import open_memory
from soma.runtime.pipeline import SomaRuntime, fit_dim
from soma.runtime.telemetry import Telemetry
from soma.types import ActionType, FeedbackOutcome, ScreenFrame

SOMA_ROOT = Path(__file__).resolve().parent.parent


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class CountingOCR(SyntheticOCREngine):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def recognize(self, image, window_id="", timestamp_ms=0):
        self.calls += 1
        return super().recognize(image, window_id, timestamp_ms)


def make_cfg(eager: bool = False) -> SomaConfig:
    cfg = SomaConfig(device="cpu", model=small_model_config())
    if eager:  # speak whenever any candidate exists (tests the gate's other rules)
        cfg.policy = PolicyConfig(relevance_threshold=0.0, margin_over_abstain=-1.0, cooldown_s=0.0,
                                  duplicate_window_s=600.0)
    return cfg


def granted(*scopes: Scope) -> ConsentManager:
    c = ConsentManager()
    for s in scopes or (Scope.SCREEN_CAPTURE,):
        c.grant(s)
    return c


_counter = {"v": 0}


def frame_of(name: str, seed: int = 1) -> ScreenFrame:
    image, obs, _ = render_scenario(name, seed=seed, size=(640, 400))
    _counter["v"] += 1
    return ScreenFrame(image=image, window=obs.window, version=_counter["v"])


def count_model_calls(rt: SomaRuntime) -> dict[str, int]:
    calls = {"encode": 0}
    orig = rt.model.encode_screen

    def wrapped(*a, **k):
        calls["encode"] += 1
        return orig(*a, **k)

    rt.model.encode_screen = wrapped  # type: ignore[method-assign]
    return calls


def runtime(cfg=None, consent=None, **kw) -> SomaRuntime:
    cfg = cfg or make_cfg()
    kw.setdefault("ocr", CountingOCR())
    kw.setdefault("clock", Clock())
    return SomaRuntime(cfg, consent or granted(), **kw)


# ------------------------------------------------------------ privacy gates


def capturer_for(consent, app="WindowsTerminal.exe", title="pytest"):
    grabs = {"n": 0}
    image, _, _ = render_scenario("terminal_traceback", seed=1, size=(640, 400))

    def grab(window):
        grabs["n"] += 1
        return image

    cap = ScreenCapturer(consent, CaptureConfig(min_interval_s=0.0), grab=grab,
                         window_fn=static_window(app, title, bounds=(0, 0, 640, 400)))
    return cap, grabs


@pytest.mark.parametrize("case", ["no-consent", "paused", "excluded-app", "excluded-title"])
def test_gated_steps_do_no_ocr_and_no_model_work(case):
    consent = ConsentManager() if case == "no-consent" else granted()
    app, title = "WindowsTerminal.exe", "pytest"
    if case == "excluded-app":
        app = "1Password.exe"
    if case == "excluded-title":
        title = "Sign in to your bank"
    cap, grabs = capturer_for(consent, app, title)
    rt = runtime(consent=consent, capturer=cap)
    calls = count_model_calls(rt)
    if case == "paused":
        rt.pause()
    for _ in range(3):
        r = rt.step(force=True)
        assert r.skipped_reason is not None
        assert r.suggestion.is_silent and r.candidates == [] and r.observation.elements == []
        assert r.window.title == ""                         # excluded titles are not echoed
    assert rt.ocr.calls == 0 and calls["encode"] == 0 and grabs["n"] == 0
    assert rt.last_result is None
    expected = {"no-consent": "no-consent", "paused": "paused"}.get(case, "excluded")
    assert r.skipped_reason.startswith(expected)


def test_process_frame_applies_the_same_gates():
    rt = runtime(consent=ConsentManager())
    calls = count_model_calls(rt)
    assert rt.process_frame(frame_of("terminal_traceback")).skipped_reason == "no-consent"
    rt.consent.grant(Scope.SCREEN_CAPTURE)
    frame = frame_of("terminal_traceback")
    frame.window.app_name = "KeePassXC.exe"
    assert rt.process_frame(frame).skipped_reason.startswith("excluded")
    rt.pause()
    assert rt.process_frame(frame_of("terminal_traceback")).skipped_reason == "paused"
    assert rt.ocr.calls == 0 and calls["encode"] == 0


def test_capture_unchanged_frame_is_skipped_without_ocr():
    consent = granted()
    cap, grabs = capturer_for(consent)
    rt = runtime(consent=consent, capturer=cap)
    assert rt.step().skipped_reason is None
    r = rt.step()
    assert r.skipped_reason == "unchanged"
    assert rt.ocr.calls == 1 and grabs["n"] == 2


def test_resume_after_pause_processes_again_and_history_was_cleared():
    consent = granted()
    cap, _ = capturer_for(consent)
    rt = runtime(consent=consent, capturer=cap)
    rt.step(force=True)
    rt.step(force=True)
    assert len(rt.history) == 2
    rt.pause()
    assert rt.paused and consent.paused and len(rt.history) == 0
    rt.resume()
    assert not rt.paused
    assert rt.step().skipped_reason is None


# ------------------------------------------------------------ candidates and silence


def test_terminal_traceback_with_preference_memory(tmp_path):
    cfg = make_cfg()
    consent = granted()
    mem = open_memory(cfg, consent, tmp_path / "data")
    pref = mem.ingestor.add_preference("When a traceback appears, inspect the last frame first",
                                       tags=["coding"])
    rt = runtime(cfg, consent, memory=mem)
    r = rt.process_frame(frame_of("terminal_traceback"))
    assert r.skipped_reason is None
    actions = [c.action for c in r.candidates]
    assert ActionType.EXPLAIN_ERROR in actions
    assert ActionType.SUGGEST_NEXT_STEP in actions           # from the preference memory
    assert pref.id in {m.item.id for m in r.retrieved}
    assert "NO_SUGGESTION" in r.suggestion.distribution
    assert sum(r.suggestion.distribution.values()) == pytest.approx(1.0, abs=1e-4)
    assert set(r.timings_ms) >= {"capture", "ocr", "parse", "encode", "retrieve", "policy",
                                 "gate", "total"}
    assert r.suggestion.screen_version == r.frame_version
    mem.close()


def test_idle_desktop_stays_silent():
    rt = runtime(make_cfg(eager=True))
    r = rt.process_frame(frame_of("idle_desktop"))
    assert r.skipped_reason is None
    assert r.candidates == []
    assert r.suggestion.is_silent
    assert r.suggestion.distribution == {"NO_SUGGESTION": pytest.approx(1.0)}


def test_empty_memory_bundle_and_no_memory_both_work(tmp_path):
    cfg = make_cfg()
    consent = granted()
    mem = open_memory(cfg, consent, tmp_path / "data")
    for m in (mem, None):
        rt = runtime(cfg, consent, memory=m)
        r = rt.process_frame(frame_of("editor_error"))
        assert r.retrieved == []
        assert r.skipped_reason is None
        assert all(torch.isfinite(torch.tensor(list(r.suggestion.distribution.values()))))
    mem.close()


def test_score_matches_full_model_forward():
    rt = runtime()
    prepared = rt.prepare(frame_of("terminal_traceback"))
    rt.ranker = None
    fast = rt.score(prepared)
    with torch.inference_mode():
        full = rt.model(prepared.batch)["action_logits"].float().cpu()
    torch.testing.assert_close(fast, full, rtol=1e-4, atol=1e-4)
    assert fast.shape == (1, rt.cfg.model.max_candidates + 1)


def test_dismissed_suggestion_is_not_repeated():
    clock = Clock()
    rt = runtime(make_cfg(eager=True), clock=clock)
    first = rt.process_frame(frame_of("terminal_traceback"))
    assert not first.suggestion.is_silent
    key = first.suggestion.candidate.key
    rt.feedback(first.suggestion.id, FeedbackOutcome.REJECTED)
    for i in range(5):
        clock.t += 5.0
        r = rt.process_frame(frame_of("terminal_traceback"))
        assert r.suggestion.is_silent or r.suggestion.candidate.key != key
        assert key in {c.key for c in r.candidates}          # still a candidate, just never shown


def test_ignored_is_not_a_dismissal():
    clock = Clock()
    cfg = make_cfg(eager=True)
    cfg.policy.duplicate_window_s = 1.0
    rt = runtime(cfg, clock=clock)
    first = rt.process_frame(frame_of("editor_error"))
    key = first.suggestion.candidate.key
    rt.feedback(first.suggestion.id, FeedbackOutcome.IGNORED)
    clock.t += 2.0
    again = rt.process_frame(frame_of("editor_error"))
    assert not again.suggestion.is_silent and again.suggestion.candidate.key == key


def test_suggestion_invalidated_when_version_changes_or_after_feedback():
    rt = runtime(make_cfg(eager=True))
    r = rt.process_frame(frame_of("terminal_traceback"))
    assert not r.suggestion.is_silent
    v = r.frame_version
    assert rt.current_suggestion_valid(v)
    assert not rt.current_suggestion_valid(v + 1)
    rt.feedback(r.suggestion.id, FeedbackOutcome.ACCEPTED)
    assert not rt.current_suggestion_valid(v)


def test_feedback_routing_respects_consent(tmp_path):
    cfg = make_cfg(eager=True)
    consent = granted(Scope.SCREEN_CAPTURE)
    mem = open_memory(cfg, consent, tmp_path / "data")
    rt = runtime(cfg, consent, memory=mem)
    r = rt.process_frame(frame_of("terminal_traceback"))
    rt.feedback(r.suggestion.id, FeedbackOutcome.ACCEPTED)
    assert mem.store.list_feedback() == [] and rt.recorder.examples == []
    assert rt.ranker.updates == 0
    consent.grant(Scope.FEEDBACK_STORAGE)
    rt.clock.t += 1000
    r2 = rt.process_frame(frame_of("editor_error"))
    rt.feedback(r2.suggestion.id, FeedbackOutcome.EDITED, edited_text="explain it briefly")
    events = mem.store.list_feedback()
    assert [e.outcome for e in events] == [FeedbackOutcome.EDITED]
    assert events[0].consented_for_training is False          # no MODEL_TRAINING grant
    assert rt.ranker.updates == 1
    ex = rt.recorder.examples[-1]
    assert ex.item.target == ex.shown_index and bool((ex.item.images == 0).all())
    feedback_mems = [i for i in mem.store.list_items() if i.source_type.value == "feedback"]
    assert len(feedback_mems) == 1 and "explain it briefly" in feedback_mems[0].text_summary
    rt.feedback("sug_unknown", FeedbackOutcome.ACCEPTED)    # ignored, no crash
    mem.close()


# ------------------------------------------------------------ privacy and safety


def test_no_frame_or_screen_text_persisted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    consent = granted()
    cap, _ = capturer_for(consent)
    rt = runtime(consent=consent, capturer=cap, telemetry=Telemetry(enabled=True))
    for _ in range(3):
        rt.step(force=True)
    rt.process_frame(frame_of("browser_docs"))
    assert list(work.iterdir()) == []
    exported = repr(rt.telemetry.export())
    assert "Traceback" not in exported and "pytest" not in exported


def test_memory_dir_holds_no_images_after_steps(tmp_path):
    cfg = make_cfg(eager=True)
    consent = granted(Scope.SCREEN_CAPTURE, Scope.FEEDBACK_STORAGE)
    mem = open_memory(cfg, consent, tmp_path / "data")
    rt = runtime(cfg, consent, memory=mem)
    r = rt.process_frame(frame_of("terminal_traceback"))
    rt.feedback(r.suggestion.id, FeedbackOutcome.ACCEPTED)
    files = [p for p in (tmp_path / "data").rglob("*") if p.is_file()]
    assert all(p.suffix.lower() not in {".png", ".jpg", ".npy", ".bmp"} for p in files)
    logged = mem.store.list_suggestions()
    assert logged and all("Traceback" not in repr(row) for row in logged)
    mem.close()


def test_zero_executions():
    rt = runtime(make_cfg(eager=True))
    for name in ("terminal_traceback", "editor_error", "file_explorer"):
        r = rt.process_frame(frame_of(name))
        if not r.suggestion.is_silent:
            rt.feedback(r.suggestion.id, FeedbackOutcome.ACCEPTED)
    assert rt.ledger.executors == {}
    assert rt.ledger.executed_count == 0
    assert rt.ledger.unapproved_execution_count == 0


FORBIDDEN_MODULES = {"subprocess", "pyautogui", "pynput", "keyboard", "mouse", "webbrowser",
                     "shutil", "ctypes", "socket", "urllib", "requests"}
FORBIDDEN_ATTR_CALLS = {"system", "popen", "startfile", "execv", "execve", "spawnv"}
FORBIDDEN_BUILTINS = {"eval", "exec", "__import__"}


@pytest.mark.parametrize("rel", ["soma/runtime/pipeline.py", "soma/runtime/scheduler.py",
                                 "soma/runtime/factory.py", "soma/runtime/history.py",
                                 "soma/runtime/telemetry.py"])
def test_runtime_modules_import_no_execution_machinery(rel):
    tree = ast.parse((SOMA_ROOT / rel).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            names = []
        assert not set(names) & FORBIDDEN_MODULES, (rel, names)
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute):
                assert fn.attr not in FORBIDDEN_ATTR_CALLS, (rel, fn.attr)
            elif isinstance(fn, ast.Name):
                assert fn.id not in FORBIDDEN_BUILTINS, (rel, fn.id)


def test_fit_dim_pads_truncates_and_normalises():
    import numpy as np
    v = fit_dim(np.ones(3, dtype=np.float32), 5)
    assert v.shape == (5,) and abs(float(np.linalg.norm(v)) - 1.0) < 1e-6 and v[-1] == 0
    assert fit_dim(np.arange(8, dtype=np.float32), 4).shape == (4,)
    assert not np.any(fit_dim(np.zeros(4), 4))
