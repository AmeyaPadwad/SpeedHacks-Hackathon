"""scripts.serve: consent opt-in, one-shot diagnostic, and the served endpoint file."""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import pytest

from scripts import serve
from soma.consent import ConsentManager, Scope

SMALL = """
device: cpu
model: {d_model: 64, n_heads: 4, dropout: 0.0, image_size: 64, patch_size: 16, vision_layers: 1,
        vocab_size: 512, max_ocr_tokens: 32, text_layers: 1, fusion_layers: 1, history_len: 4,
        n_apps: 16, max_memories: 4, max_candidates: 4}
capture: {min_interval_s: 0.0}
policy: {relevance_threshold: 0.0, margin_over_abstain: -1.0, cooldown_s: 0, context_templates: true}
"""


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "small.yaml"
    p.write_text(SMALL, encoding="utf-8")
    return str(p)


def test_grant_screen_is_an_explicit_opt_in(tmp_path):
    assert not serve.open_consent(tmp_path, grant_screen=False).allowed(Scope.SCREEN_CAPTURE)
    serve.open_consent(tmp_path, grant_screen=True)
    reloaded = ConsentManager(tmp_path / serve.CONSENT_FILE)
    assert reloaded.allowed(Scope.SCREEN_CAPTURE) and reloaded.allowed(Scope.APP_MONITORING)
    assert not reloaded.allowed(Scope.FEEDBACK_STORAGE)          # nothing else is granted


def test_once_without_consent_reports_and_exits(tmp_path, cfg, capsys):
    assert serve.main(["--config", cfg, "--data-dir", str(tmp_path), "--synthetic", "--once"]) == 0
    out = capsys.readouterr()
    assert "not granted" in out.err and "skipped: no-consent" in out.out


def test_once_with_consent_runs_a_step(tmp_path, cfg, capsys):
    assert serve.main(["--config", cfg, "--data-dir", str(tmp_path), "--synthetic", "--once",
                       "--grant-screen"]) == 0
    out = capsys.readouterr().out
    assert "skipped: no" in out and "decision:" in out and "NO_SUGGESTION" in out


def test_serve_writes_endpoint_and_answers_with_the_token(tmp_path, cfg):
    endpoint = tmp_path / "endpoint.json"
    stop = threading.Event()
    t = threading.Thread(target=serve.main, args=([
        "--config", cfg, "--data-dir", str(tmp_path), "--synthetic", "--grant-screen",
        "--interval", "0.2", "--endpoint-file", str(endpoint)], stop), daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 30
        while not endpoint.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        info = json.loads(endpoint.read_text())
        req = urllib.request.Request(info["url"] + "/status",
                                     headers={"Authorization": f"Bearer {info['token']}"})
        with urllib.request.urlopen(req, timeout=5) as r:
            status = json.loads(r.read())
        assert status["consent"]["screen_capture"] is True and status["paused"] is False
    finally:
        stop.set()
        t.join(15)
    assert not endpoint.exists()                                   # removed on exit
