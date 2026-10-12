"""Local API: loopback only, bearer auth, endpoints, memory deletion."""

from __future__ import annotations

import http.client
import json
from pathlib import Path

import pytest

from soma.api.local_api import LocalAPIError, LocalAPIServer
from soma.config import PolicyConfig, SomaConfig, small_model_config
from soma.consent import ConsentManager, Scope
from soma.perception.ocr import SyntheticOCREngine
from soma.perception.synthetic import render_scenario
from soma.runtime.factory import open_memory
from soma.runtime.pipeline import SomaRuntime
from soma.types import ScreenFrame


@pytest.fixture
def env(tmp_path: Path):
    cfg = SomaConfig(device="cpu", model=small_model_config())
    cfg.policy = PolicyConfig(relevance_threshold=0.0, margin_over_abstain=-1.0, cooldown_s=0.0)
    consent = ConsentManager()
    consent.grant(Scope.SCREEN_CAPTURE)
    consent.grant(Scope.FEEDBACK_STORAGE)
    mem = open_memory(cfg, consent, tmp_path / "data")
    rt = SomaRuntime(cfg, consent, ocr=SyntheticOCREngine(), memory=mem)
    token_file = tmp_path / "secret" / "api.token"
    api = LocalAPIServer(rt, port=0, token_path=token_file).start()
    yield rt, mem, api, token_file
    api.stop()
    mem.close()


def call(api, method, path, token=None, body=None, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", api.port, timeout=10)
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if host is not None:
        headers["Host"] = host
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=headers)
    resp = conn.getresponse()
    payload = json.loads(resp.read() or b"{}")
    conn.close()
    return resp.status, payload


def test_refuses_non_loopback_bind():
    for host in ("0.0.0.0", "192.168.1.5", "example.com", ""):
        with pytest.raises(LocalAPIError):
            LocalAPIServer(object(), host=host)


def test_auth_required_and_token_written_then_removed(env):
    rt, mem, api, token_file = env
    assert api.url.startswith("http://127.0.0.1:") and api.port > 0
    assert token_file.read_text() == api.token and len(api.token) >= 32
    for method, path in (("GET", "/status"), ("GET", "/suggestion"), ("GET", "/memories"),
                         ("POST", "/pause"), ("DELETE", "/memories/x")):
        assert call(api, method, path)[0] == 401
        assert call(api, method, path, token="wrong")[0] == 401
    assert not rt.paused
    api.stop()
    assert not token_file.exists()


def test_rebinding_host_header_rejected(env):
    _, _, api, _ = env
    assert call(api, "GET", "/status", api.token, host="evil.example:80")[0] == 403
    assert call(api, "GET", "/status", api.token, host=f"localhost:{api.port}")[0] == 200


def test_status_pause_resume(env):
    rt, _, api, _ = env
    code, st = call(api, "GET", "/status", api.token)
    assert code == 200 and st["paused"] is False and st["consent"]["screen_capture"] is True
    assert call(api, "POST", "/pause", api.token)[1] == {"paused": True}
    assert rt.paused and rt.consent.paused
    assert call(api, "GET", "/status", api.token)[1]["paused"] is True
    assert call(api, "POST", "/resume", api.token)[1] == {"paused": False}
    assert not rt.paused


def test_suggestion_and_feedback(env):
    rt, mem, api, _ = env
    assert call(api, "GET", "/suggestion", api.token)[1] == {"available": False, "silent": True}
    image, obs, _ = render_scenario("terminal_traceback", seed=3, size=(640, 400))
    r = rt.process_frame(ScreenFrame(image=image, window=obs.window, version=1))
    assert not r.suggestion.is_silent
    code, s = call(api, "GET", "/suggestion", api.token)
    assert code == 200 and s["id"] == r.suggestion.id and s["text"] == r.suggestion.candidate.text
    assert "NO_SUGGESTION" in s["distribution"] and s["valid"] is True
    # no raw OCR dump in the payload
    assert not ({"observation", "elements", "ocr", "screen_text"} & set(s))
    assert call(api, "POST", "/feedback", api.token, {"suggestion_id": "nope", "outcome": "accepted"})[0] == 404
    assert call(api, "POST", "/feedback", api.token, {"suggestion_id": s["id"], "outcome": "meh"})[0] == 400
    code, out = call(api, "POST", "/feedback", api.token, {"suggestion_id": s["id"], "outcome": "rejected"})
    assert code == 200 and out["outcome"] == "rejected"
    assert [e.outcome.value for e in mem.store.list_feedback()] == ["rejected"]
    assert rt.gate.is_duplicate(r.suggestion.candidate.key)
    assert call(api, "GET", "/suggestion", api.token)[1]["valid"] is False


def test_memories_list_and_delete(env):
    rt, mem, api, _ = env
    item = mem.ingestor.add_preference("Prefer short explanations")
    code, data = call(api, "GET", "/memories", api.token)
    assert code == 200 and [m["id"] for m in data["memories"]] == [item.id]
    code, out = call(api, "DELETE", f"/memories/{item.id}", api.token)
    assert code == 200 and out == {"deleted": True, "verified": True, "id": item.id}
    assert mem.store.get_item(item.id) is None and item.id not in mem.index
    assert call(api, "GET", "/memories", api.token)[1]["memories"] == []
    assert call(api, "DELETE", f"/memories/{item.id}", api.token)[0] == 404


def test_unknown_routes_and_bad_bodies(env):
    _, _, api, _ = env
    assert call(api, "GET", "/execute", api.token)[0] == 404
    assert call(api, "POST", "/run", api.token, {})[0] == 404
    assert call(api, "PUT", "/status", api.token)[0] == 405
    conn = http.client.HTTPConnection("127.0.0.1", api.port, timeout=10)
    conn.request("POST", "/feedback", body=b"not json",
                 headers={"Authorization": f"Bearer {api.token}"})
    assert conn.getresponse().status == 400
    conn.close()
