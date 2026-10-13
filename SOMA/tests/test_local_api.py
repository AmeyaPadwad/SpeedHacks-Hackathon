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


def test_build_memory_script_requires_grant(tmp_path: Path):
    import io

    from scripts.build_memory import main
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "a.md").write_text("Prefer pytest -x when tests fail", encoding="utf-8")
    data = tmp_path / "data"
    out = io.StringIO()
    assert main(["--folder", str(folder), "--data-dir", str(data)], out) == 2
    assert "--grant" in out.getvalue()
    out = io.StringIO()
    assert main(["--folder", str(folder), "--data-dir", str(data), "--grant", "--list"], out) == 0
    assert "added=1" in out.getvalue() and "a.md" in out.getvalue()
    mem_id = out.getvalue().split("memories\n", 1)[1].split()[0]
    out = io.StringIO()
    assert main(["--data-dir", str(data), "--export", str(tmp_path / "x.json"), "--forget", mem_id], out) == 0
    assert "verified=True" in out.getvalue()
    out = io.StringIO()
    assert main(["--folder", str(Path.home()), "--data-dir", str(data), "--grant"], out) == 2


# ------------------------------------------------------------ companion UI endpoints


def wait_indexed(api, token, timeout=30.0):
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _, body = call(api, "GET", "/folders", token)
        if not body["indexing"]:
            return body
        time.sleep(0.05)
    raise AssertionError("indexing did not finish")


def test_folders_add_index_list_and_remove(env, tmp_path):
    rt, mem, api, _ = env
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "retries.md").write_text("HTTP retries with exponential backoff and timeouts", encoding="utf-8")
    code, body = call(api, "POST", "/folders", api.token, {"path": str(folder)})
    assert code == 200 and body["indexing"] is True
    listing = wait_indexed(api, api.token)
    assert listing["folders"] == [{"path": str(folder.resolve()), "memories": 1}]
    assert listing["last_report"]["added"] == 1
    assert rt.consent.allowed(Scope.FOLDER_INDEXING)

    code, body = call(api, "POST", "/folders/remove", api.token, {"path": str(folder)})
    assert code == 200 and body["forgotten"] == 1
    _, listing = call(api, "GET", "/folders", api.token)
    assert listing["folders"] == [] and mem.store.count_items() == 0


def test_folders_refuse_bad_input(env, tmp_path):
    _, _, api, _ = env
    assert call(api, "POST", "/folders", api.token, {"path": str(Path.home())})[0] == 400   # whole home dir
    assert call(api, "POST", "/folders", api.token, {"path": str(tmp_path / "missing")})[0] == 400
    assert call(api, "POST", "/folders", api.token, {})[0] == 400
    assert call(api, "POST", "/folders", None, {"path": str(tmp_path)})[0] == 401


def test_trigger_needs_a_scheduler_and_no_pause(env):
    rt, _, api, _ = env

    class Sched:
        n = 0

        def trigger_now(self):
            Sched.n += 1

        def pause(self):
            rt.pause()

        def resume(self):
            rt.resume()

    assert call(api, "POST", "/trigger", api.token, {})[1] == {"triggered": False}   # no scheduler
    api.scheduler = Sched()
    assert call(api, "POST", "/trigger", api.token, {})[1] == {"triggered": True} and Sched.n == 1
    call(api, "POST", "/pause", api.token, {})
    assert call(api, "POST", "/trigger", api.token, {})[1] == {"triggered": False}


def test_suggestion_carries_openable_paths_only_inside_allowed_folders(env, tmp_path):
    from soma.api.local_api import _memory_source
    from soma.types import MemoryItem, SourceType
    rt, mem, api, _ = env
    inside = tmp_path / "ok"
    inside.mkdir()
    (inside / "a.md").write_text("x", encoding="utf-8")
    rt.consent.add_folder(inside)
    rt.consent.grant(Scope.FOLDER_INDEXING)
    good = mem.store.upsert_item(MemoryItem(id="m1", source_type=SourceType.DOCUMENT, user_approved=True,
                                            source_uri=str((inside / "a.md").resolve()), text_summary="a"))
    bad = mem.store.upsert_item(MemoryItem(id="m2", source_type=SourceType.DOCUMENT, user_approved=True,
                                           source_uri=str(tmp_path / "elsewhere.md"), text_summary="b"))
    assert _memory_source(rt, good.id) == str((inside / "a.md").resolve())
    assert _memory_source(rt, bad.id) is None and _memory_source(rt, None) is None
