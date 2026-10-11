"""Approved collector inputs continue through the real drafting and dispatch pipeline."""
import json
import threading
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QBuffer, QIODevice, Qt
from PySide6.QtGui import QImage

from pet import dispatch
from pet.agents import AgentResult, build_prompt
from pet.analyst import Analysis, Analyst
from pet.brain import Brain, BrainError, Draft
from pet.context import Context, reviewed_file_text
from pet.inputs.companion import CodingOverlay, InputCompanion
from pet.inputs.macos import WindowCapture
from pet.inputs.packet import ContextPacket, FileExcerpt, WindowContext
from pet.inputs.service import CollectionService
from pet.inputs.ui import InputPanel, build_input_gathering
from pet.pipeline import Pipeline


def png():
    image = QImage(1800, 900, QImage.Format_RGB32)
    image.fill(Qt.green)
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    image.save(buf, "PNG")
    return bytes(buf.data())


@pytest.fixture
def flow(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr("pet.pipeline.MIN_INTERVAL_S", 0)
    monkeypatch.setattr("pet.inputs.companion.foreground_target", lambda: None)
    calls, posts = [], []
    result = {"platform": "slack", "recipient": "Sarah", "subject": "", "message": "Auth calls the repository."}

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))])

    def post(url, json, timeout):
        posts.append((url, json))
        return SimpleNamespace(status_code=200)

    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.invalid/slack")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/discord")
    monkeypatch.setenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
    monkeypatch.setattr(dispatch.requests, "post", post)
    monkeypatch.setattr(dispatch, "load_contacts", lambda: {})
    service = CollectionService(tmp_path, capture=lambda: WindowCapture(WindowContext("Code", "auth.py", 1), png()),
                                spawn=lambda f: f())
    panel = InputPanel(service)
    companion = InputCompanion(service, panel, CodingOverlay(tray=False))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    pipeline = Pipeline(companion.overlay, Brain(client, model="openai/gpt-oss-20b"),
                        capture=lambda: pytest.fail("Approved context must not be recaptured"),
                        read_file=lambda: pytest.fail("Excluded IDE context must not be read"),
                        use_voice=False, connect_inputs=False, spawn=lambda f: f())
    companion.bind_pipeline(pipeline)
    yield SimpleNamespace(service=service, panel=panel, pet=companion, pipeline=pipeline,
                          calls=calls, posts=posts, result=result)
    service.close()
    companion.close()


def review(flow, command="Send Sarah a Slack summary of these files"):
    flow.pet.type_task()
    flow.pet._typed(command)
    flow.service.files = (
        FileExcerpt("auth.py", 20, 21, "authenticate = repository.lookup()\nreturn authenticate"),
        FileExcerpt("repository.py", 80, 80, "lookup = database.get_user"),
        FileExcerpt("excluded.py", 1, 1, "excluded_content"),
    )
    flow.service._state("reviewing")
    flow.panel.files.item(2).setCheckState(Qt.Unchecked)


@pytest.mark.parametrize("platform", ["slack", "discord"])
def test_use_context_drafts_then_send_posts_exactly_once(flow, platform):
    flow.result["platform"] = platform
    review(flow)
    flow.panel.task.setPlainText(f"Send Sarah a {platform} summary of these two files")
    assert flow.calls == [] and flow.posts == []
    flow.panel.use.click()
    flow.panel.use.click()
    assert len(flow.calls) == 1 and not flow.panel.isVisible()
    assert flow.pet.overlay.bubble.primary.text() == "Send" and flow.posts == []
    parts = flow.calls[0]["messages"][1]["content"]
    text = "\n".join(p["text"] for p in parts if p["type"] == "text")
    assert f"Send Sarah a {platform}" in text
    assert "auth.py (lines 20-21)" in text and "repository.py (lines 80-80)" in text
    assert "database.get_user" in text and "excluded_content" not in text
    assert flow.calls[0]["model"] == "qwen/qwen3.8-27b"
    url = next(p["image_url"]["url"] for p in parts if p["type"] == "image_url")
    assert url.startswith("data:image/jpeg;base64,")
    assert not flow.pipeline.submit_context(flow.service.packet)
    flow.pet.overlay.bubble.primary.click()
    flow.pet.overlay.bubble.primary.click()
    assert len(flow.posts) == 1 and flow.posts[0][0].endswith(platform)
    assert "Auth calls the repository." in str(flow.posts[0][1])
    assert not flow.pipeline.busy
    flow.pet.type_task()
    assert flow.service.state == "editing" and flow.pet.overlay.bubble.asking


def test_unchecked_screenshot_and_files_are_never_uploaded(flow):
    review(flow)
    flow.panel.include_image.setChecked(False)
    for index in range(flow.panel.files.count()):
        flow.panel.files.item(index).setCheckState(Qt.Unchecked)
    flow.panel.use.click()
    parts = flow.calls[0]["messages"][1]["content"]
    assert all(p["type"] == "text" for p in parts)
    assert "database.get_user" not in str(parts)
    assert flow.calls[0]["model"] == "openai/gpt-oss-20b"
    assert flow.calls[0]["reasoning_effort"] == "low"
    flow.pet.overlay.bubble.cancel.click()
    assert flow.posts == [] and not flow.pipeline.pending


@pytest.mark.parametrize("platform", ["slack", "discord"])
def test_preview_destination_creates_a_channel_message_without_a_spoken_platform(flow, platform):
    review(flow, "Explain how these files work")
    flow.result.update(platform="none", recipient="", message="Auth calls the repository.")
    flow.panel.destination.setCurrentIndex(flow.panel.destination.findData(platform))
    flow.panel.use.click()
    assert flow.service.packet.message_platform == platform
    assert flow.pet.overlay.bubble.destination.currentData() == platform
    assert flow.pet.overlay.bubble.delivery.isVisible()
    assert flow.pet.overlay.bubble.primary.text() == "Send"
    assert f"explicitly chose {platform}" in flow.calls[0]["messages"][0]["content"]
    flow.pet.overlay.bubble.primary.click()
    assert flow.posts[0][0].endswith(platform)
    key = "text" if platform == "slack" else "content"
    assert flow.posts[0][1][key] == "Auth calls the repository."
    assert f"configured {platform.capitalize()} channel" in flow.pet.overlay.bubble.body.text()


def test_recipient_in_preview_overrides_the_model_recipient(flow):
    review(flow)
    flow.result["recipient"] = "Bob"
    flow.panel.destination.setCurrentIndex(flow.panel.destination.findData("discord"))
    flow.panel.recipient.setText("Sarah")
    flow.panel.use.click()
    assert next(iter(flow.pipeline.pending.values())).recipient == "Sarah"
    assert flow.service.packet.message_recipient == "Sarah"


def test_message_preview_can_switch_channels_without_another_ai_call(flow):
    review(flow)
    flow.panel.use.click()
    pid = flow.pet.overlay.bubble.proposal.id
    body = flow.pet.overlay.bubble.body.text()
    flow.pet.overlay.bubble.destination.setCurrentIndex(flow.pet.overlay.bubble.destination.findData("discord"))
    assert flow.pet.overlay.bubble.proposal.id == pid
    assert flow.pet.overlay.bubble.body.text() == body
    assert len(flow.calls) == 1 and flow.posts == []
    flow.pet.overlay.bubble.primary.click()
    assert flow.posts[0][0].endswith("discord")


def test_unconfigured_channel_disables_send_until_another_channel_is_selected(flow):
    flow.pipeline._configured = lambda platform: platform == "discord"
    review(flow)
    flow.panel.use.click()
    assert not flow.pet.overlay.bubble.primary.isEnabled()
    assert "not configured" in flow.pet.overlay.bubble.delivery_status.text()
    flow.pet.overlay.bubble.primary.click()
    assert flow.posts == []
    flow.pet.overlay.bubble.destination.setCurrentIndex(flow.pet.overlay.bubble.destination.findData("discord"))
    assert flow.pet.overlay.bubble.primary.isEnabled()
    flow.pet.overlay.bubble.primary.click()
    assert flow.posts[0][0].endswith("discord")


def test_failed_send_keeps_draft_and_allows_switching_channels(flow, monkeypatch):
    attempts = []
    def post(url, json, timeout):
        attempts.append((url, json))
        return SimpleNamespace(status_code=403 if len(attempts) == 1 else 200)
    monkeypatch.setattr(dispatch.requests, "post", post)
    review(flow)
    flow.panel.use.click()
    old_pid = flow.pet.overlay.bubble.proposal.id
    flow.pet.overlay.bubble.primary.click()
    assert "403" in flow.pet.overlay.bubble.delivery_status.text()
    assert flow.pet.overlay.bubble.body.text() == "Auth calls the repository."
    assert flow.pet.overlay.bubble.primary.text() == "Retry"
    assert flow.pipeline.pending and not flow.pipeline.busy
    assert flow.pet.overlay.bubble.proposal.id != old_pid
    flow.pet.overlay.approved.emit(old_pid)
    assert len(attempts) == 1
    flow.pet.overlay.bubble.destination.setCurrentIndex(flow.pet.overlay.bubble.destination.findData("discord"))
    flow.pet.overlay.bubble.primary.click()
    assert len(attempts) == 2 and attempts[1][0].endswith("discord")
    assert flow.pipeline.pending == {} and not flow.pipeline.busy


def test_escape_discards_late_agent_results(flow):
    review(flow)
    jobs = []
    flow.pipeline._spawn = jobs.append
    flow.panel.use.click()
    assert flow.pipeline.busy
    flow.service.cancel_pressed.emit()
    jobs.pop()()
    assert not flow.pipeline.busy and not flow.pipeline.pending and not flow.posts
    assert flow.service.state == "idle"
    assert flow.pet.overlay.bubble.primary.text() == "Record a task"


def test_hotkey_does_not_replace_a_pending_draft_or_inflight_send(flow):
    review(flow)
    flow.panel.use.click()
    proposal_id = flow.pet.overlay.bubble.proposal.id
    flow.service.hotkey_pressed.emit()
    assert flow.pet.overlay.bubble.proposal.id == proposal_id
    jobs = []
    flow.pipeline._spawn = jobs.append
    flow.pet.overlay.bubble.primary.click()
    assert flow.pipeline.busy
    flow.service.hotkey_pressed.emit()
    assert flow.service.state == "ready" and len(jobs) == 1
    jobs.pop()()
    assert len(flow.posts) == 1 and not flow.pipeline.busy


def test_pause_drops_late_draft_and_allows_a_new_collection(flow):
    review(flow)
    jobs = []
    flow.pipeline._spawn = jobs.append
    flow.panel.use.click()
    flow.pet.overlay.set_paused(True)
    jobs.pop()()
    assert flow.pipeline.pending == {} and flow.posts == []
    flow.pet.overlay.set_paused(False)
    flow.pet.type_task()
    assert flow.service.state == "editing"


def test_coding_request_reuses_every_reviewed_file_in_analysis_and_cli_task(flow):
    review(flow, "Ask Claude why authentication fails")
    flow.result.update(platform="agent", recipient="", message="Investigate authentication")
    contexts, tasks = [], []

    def analyze(ctx):
        contexts.append(ctx)
        return Analysis("Auth fails", ("auth.py:20",), "Bad lookup", "Inspect auth", .8, "ai")

    flow.pipeline.analyst = SimpleNamespace(analyze=analyze)
    agent = SimpleNamespace(name="claude", label="Claude Code", run=lambda task: (
        tasks.append(task) or AgentResult("claude", "Proposed fix")))
    flow.pipeline._pick_agent = lambda _: agent
    flow.panel.use.click()
    assert tasks == [] and len(contexts[0].reviewed_files) == 2
    assert "repository.py" in Analyst._text(contexts[0])
    flow.pet.overlay.bubble.primary.click()
    prompt = build_prompt(tasks[0])
    assert "auth.py (lines 20-21)" in prompt and "repository.py (lines 80-80)" in prompt
    assert "excluded_content" not in prompt and "base64" not in prompt
    assert tasks[0].workspace == str(flow.service.project)
    assert flow.posts == []


def test_prompt_budget_keeps_every_checked_file():
    files = tuple(FileExcerpt(f"source{i}.py", 100, 200, "x" * 4000) for i in range(8))
    text = reviewed_file_text(Context(None, None, files))
    assert len(text) <= 12_000 and "truncated for agent prompt" in text
    assert all(f"source{i}.py (lines 100-200)" in text for i in range(8))


def test_agent_error_leaves_a_recoverable_result(flow):
    review(flow)
    def fail(*args):
        raise BrainError("Groq rate limit hit. Wait a moment and try again.")
    flow.pipeline.brain.draft = fail
    flow.panel.use.click()
    assert "rate limit" in flow.pet.overlay.bubble.body.text()
    assert flow.pipeline.pending == {} and not flow.pipeline.busy and flow.posts == []
    flow.pet.type_task()
    assert flow.service.state == "editing"


@pytest.mark.parametrize("error, message", [
    (BrainError("Groq rate limit hit. Wait a moment and try again."), "Groq rate limit hit"),
    (RuntimeError("private error details"), "Unexpected error: RuntimeError"),
])
def test_worker_error_is_delivered_after_exception_scope_exits(flow, qapp, capsys, error, message):
    review(flow)

    def fail(*args):
        raise error

    def spawn(work):
        worker = threading.Thread(target=work)
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive()

    flow.pipeline.brain.draft = fail
    flow.pipeline._spawn = spawn
    flow.panel.use.click()
    # Delivery is queued to Qt. The worker has already exited its except block.
    assert flow.pipeline.busy
    qapp.processEvents()
    assert message in flow.pet.overlay.bubble.body.text()
    assert "private error details" not in flow.pet.overlay.bubble.body.text()
    assert not flow.pipeline.busy and not flow.pipeline.pending and flow.posts == []
    assert "Traceback" not in capsys.readouterr().err


def test_cancel_discards_queued_worker_failure(flow, qapp):
    review(flow)

    def fail(*args):
        raise BrainError("Late Groq failure")

    def spawn(work):
        worker = threading.Thread(target=work)
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive()

    flow.pipeline.brain.draft = fail
    flow.pipeline._spawn = spawn
    flow.panel.use.click()
    flow.service.cancel_pressed.emit()
    qapp.processEvents()
    assert flow.pet.overlay.bubble.primary.text() == "Record a task"
    assert "Late Groq failure" not in flow.pet.overlay.bubble.body.text()
    assert not flow.pipeline.busy and not flow.pipeline.pending and flow.posts == []


def test_application_build_wires_packet_to_the_existing_pipeline(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr("pet.inputs.ui.Hotkeys.start", lambda *args: True)
    monkeypatch.setattr("pet.inputs.companion.foreground_target", lambda: None)
    ctx = build_input_gathering(qapp, tmp_path)
    calls = []
    ctx.pipeline.brain = SimpleNamespace(draft=lambda task, context: (
        calls.append((task, context)) or Draft("none", "", "", "Here is the explanation.")))
    ctx.pipeline._spawn = lambda f: f()
    ctx.service._spawn = lambda f: f()
    ctx.service._capture = lambda: WindowCapture()
    try:
        ctx.companion.type_task()
        ctx.companion._typed("Explain this project")
        ctx.panel.use.click()
        assert len(calls) == 1 and calls[0][0] == "Explain this project"
        assert ctx.pipeline.overlay is ctx.companion.overlay
        assert "Here is the explanation" in ctx.overlay.bubble.body.text()
    finally:
        ctx.service.close()
        ctx.companion.close()
        ctx.hotkeys.close()
        ctx.results.close()
