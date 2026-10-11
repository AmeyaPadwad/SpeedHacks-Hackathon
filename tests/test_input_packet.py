import base64
import json

import pytest

from pet.inputs.packet import ContextPacket, FileExcerpt, MAX_IMAGE_BYTES, WindowContext, to_darwin_search_input


def test_reviewed_packet_has_portable_bytes_and_bounded_darwin_context():
    png = b"\x89PNG\r\n\x1a\nfixture"
    files = tuple(FileExcerpt(f"src/{i}.py", 1, 200, "x" * 5000, "relevant") for i in range(8))
    packet = ContextPacket("Fix authentication", "fix auth", "/private/project", WindowContext("Code", "auth.py", 4), files, png)
    data = json.loads(packet.to_json())
    assert ContextPacket.from_json(packet.to_json()) == packet
    assert data["schema_version"] == 1
    assert base64.b64decode(data["screenshot"]["data"]) == png
    request = to_darwin_search_input(packet)
    assert len(request["context"]) == 10
    assert request["query"] == "Fix authentication"
    assert all(len(item["text"]) <= 4000 for item in request["context"] if item["type"] == "text")
    assert "/private/project" not in json.dumps(request)
    assert request["context"][-1]["mimeType"] == "image/png"
    assert "[Excerpt truncated]" in request["context"][0]["text"]


def test_partial_packet_needs_no_files_or_screenshot():
    packet = ContextPacket("Explain an error", warnings=("Screen capture denied",))
    request = to_darwin_search_input(packet)
    assert len(request["context"]) == 1
    assert "Screen capture denied" in request["context"][0]["text"]
    assert json.loads(packet.to_json())["screenshot"] is None


def test_selection_status_roundtrips_and_older_exports_remain_readable():
    packet = ContextPacket("Describe macos.py", file_selection_status="selected", file_selection_model="openai/gpt-oss-20b", file_selection_provider="groq")
    assert ContextPacket.from_json(packet.to_json()) == packet
    older = packet.to_dict()
    older.pop("file_selection_status")
    older.pop("file_selection_model")
    older.pop("file_selection_provider")
    loaded = ContextPacket.from_json(json.dumps(older))
    assert loaded.file_selection_status == "unknown" and loaded.file_selection_model == ""
    assert loaded.file_selection_provider == ""


def test_message_destination_roundtrips_and_old_packets_keep_automatic_routing():
    packet = ContextPacket("Summarize the code", message_platform="discord", message_recipient="Sarah")
    assert ContextPacket.from_json(packet.to_json()) == packet
    older = packet.to_dict()
    older.pop("message_platform")
    older.pop("message_recipient")
    loaded = ContextPacket.from_json(json.dumps(older))
    assert loaded.message_platform == "" and loaded.message_recipient == ""


@pytest.mark.parametrize("kwargs", [{"task": " "}, {"task": "fix", "schema_version": 2},
    {"task": "fix", "screenshot_png": b"not png"},
    {"task": "fix", "screenshot_png": b"\x89PNG\r\n\x1a\n" + b"x" * MAX_IMAGE_BYTES},
    {"task": "fix", "files": (FileExcerpt("a", 1, 1, "a"),) * 9},
    {"task": "fix", "message_platform": "sms"},
    {"task": "fix", "message_platform": []},
    {"task": "fix", "message_recipient": 5}])
def test_invalid_packet_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ContextPacket(**kwargs)
