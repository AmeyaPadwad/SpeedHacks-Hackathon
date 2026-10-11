import io
import multiprocessing
import sys
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from pet.inputs.audio import InputError, LocalWhisper, Recorder, _record_microphone


def test_native_microphone_rate_is_normalized_and_duration_is_bounded(monkeypatch):
    stop, cancel = threading.Event(), threading.Event()
    observed = {}

    class Stream:
        active = True

        def __init__(self, **kwargs):
            observed.update(kwargs)

        def start(self):
            try:
                observed["callback"](b"\x01\x00" * (48000 * 61), 48000 * 61, None, False)
            except RuntimeError:
                pass  # PortAudio consumes CallbackAbort internally.

        def abort(self):
            observed["aborted"] = True

        def close(self):
            observed["closed"] = True

        def stop(self):
            pytest.fail("Input capture must not use graceful stream.stop()")

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(query_devices=lambda **kwargs: {"default_samplerate": 48000},
        RawInputStream=Stream, CallbackAbort=RuntimeError))
    wav = _record_microphone(stop, cancel)
    with wave.open(io.BytesIO(wav)) as f:
        assert f.getframerate() == 16000 and f.getnchannels() == 1 and f.getsampwidth() == 2
        assert f.getnframes() <= 16000 * 60
    assert stop.is_set()
    assert observed["aborted"] and observed["closed"]


def test_recording_ready_is_reported_only_after_audio_frames(monkeypatch):
    events = []

    class Stream:
        active = False

        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def start(self):
            events.append("stream started")
            self.callback(b"\x01\x00" * 8000, 8000, None, False)
            events.append("audio frames")

        def abort(self):
            events.append("abort")

        def close(self):
            events.append("close")

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(
        query_devices=lambda **kwargs: {"default_samplerate": 16000}, RawInputStream=Stream))
    _record_microphone(threading.Event(), threading.Event(), on_started=lambda: events.append("ready"))
    assert events == ["stream started", "audio frames", "ready", "abort", "close"]


def _stuck_microphone_worker(sender, stop):
    sender.send(("started", None))
    stop.wait()
    threading.Event().wait(30)  # Simulate a native stop/close that never returns.


def _unresponsive_microphone_worker(sender, stop):
    threading.Event().wait(30)  # Simulate microphone initialization blocking.


def _successful_microphone_worker(sender, stop):
    sender.send(("started", None))
    stop.wait()
    sender.send(("audio", b"recorded wav"))
    threading.Event().wait(30)  # Result delivered; device teardown is stuck.


@pytest.mark.parametrize("worker, cancel_recording, expected", [
    (_stuck_microphone_worker, False, "did not stop"),
    (_unresponsive_microphone_worker, False, "did not start"),
    (_stuck_microphone_worker, True, None),
    (_successful_microphone_worker, False, None),
])
def test_native_worker_hangs_are_bounded_and_process_is_reaped(monkeypatch, worker, cancel_recording, expected):
    monkeypatch.setattr("pet.inputs.audio._record_worker", worker)
    monkeypatch.setattr("pet.inputs.audio.MICROPHONE_START_TIMEOUT", 2)
    monkeypatch.setattr("pet.inputs.audio.MICROPHONE_STOP_TIMEOUT", 0.1)
    stop, cancel = threading.Event(), threading.Event()
    previous_children = {p.pid for p in multiprocessing.active_children()}

    def ready():
        (cancel if cancel_recording else stop).set()

    started = time.monotonic()
    if expected:
        with pytest.raises(InputError, match=expected):
            Recorder().record(stop, cancel, on_started=ready)
    else:
        result = Recorder().record(stop, cancel, on_started=ready)
        assert result == (b"" if cancel_recording else b"recorded wav")
    assert time.monotonic() - started < 5
    assert {p.pid for p in multiprocessing.active_children()} <= previous_children


def test_missing_local_runtime_and_empty_audio_have_recoverable_errors(tmp_path):
    with pytest.raises(InputError, match="Local Whisper needs"):
        LocalWhisper(executable="no-such-whisper", model=tmp_path / "missing").transcribe(b"wav", threading.Event())


def test_transcription_temp_files_are_deleted_and_cancel_stops_process(tmp_path, monkeypatch):
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    folders = []
    monkeypatch.setattr("pet.inputs.audio.shutil.which", lambda name: "/fake/whisper")

    class Process:
        returncode = 0

        def __init__(self, args, **kwargs):
            folder = Path(args[args.index("-f") + 1]).parent
            folders.append(folder)
            assert (folder / "input.wav").read_bytes() == b"wav"
            Path(args[args.index("-of") + 1] + ".txt").write_text("Fix the login bug")

        def poll(self):
            return 0

    monkeypatch.setattr("pet.inputs.audio.subprocess.Popen", Process)
    assert LocalWhisper(model=model).transcribe(b"wav", threading.Event()) == "Fix the login bug"
    assert not folders[0].exists()

    cancel = threading.Event()
    stopped = []

    class SlowProcess(Process):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            cancel.set()

        def poll(self):
            return 0 if stopped else None

        def terminate(self):
            stopped.append(True)

        def wait(self, **kwargs):
            return 0

    monkeypatch.setattr("pet.inputs.audio.subprocess.Popen", SlowProcess)
    assert LocalWhisper(model=model).transcribe(b"wav", cancel) == ""
    assert stopped and all(not f.exists() for f in folders)
