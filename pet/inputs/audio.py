"""Press-to-start/stop recording and cancellable local Whisper transcription."""
import io
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

MAX_RECORD_SECONDS = 60
MICROPHONE_START_TIMEOUT = 15
MICROPHONE_STOP_TIMEOUT = 5


class InputError(Exception):
    pass


class Recorder:
    def record(self, stop: threading.Event, cancel: threading.Event, *, on_started=None) -> bytes:
        """Isolate native microphone calls so a stuck device can be terminated."""
        if cancel.is_set() or stop.is_set():
            return b""
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        child_stop = context.Event()
        process = context.Process(target=_record_worker, args=(sender, child_stop), daemon=True)
        started = False
        ready = False
        deadline = time.monotonic() + MICROPHONE_START_TIMEOUT
        stopping_deadline = None
        try:
            process.start()
            started = True
            sender.close()
            while True:
                if cancel.is_set():
                    return b""
                now = time.monotonic()
                if stop.is_set() or (ready and now >= deadline):
                    if stopping_deadline is None:
                        child_stop.set()
                        stopping_deadline = now + MICROPHONE_STOP_TIMEOUT
                if receiver.poll(0.05):
                    kind, payload = receiver.recv()
                    if kind == "started":
                        ready = True
                        deadline = time.monotonic() + MAX_RECORD_SECONDS
                        if on_started is not None:
                            on_started()
                    elif kind == "audio":
                        return payload
                    elif kind == "error":
                        raise InputError(payload)
                if stopping_deadline is not None and time.monotonic() >= stopping_deadline:
                    raise InputError("Microphone did not stop within 5 seconds. The recording was cancelled; try again or type the task.")
                if not ready and stopping_deadline is None and time.monotonic() >= deadline:
                    raise InputError("Microphone did not start within 15 seconds. Check microphone access for your terminal/app, then retry or type the task.")
                if not process.is_alive():
                    # A final message may arrive just as the child exits.
                    if receiver.poll():
                        continue
                    raise InputError("Microphone stopped unexpectedly. Try recording again or type the task.")
        except InputError:
            raise
        except (OSError, EOFError) as exc:
            raise InputError("Microphone unavailable. Check microphone access or type the task.") from exc
        finally:
            if started:
                # Never let native stream teardown hold the collection worker.
                process.join(timeout=0.2)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=0.5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=0.5)
                if not process.is_alive():
                    process.close()
            receiver.close()
            sender.close()


def _record_worker(sender, stop):
    try:
        wav = _record_microphone(stop, threading.Event(), on_started=lambda: sender.send(("started", None)))
        sender.send(("audio", wav))
    except InputError as exc:
        sender.send(("error", str(exc)))
    except Exception:
        sender.send(("error", "Microphone unavailable. Try recording again or type the task."))
    finally:
        sender.close()


def _record_microphone(stop, cancel, *, on_started=None) -> bytes:
    """Runs in the child process; no Qt widgets are touched."""
    try:
        import sounddevice as sd
        # Use the device's native rate, then resample to 16 kHz for whisper-cli.
        rate = int(sd.query_devices(kind="input")["default_samplerate"])
        chunks = []
        remaining = rate * MAX_RECORD_SECONDS * 2
        overflowed = False
        first_frame = threading.Event()

        def callback(indata, frames, time_info, status):
            nonlocal remaining, overflowed
            if status:
                overflowed = True
            if cancel.is_set() or stop.is_set():
                raise sd.CallbackAbort
            chunk = bytes(indata)[:remaining]
            chunks.append(chunk)
            first_frame.set()
            remaining -= len(chunk)
            if remaining <= 0:
                stop.set()
                raise sd.CallbackAbort

        notified = False
        stream = sd.RawInputStream(samplerate=rate, channels=1, dtype="int16", callback=callback)
        try:
            stream.start()
            started = time.monotonic()
            while not stop.wait(0.05) and not cancel.is_set():
                if first_frame.is_set() and not notified:
                    notified = True
                    if on_started is not None:
                        on_started()
                if time.monotonic() - started >= MAX_RECORD_SECONDS:
                    stop.set()
                if not stream.active:
                    break
        finally:
            try:
                stream.abort()
            finally:
                stream.close()
        if cancel.is_set():
            return b""
        if overflowed:
            raise InputError("Microphone audio dropped frames. Record again or type the task.")
        pcm = b"".join(chunks)
        if not pcm or len(pcm) < rate // 2:
            raise InputError("Recording was too short. Record again or type the task.")
        if rate != 16_000:
            # audioop is part of Python 3.11/3.12; input mode documents these versions.
            import audioop
            pcm, _ = audioop.ratecv(pcm, 2, 1, rate, 16_000, None)
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16_000)
            wav.writeframes(pcm)
        return output.getvalue()
    except InputError:
        raise
    except Exception as e:
        raise InputError("Microphone unavailable. Enable microphone access or type the task.") from e


class LocalWhisper:
    def __init__(self, executable=None, model=None, timeout=120):
        root = Path(__file__).resolve().parents[2]
        bundled = root / ".whisper-runtime/build/bin/whisper-cli"
        self.executable = executable or os.environ.get("FLOWSTATE_WHISPER_BIN") or (
            str(bundled) if bundled.is_file() else "whisper-cli")
        self.model = Path(model or os.environ.get("FLOWSTATE_WHISPER_MODEL") or root / "models/ggml-small.en.bin").expanduser()
        self.timeout = timeout

    def transcribe(self, wav: bytes, cancel: threading.Event) -> str:
        executable = shutil.which(self.executable)
        if executable is None or not self.model.is_file():
            raise InputError("Local Whisper needs whisper-cli and small.en. See README setup; you can type instead.")
        if not wav:
            raise InputError("No audio recorded. Type the task or try recording again.")
        if cancel.is_set():
            return ""
        # whisper-cli requires paths. Only this private temp directory holds audio/output.
        with tempfile.TemporaryDirectory(prefix="flowstate-whisper-") as folder:
            audio = Path(folder) / "input.wav"
            audio.write_bytes(wav)
            output = Path(folder) / "transcript"
            args = [executable, "-m", str(self.model.resolve()), "-f", str(audio),
                    "-l", "en", "-nt", "-otxt", "-of", str(output)]
            try:
                process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                deadline = time.monotonic() + self.timeout
                try:
                    while process.poll() is None:
                        if cancel.wait(0.05) or time.monotonic() >= deadline:
                            process.terminate()
                            try:
                                process.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            if cancel.is_set():
                                return ""
                            raise InputError("Local transcription timed out. Type the task or use a smaller model.")
                    if process.returncode != 0:
                        raise InputError("Local Whisper failed. Check executable/model setup or type the task.")
                    text = output.with_suffix(".txt").read_text(encoding="utf-8").strip()
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
            except InputError:
                raise
            except (OSError, UnicodeError) as e:
                raise InputError("Local transcription failed. Type the task or check Whisper setup.") from e
        if not text:
            raise InputError("No words recognized. Type the task or record again.")
        return text
