"""ElevenLabs speech-to-text. Transcription only: the agent's voice is never played and its replies are ignored.

Two paths, both returning the user's words for Pette's bubble:
- **Scribe** (`ELEVENLABS_API_KEY` set): record until silence locally, then one request to the
  speech-to-text API (`scribe_v1`).
- **Live** (public agent, `ELEVENLABS_AGENT_ID`): stream the mic to the Conversational AI websocket and
  take its first finished `user_transcript`. ElevenLabs' turn detection decides when you've stopped.
Audio is streamed or held in memory for one request and never saved.
"""
import base64
import json
import os
import re
import threading
import time
from dataclasses import dataclass

from .voice import VoiceError

WS_URL = "wss://api.elevenlabs.io/v1/convai/conversation"
SIGNED_URL = "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url"
STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
STT_MODEL = "scribe_v1"
RATE = 16_000
CHUNK = 1600                 # 100 ms of 16 kHz mono int16 per websocket message
START_TIMEOUT_S = 8          # nobody spoke
MAX_LISTEN_S = 25            # hard cap on one command
TAG = re.compile(r"\[[a-z ]+\]\s*")


@dataclass(frozen=True)
class Settings:
    agent_id: str = ""
    branch_id: str = ""
    api_key: str = ""

    @classmethod
    def from_env(cls) -> "Settings | None":
        """None when ElevenLabs isn't configured or FLOWSTATE_VOICE=whisper."""
        if (os.environ.get("FLOWSTATE_VOICE") or "").strip().lower() == "whisper":
            return None
        s = cls((os.environ.get("ELEVENLABS_AGENT_ID") or "").strip(),
                (os.environ.get("ELEVENLABS_BRANCH_ID") or "").strip(),
                (os.environ.get("ELEVENLABS_API_KEY") or "").strip())
        return s if (s.agent_id or s.api_key) else None

    @property
    def mode(self) -> str:
        return "scribe" if self.api_key else "live"

    def ws_url(self) -> str:
        if self.api_key and self.agent_id:
            import requests
            r = requests.get(SIGNED_URL, params={"agent_id": self.agent_id},
                             headers={"xi-api-key": self.api_key}, timeout=10)
            r.raise_for_status()
            url = r.json()["signed_url"]
        else:
            url = f"{WS_URL}?agent_id={self.agent_id}"
        return url + (f"&branch_id={self.branch_id}" if self.branch_id else "")


def scribe(wav: bytes, api_key: str, post=None) -> str:
    """Batch speech-to-text with ElevenLabs Scribe."""
    if post is None:
        import requests
        post = requests.post
    try:
        r = post(STT_URL, headers={"xi-api-key": api_key}, data={"model_id": STT_MODEL},
                 files={"file": ("command.wav", wav, "audio/wav")}, timeout=20)
    except Exception as e:
        raise VoiceError(f"Couldn't reach ElevenLabs: {type(e).__name__}") from e
    if r.status_code == 401:
        raise VoiceError("ElevenLabs rejected the API key.")
    if r.status_code >= 300:
        raise VoiceError(f"ElevenLabs transcription failed (HTTP {r.status_code}).")
    text = (r.json().get("text") or "").strip()
    if not text:
        raise VoiceError("Couldn't make out any words.")
    return text


class LiveTranscriber:
    """Streams the mic to the agent websocket and returns the first finished user transcript.

    `run()` blocks (call it from a worker thread). `cancel()` from any thread stops it early.
    The agent's audio and replies are dropped: Pette is silent and only shows your words.
    """

    def __init__(self, settings: Settings, *, connect=None, audio=None, on_level=None, clock=time.monotonic):
        self.settings = settings
        self._connect = connect or _default_connect
        self._audio = audio
        self._on_level = on_level or (lambda level: None)
        self._clock = clock
        self._stop = threading.Event()
        self._ws = None
        self._lock = threading.Lock()

    def cancel(self):
        self._stop.set()

    def run(self) -> str:
        audio = None
        try:
            try:
                self._ws = self._connect(self.settings.ws_url())
            except Exception as e:
                raise VoiceError(f"Couldn't reach ElevenLabs: {type(e).__name__}") from e
            self._send({"type": "conversation_initiation_client_data"})
            audio = self._audio or MicIn()
            try:
                audio.open()
            except OSError as e:
                raise VoiceError("Microphone unavailable.") from e
            threading.Thread(target=self._mic_loop, args=(audio,), daemon=True).start()
            return self._read_until_transcript()
        finally:
            self._stop.set()
            ws, self._ws = self._ws, None
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
            if audio is not None:
                audio.close()

    def _send(self, payload: dict):
        ws = self._ws
        if ws is None:
            return
        with self._lock:
            try:
                ws.send(json.dumps(payload))
            except Exception:
                self._stop.set()

    def _mic_loop(self, audio):
        while not self._stop.is_set():
            try:
                pcm = audio.read(CHUNK)
            except Exception:
                self._stop.set()
                return
            self._on_level(_level(pcm))
            self._send({"user_audio_chunk": base64.b64encode(pcm).decode("ascii")})

    def _read_until_transcript(self) -> str:
        start = self._clock()
        heard = False
        while not self._stop.is_set():
            elapsed = self._clock() - start
            if elapsed > MAX_LISTEN_S or (not heard and elapsed > START_TIMEOUT_S):
                raise VoiceError("Didn't hear anything." if not heard else "That was too long; try a shorter command.")
            try:
                msg = json.loads(self._ws.recv(timeout=0.5))
            except TimeoutError:
                continue
            except Exception as e:
                if self._stop.is_set():
                    break
                raise VoiceError("The ElevenLabs connection dropped.") from e
            kind = msg.get("type")
            if kind == "ping":
                self._send({"type": "pong", "event_id": msg.get("ping_event", {}).get("event_id")})
            elif kind == "vad_score" and msg.get("vad_score_event", {}).get("vad_score", 0) > 0.5:
                heard = True
            elif kind == "user_transcript":
                text = TAG.sub("", msg.get("user_transcription_event", {}).get("user_transcript", "")).strip()
                if text and text not in ("...", "…"):
                    return text
        raise VoiceError("Listening cancelled.")


def _level(pcm: bytes) -> float:
    """0..1 loudness of one chunk, for the listening ring."""
    if not pcm:
        return 0.0
    import array
    a = array.array("h", pcm[: len(pcm) // 2 * 2])
    peak = max((abs(x) for x in a[::8]), default=0)
    return min(1.0, peak / 12000)


def _default_connect(url: str):
    from websockets.sync.client import connect
    return connect(url, max_size=None, open_timeout=10)


class MicIn:
    """16 kHz mono int16 microphone stream. Nothing is buffered beyond one chunk or written to disk."""

    def __init__(self):
        self._pa = self._in = None

    def open(self):
        import pyaudio
        self._pa = pyaudio.PyAudio()
        self._in = self._pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

    def read(self, frames: int) -> bytes:
        return self._in.read(frames, exception_on_overflow=False)

    def close(self):
        try:
            if self._in is not None:
                self._in.stop_stream()
                self._in.close()
        except Exception:
            pass
        if self._pa is not None:
            self._pa.terminate()
