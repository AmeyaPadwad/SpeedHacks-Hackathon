"""Module 1: input engine. Global hotkey, record until silence, Whisper transcription."""
import io
import os

HOTKEY = os.environ.get("FLOWSTATE_HOTKEY", "ctrl+shift+space")
LISTEN_TIMEOUT_S = 6      # wait this long for speech to start
PHRASE_LIMIT_S = 20       # hard cap on one command
TRANSCRIBE_TIMEOUT_S = 20


class VoiceError(Exception):
    """Mic unavailable, nothing heard, or transcription failed; caller offers typed input instead."""


def register_hotkey(callback) -> bool:
    """Call `callback` (on the keyboard library's thread) when the hotkey is pressed. False if unavailable."""
    try:
        import keyboard
        keyboard.add_hotkey(HOTKEY, callback, suppress=False)
        return True
    except Exception:  # missing package, or OS refused the hook
        return False


def unregister_hotkeys():
    try:
        import keyboard
        keyboard.unhook_all_hotkeys()
    except Exception:
        pass


def record_until_silence() -> bytes:
    """Block until the user stops talking; returns WAV bytes. Run off the UI thread."""
    try:
        import speech_recognition as sr
    except ImportError as e:
        raise VoiceError("SpeechRecognition is not installed.") from e
    r = sr.Recognizer()
    r.pause_threshold = 0.8
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.3)
            audio = r.listen(source, timeout=LISTEN_TIMEOUT_S, phrase_time_limit=PHRASE_LIMIT_S)
    except sr.WaitTimeoutError as e:
        raise VoiceError("Didn't hear anything.") from e
    except (OSError, AttributeError) as e:  # no mic / PyAudio missing
        raise VoiceError("Microphone unavailable.") from e
    return audio.get_wav_data()


def transcribe(wav: bytes, client=None) -> str:
    """Whisper speech-to-text. Audio stays in memory and is not saved."""
    if client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise VoiceError("OPENAI_API_KEY is not set.")
        from openai import OpenAI
        client = OpenAI(timeout=TRANSCRIBE_TIMEOUT_S, max_retries=0)
    f = io.BytesIO(wav)
    f.name = "command.wav"
    try:
        text = client.audio.transcriptions.create(model="whisper-1", file=f).text
    except Exception as e:
        raise VoiceError(f"Transcription failed: {type(e).__name__}") from e
    text = (text or "").strip()
    if not text:
        raise VoiceError("Couldn't make out any words.")
    return text
