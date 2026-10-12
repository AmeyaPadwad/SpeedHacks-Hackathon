"""Consent-gated, rate-limited, change-detecting screen capture.

:class:`ScreenCapturer` produces in-memory :class:`~soma.types.ScreenFrame`
objects. Privacy gates run **before** any pixels are grabbed:

1. consent for :attr:`Scope.SCREEN_CAPTURE` (a paused manager denies it);
2. the foreground window is not excluded (:func:`redaction.is_excluded`);
3. the rate limit (``min_interval_s``) has elapsed.

After grabbing, frames whose downsampled difference against the last emitted
frame is below ``change_threshold`` are dropped. Nothing is written to disk;
:meth:`ScreenCapturer.persist` writes a masked PNG only when
``cfg.persist_screenshots`` is enabled *and* the caller supplied a directory.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from soma.capture.redaction import is_excluded, mask_regions, redact_text
from soma.capture.window import WindowFn, foreground_window
from soma.config import CaptureConfig
from soma.consent import ConsentManager, Scope
from soma.perception.preprocess import change_ratio
from soma.types import ScreenFrame, WindowInfo, now_ms

GrabFn = Callable[[WindowInfo], np.ndarray]
Clock = Callable[[], float]

# Skip reasons reported through ``ScreenCapturer.last_skip_reason``.
SKIP_NO_CONSENT = "no-consent"
SKIP_PAUSED = "paused"
SKIP_EXCLUDED = "excluded"
SKIP_RATE_LIMITED = "rate-limited"
SKIP_UNCHANGED = "unchanged"
SKIP_ERROR = "grab-error"


@dataclass
class CaptureStats:
    """Counters for profiling and the debug overlay (no content)."""

    attempts: int = 0
    captured: int = 0
    skipped_no_consent: int = 0
    skipped_paused: int = 0
    skipped_excluded: int = 0
    skipped_rate_limited: int = 0
    skipped_unchanged: int = 0
    errors: int = 0
    persisted: int = 0
    last_change_ratio: float = 0.0
    last_grab_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------- default grab

_mss_local = threading.local()


def _mss_instance() -> Any:
    """One ``mss`` handle per thread (mss handles are not thread-safe)."""
    inst = getattr(_mss_local, "sct", None)
    if inst is None:
        import mss  # lazy: optional dependency
        inst = mss.mss()
        _mss_local.sct = inst
    return inst


def _clip_region(bounds: tuple[int, int, int, int], virtual: dict[str, int]) -> dict[str, int] | None:
    x, y, w, h = bounds
    if w <= 0 or h <= 0:
        return None
    left = max(x, virtual["left"])
    top = max(y, virtual["top"])
    right = min(x + w, virtual["left"] + virtual["width"])
    bottom = min(y + h, virtual["top"] + virtual["height"])
    if right - left < 8 or bottom - top < 8:
        return None
    return {"left": left, "top": top, "width": right - left, "height": bottom - top}


def mss_grab(window: WindowInfo) -> np.ndarray:
    """Grab the foreground window bounds (or the primary monitor) as RGB uint8."""
    sct = _mss_instance()
    region = _clip_region(window.bounds, sct.monitors[0]) or sct.monitors[1]
    shot = sct.grab(region)
    bgra = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)
    return np.ascontiguousarray(bgra[:, :, 2::-1])  # BGRA -> RGB copy


# ----------------------------------------------------------------- capturer


class ScreenCapturer:
    """Produce :class:`ScreenFrame` objects only when allowed and meaningful.

    Args:
        consent: consent store; ``SCREEN_CAPTURE`` must be granted and not paused.
        cfg: capture configuration (rate limit, change threshold, exclusions).
        grab: ``grab(window) -> [H,W,3] uint8 RGB``. Defaults to :func:`mss_grab`.
        window_fn: returns the foreground :class:`WindowInfo`.
        clock: monotonic seconds, injectable for tests.
    """

    def __init__(self, consent: ConsentManager, cfg: CaptureConfig | None = None,
                 grab: GrabFn | None = None, window_fn: WindowFn | None = None,
                 clock: Clock = time.monotonic) -> None:
        self.consent = consent
        self.cfg = cfg or CaptureConfig()
        self._grab = grab or mss_grab
        self._window_fn = window_fn or foreground_window
        self._clock = clock
        self._lock = threading.Lock()
        self._version = 0
        self._last_grab_t: float | None = None
        self._last_image: np.ndarray | None = None
        self._last_window_id: str | None = None
        self.last_skip_reason: str = ""
        self.last_window: WindowInfo | None = None
        self.stats = CaptureStats()

    @property
    def version(self) -> int:
        """Version of the last emitted frame (0 before the first one)."""
        return self._version

    def reset(self) -> None:
        """Forget the change-detection reference (e.g. after resume)."""
        with self._lock:
            self._last_image = None
            self._last_window_id = None
            self._last_grab_t = None

    def _skip(self, reason: str) -> None:
        self.last_skip_reason = reason
        counter = {
            SKIP_NO_CONSENT: "skipped_no_consent", SKIP_PAUSED: "skipped_paused",
            SKIP_EXCLUDED: "skipped_excluded", SKIP_RATE_LIMITED: "skipped_rate_limited",
            SKIP_UNCHANGED: "skipped_unchanged", SKIP_ERROR: "errors",
        }[reason.split(":", 1)[0]]
        setattr(self.stats, counter, getattr(self.stats, counter) + 1)

    def capture(self, force: bool = False) -> ScreenFrame | None:
        """Return a new frame, or ``None`` with ``last_skip_reason`` set.

        ``force`` bypasses the rate limit and change detection but **never**
        the consent, pause or exclusion gates.
        """
        with self._lock:
            self.stats.attempts += 1
            # 1. consent / pause: decided before touching the window or pixels
            if self.consent.paused:
                self._skip(SKIP_PAUSED)
                return None
            if not self.consent.allowed(Scope.SCREEN_CAPTURE):
                self._skip(SKIP_NO_CONSENT)
                return None
            # 2. exclusion
            window = self._window_fn()
            self.last_window = window
            excluded, why = is_excluded(window, self.cfg)
            if excluded:
                self._skip(f"{SKIP_EXCLUDED}:{why}")
                # never keep a reference frame across an excluded window
                self._last_image = None
                return None
            # 3. rate limit
            t = self._clock()
            if (not force and self._last_grab_t is not None
                    and t - self._last_grab_t < self.cfg.min_interval_s):
                self._skip(SKIP_RATE_LIMITED)
                return None
            # 4. grab
            self._last_grab_t = t
            t0 = time.perf_counter()
            try:
                image = np.asarray(self._grab(window))
            except Exception as exc:  # noqa: BLE001 - surface as a skip, not a crash
                self._skip(f"{SKIP_ERROR}:{type(exc).__name__}")
                return None
            self.stats.last_grab_ms = (time.perf_counter() - t0) * 1000.0
            if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
                self._skip(f"{SKIP_ERROR}:bad-shape")
                return None
            image = np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)
            # 5. change detection vs. the last emitted frame of the same window
            same_window = window.window_id == self._last_window_id
            ratio = change_ratio(self._last_image, image) if same_window else 1.0
            self.stats.last_change_ratio = ratio
            if not force and same_window and ratio < self.cfg.change_threshold:
                self._skip(SKIP_UNCHANGED)
                return None
            self._version += 1
            self._last_image = image
            self._last_window_id = window.window_id
            self.stats.captured += 1
            self.last_skip_reason = ""
            safe_window = WindowInfo(app_name=window.app_name, title=redact_text(window.title),
                                     window_id=window.window_id, pid=window.pid,
                                     bounds=window.bounds)
            return ScreenFrame(image=image, window=safe_window, version=self._version,
                               timestamp_ms=now_ms())

    def persist(self, frame: ScreenFrame, directory: str | Path | None,
                extra_masks: Sequence[Sequence[float]] = ()) -> Path | None:
        """Write a masked PNG of ``frame`` into ``directory``.

        Does nothing (returns ``None``) unless ``cfg.persist_screenshots`` is
        enabled and a directory is given. ``frame.redacted_regions`` and
        ``extra_masks`` (normalized ``xyxy``) are blacked out first.
        """
        if not self.cfg.persist_screenshots or directory is None:
            return None
        from PIL import Image

        boxes = list(frame.redacted_regions) + [tuple(b) for b in extra_masks]
        masked = mask_regions(frame.image, boxes)
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"frame_{frame.version:06d}_{frame.timestamp_ms}.png"
        Image.fromarray(masked).save(path)
        self.stats.persisted += 1
        return path
