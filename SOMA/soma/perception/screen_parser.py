"""Screen parser: OCR + accessibility -> :class:`ScreenObservation`.

OCR text is not assumed to be clickable. Roles come from, in order:

1. the accessibility element the OCR box overlaps (IoU / containment);
2. heuristics: error lines (``Traceback``, ``...Error:``, ``FAILED``),
   file references (``File "x.py", line 12`` / ``x.py:12``), URLs (link) and
   short labels drawn on a distinct filled box (button);
3. otherwise ``"text"``.

Also provides resize-safe geometry helpers (normalized <-> pixel, window <->
screen) and :func:`find_errors` / :func:`find_file_refs`.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from soma.types import OCRElement, ScreenFrame, ScreenObservation, WindowInfo, now_ms

Box = tuple[float, float, float, float]
PixelBox = tuple[int, int, int, int]

# ----------------------------------------------------------------- geometry


def to_pixels(bbox: Sequence[float], size: tuple[int, int]) -> PixelBox:
    """Normalized ``xyxy`` -> integer pixel ``xyxy`` for an image of ``size=(w,h)``."""
    w, h = size
    x0, y0, x1, y1 = bbox
    return (int(round(x0 * w)), int(round(y0 * h)), int(round(x1 * w)), int(round(y1 * h)))


def to_normalized(box: Sequence[float], size: tuple[int, int]) -> Box:
    """Pixel ``xyxy`` -> normalized ``xyxy`` (clipped to ``[0,1]``) for ``size=(w,h)``."""
    w, h = max(1, size[0]), max(1, size[1])
    x0, y0, x1, y1 = box

    def c(v: float) -> float:
        return float(min(1.0, max(0.0, v)))

    return (c(x0 / w), c(y0 / h), c(x1 / w), c(y1 / h))


def rescale_pixels(box: Sequence[float], from_size: tuple[int, int], to_size: tuple[int, int]) -> PixelBox:
    """Map a pixel box between two resolutions of the same content (e.g. after a resize)."""
    return to_pixels(to_normalized(box, from_size), to_size)


def window_to_screen(bbox: Sequence[float], bounds: tuple[int, int, int, int]) -> PixelBox:
    """Normalized box inside a window with ``bounds=(x,y,w,h)`` -> screen pixels."""
    x, y, w, h = bounds
    px0, py0, px1, py1 = to_pixels(bbox, (w, h))
    return (px0 + x, py0 + y, px1 + x, py1 + y)


def screen_to_window(box: Sequence[float], bounds: tuple[int, int, int, int]) -> Box:
    """Screen-pixel box -> normalized box inside a window with ``bounds=(x,y,w,h)``."""
    x, y, w, h = bounds
    x0, y0, x1, y1 = box
    return to_normalized((x0 - x, y0 - y, x1 - x, y1 - y), (w, h))


def box_area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: Sequence[float], b: Sequence[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``xyxy`` boxes."""
    inter = intersection(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


# ----------------------------------------------------------------- text heuristics

_ERROR_RES = [
    re.compile(r"^\s*Traceback\b"),
    re.compile(r"^\s*[A-Za-z_][\w.]*(?:Error|Exception|Warning)\s*(?::|$)"),
    re.compile(r"\b(?:error|fatal|panic)\s*(?:\[\w+\]|[A-Z]+\d+)?\s*:", re.I),
    re.compile(r"\bFAILED\b|^\s*FAIL\b|\bERROR\b"),
    re.compile(r"\b\d+\s*failed\b", re.I),
    re.compile(r"\bUncaught\s+\w*(?:Error|Exception)"),
    re.compile(r"^\s*Exception\b"),
]
_PY_FILE_RE = re.compile(r'File\s*"(?P<path>[^"]+)"\s*,\s*line\s*(?P<line>\d+)')
_EXTS = (r"py|pyi|ipynb|js|jsx|ts|tsx|mjs|go|rs|java|kt|c|cc|cpp|h|hpp|cs|rb|php|swift|"
         r"md|json|ya?ml|toml|txt|cfg|ini|html|css|sql|sh|ps1")
_PATH_RE = re.compile(
    rf"(?<![\w/\\.-])(?P<path>(?:[A-Za-z]:[\\/])?[\w.\\/-]*\w\.(?:{_EXTS}))\b"
    r"(?:(?:::[\w\[\]-]+)|(?::(?P<line>\d+))(?::(?P<col>\d+))?|\((?P<pline>\d+)(?:,(?P<pcol>\d+))?\))?")
_URL_RE = re.compile(r"\bhttps?://\S+|\bwww\.\S+\.\w+")
_BUTTON_WORDS = {
    "ok", "cancel", "apply", "save", "run", "submit", "close", "yes", "no", "copy",
    "install", "run tests", "debug", "next", "back", "open", "continue", "retry", "delete",
    "accept", "reject", "done", "allow", "deny", "send", "search", "commit", "push", "sign out",
}


def looks_like_error(text: str) -> bool:
    """Heuristic: does this line look like an error/traceback/test failure?"""
    return any(r.search(text or "") for r in _ERROR_RES)


@dataclass(frozen=True)
class FileRef:
    """A file reference seen on screen (never opened by the parser)."""

    path: str
    line: int | None = None
    column: int | None = None
    element: OCRElement | None = None


def extract_file_refs(text: str) -> list[FileRef]:
    """File references in one text line (Python tracebacks, ``x.py:12:5``, ``x.cs(12,5)``)."""
    refs: list[FileRef] = []
    spans: list[tuple[int, int]] = [m.span() for m in _URL_RE.finditer(text or "")]
    for m in _PY_FILE_RE.finditer(text or ""):
        refs.append(FileRef(path=m.group("path").strip(), line=int(m.group("line"))))
        spans.append(m.span())
    for m in _PATH_RE.finditer(text or ""):
        if any(s <= m.start() < e for s, e in spans):
            continue
        line = m.group("line") or m.group("pline")
        col = m.group("col") or m.group("pcol")
        refs.append(FileRef(path=m.group("path"), line=int(line) if line else None,
                            column=int(col) if col else None))
    return refs


def _is_file_ref_line(text: str) -> bool:
    if _PY_FILE_RE.search(text):
        return True
    return any(r.line is not None for r in extract_file_refs(text))


def _filled_box_contrast(image: np.ndarray | None, bbox: Box, thresh: float = 18.0) -> bool:
    """True when the text sits on a distinct filled rectangle (button-like).

    Compares the median colour just around the text with strips further out on
    each side; at least three available sides must differ.
    """
    if image is None:
        return False
    h, w = image.shape[:2]
    x0, y0, x1, y1 = to_pixels(bbox, (w, h))
    th = max(2, y1 - y0)
    ix0, iy0, ix1, iy1 = x0 - int(0.5 * th), y0 - int(0.25 * th), x1 + int(0.5 * th), y1 + int(0.25 * th)
    inner = []
    for sl in [(slice(max(0, iy0), max(0, y0)), slice(max(0, ix0), min(w, ix1))),
               (slice(min(h, y1), min(h, iy1)), slice(max(0, ix0), min(w, ix1))),
               (slice(max(0, y0), min(h, y1)), slice(max(0, ix0), max(0, x0))),
               (slice(max(0, y0), min(h, y1)), slice(min(w, x1), min(w, ix1)))]:
        patch = image[sl]
        if patch.size:
            inner.append(patch.reshape(-1, patch.shape[-1]))
    if not inner:
        return False
    ref = np.median(np.concatenate(inner), axis=0).astype(np.float32)
    o = int(2.0 * th)
    strips = [
        image[max(0, y0):min(h, y1), max(0, x0 - 2 * o):max(0, x0 - o)],          # left
        image[max(0, y0):min(h, y1), min(w, x1 + o):min(w, x1 + 2 * o)],          # right
        image[max(0, y0 - o - th // 2):max(0, y0 - o), max(0, x0):min(w, x1)],    # top
        image[min(h, y1 + o):min(h, y1 + o + th // 2), max(0, x0):min(w, x1)],    # bottom
    ]
    sides = [s for s in strips if s.size]
    if len(sides) < 3:
        return False
    differs = [float(np.abs(np.median(s.reshape(-1, s.shape[-1]), axis=0) - ref).mean()) > thresh
               for s in sides]
    return all(differs)


def infer_role(element: OCRElement, image: np.ndarray | None = None) -> str:
    """Heuristic ``ui_role`` for an element with no reliable role."""
    text = element.text.strip()
    if looks_like_error(text):
        return "error"
    if _is_file_ref_line(text):
        return "file_ref"
    if _URL_RE.search(text):
        return "link"
    words = text.split()
    short = 0 < len(text) <= 24 and len(words) <= 3 and any(ch.isalpha() for ch in text)
    if short and ((text[0].isupper() and text.lower().rstrip(".…") in _BUTTON_WORDS)
                  or _filled_box_contrast(image, element.bbox_xyxy_normalized)):
        return "button"
    return "text"


_GENERIC_ROLES = {None, "", "text", "document", "pane", "group", "custom", "window", "edit", "textbox"}


# ----------------------------------------------------------------- parser


def _match_a11y(e: OCRElement, a11y: Sequence[OCRElement], iou_threshold: float) -> int | None:
    """Index of the best accessibility element for OCR element ``e``."""
    best, best_key = None, None
    ea = box_area(e.bbox_xyxy_normalized)
    for i, a in enumerate(a11y):
        inter = intersection(e.bbox_xyxy_normalized, a.bbox_xyxy_normalized)
        if inter <= 0:
            continue
        score = iou(e.bbox_xyxy_normalized, a.bbox_xyxy_normalized)
        contained = ea > 0 and inter / ea >= 0.7
        if score < iou_threshold and not contained:
            continue
        # prefer real IoU, then the tightest containing element
        key = (score >= iou_threshold, score, -box_area(a.bbox_xyxy_normalized))
        if best_key is None or key > best_key:
            best, best_key = i, key
    return best


def reading_order(elements: Iterable[OCRElement]) -> list[OCRElement]:
    """Sort top-to-bottom, then left-to-right (rows grouped by vertical centre)."""
    els = list(elements)
    if not els:
        return els
    heights = sorted(max(1e-6, e.bbox_xyxy_normalized[3] - e.bbox_xyxy_normalized[1]) for e in els)
    tol = heights[len(heights) // 2] * 0.5

    def key(e: OCRElement) -> tuple[int, float]:
        cy = (e.bbox_xyxy_normalized[1] + e.bbox_xyxy_normalized[3]) / 2
        return int(cy / tol) if tol > 0 else 0, e.bbox_xyxy_normalized[0]

    return sorted(els, key=key)


def parse_screen(frame: ScreenFrame | None, ocr_elements: Sequence[OCRElement],
                 a11y_elements: Sequence[OCRElement] = (), iou_threshold: float = 0.3,
                 window: WindowInfo | None = None) -> ScreenObservation:
    """Fuse OCR and accessibility elements into a :class:`ScreenObservation`.

    Each OCR element takes the role of the accessibility element it overlaps
    (generic accessibility roles yield to error/file-ref heuristics); other
    roles are inferred heuristically. Accessibility elements with no OCR match
    are kept as-is. ``frame`` supplies the window, version, timestamp and
    pixels (for the button heuristic); it may be ``None`` with ``window`` given.
    """
    image = np.asarray(frame.image) if frame is not None else None
    a11y = [a for a in a11y_elements if a.text.strip()]
    used: set[int] = set()
    out: list[OCRElement] = []
    for e in ocr_elements:
        if not e.text.strip():
            continue
        idx = _match_a11y(e, a11y, iou_threshold)
        base = e.ui_role
        if idx is not None:
            used.add(idx)
            base = a11y[idx].ui_role or base
        role = base
        if base in _GENERIC_ROLES:
            heuristic = infer_role(e, image)
            role = heuristic if (heuristic != "text" or not base) else base
        out.append(dataclasses.replace(e, ui_role=role))
    for i, a in enumerate(a11y):
        if i not in used:
            out.append(dataclasses.replace(a, ui_role=a.ui_role or infer_role(a, image)))
    win = window or (frame.window if frame is not None else WindowInfo(app_name="", title=""))
    version = frame.version if frame is not None else 0
    ts = frame.timestamp_ms if frame is not None else now_ms()
    return ScreenObservation(window=win, elements=reading_order(out), version=version, timestamp_ms=ts)


def find_errors(obs: ScreenObservation) -> list[OCRElement]:
    """Elements that are error lines (by role or by text), in reading order."""
    return [e for e in obs.elements if e.ui_role == "error" or looks_like_error(e.text)]


def find_file_refs(obs: ScreenObservation) -> list[FileRef]:
    """Unique file references (path, line) visible on screen, in reading order."""
    seen: set[tuple[str, int | None]] = set()
    out: list[FileRef] = []
    for e in obs.elements:
        for r in extract_file_refs(e.text):
            k = (r.path, r.line)
            if k not in seen:
                seen.add(k)
                out.append(dataclasses.replace(r, element=e))
    return out


def find_buttons(obs: ScreenObservation) -> list[OCRElement]:
    """Elements with a button-like role."""
    return [e for e in obs.elements if e.ui_role in ("button", "link", "menuitem", "tab")]
