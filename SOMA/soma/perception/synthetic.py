"""Synthetic, non-sensitive GUI screens with ground-truth text elements.

Used for perception tests and as bootstrap training data (no personal screens
needed). Each scenario renders a plausible desktop window with PIL and records
every text line it draws as a ground-truth :class:`OCRElement`
(``source="synthetic"``) with its normalized box and ``ui_role``.

``render_scenario(name, seed, size)`` is deterministic per ``(name, seed,
size)``. The ground truth is attached to the returned image (see
:func:`ground_truth_for`) so :class:`~soma.perception.ocr.SyntheticOCREngine`
can "recognize" it without a real OCR model.
"""

from __future__ import annotations

import hashlib
import random
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from soma.types import ActionType, OCRElement, ScreenObservation, WindowInfo

Color = tuple[int, int, int]

# ----------------------------------------------------------------- ground-truth attachment


class SyntheticImage(np.ndarray):
    """``ndarray`` subclass that carries ground-truth elements as ``.elements``."""

    elements: list[OCRElement]

    def __array_finalize__(self, obj: object) -> None:  # keep elements on views
        self.elements = list(getattr(obj, "elements", []))


_REGISTRY: "OrderedDict[str, list[OCRElement]]" = OrderedDict()
_REGISTRY_LOCK = threading.Lock()
_REGISTRY_MAX = 512


def image_fingerprint(image: np.ndarray) -> str:
    """Content hash of an image (shape + bytes)."""
    arr = np.ascontiguousarray(np.asarray(image))
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _register(image: np.ndarray, elements: list[OCRElement]) -> None:
    key = image_fingerprint(image)
    with _REGISTRY_LOCK:
        _REGISTRY[key] = list(elements)
        _REGISTRY.move_to_end(key)
        while len(_REGISTRY) > _REGISTRY_MAX:
            _REGISTRY.popitem(last=False)


def ground_truth_for(image: np.ndarray) -> list[OCRElement] | None:
    """Ground-truth elements of a rendered synthetic image, or ``None``.

    Works for the returned :class:`SyntheticImage` and for plain copies of it
    (looked up by content hash in a bounded in-memory registry).
    """
    els = getattr(image, "elements", None)
    if els:
        return list(els)
    with _REGISTRY_LOCK:
        if not _REGISTRY:
            return None
    key = image_fingerprint(image)
    with _REGISTRY_LOCK:
        found = _REGISTRY.get(key)
    return list(found) if found is not None else None


# ----------------------------------------------------------------- drawing helpers


_FONT_CACHE: dict[int, ImageFont.ImageFont | ImageFont.FreeTypeFont] = {}


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.load_default(size=size)
        except TypeError:  # Pillow < 10.1
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


@dataclass
class Theme:
    bg: Color
    panel: Color
    fg: Color
    dim: Color
    accent: Color
    error: Color
    ok: Color
    button: Color


DARK = Theme(bg=(30, 30, 30), panel=(37, 37, 38), fg=(212, 212, 212), dim=(133, 133, 133),
             accent=(0, 122, 204), error=(244, 71, 71), ok=(80, 200, 120), button=(60, 60, 60))
LIGHT = Theme(bg=(255, 255, 255), panel=(243, 243, 243), fg=(30, 30, 30), dim=(110, 110, 110),
              accent=(0, 95, 184), error=(200, 30, 30), ok=(30, 140, 60), button=(225, 225, 225))


@dataclass
class Canvas:
    """PIL canvas that records every text line as a ground-truth element."""

    w: int
    h: int
    theme: Theme
    rng: random.Random
    image: Image.Image = field(init=False)
    draw: ImageDraw.ImageDraw = field(init=False)
    elements: list[OCRElement] = field(default_factory=list)
    fs: int = field(init=False)

    def __post_init__(self) -> None:
        self.image = Image.new("RGB", (self.w, self.h), self.theme.bg)
        self.draw = ImageDraw.Draw(self.image)
        self.fs = max(10, round(self.h / 50))

    @property
    def lh(self) -> int:
        """Line height in pixels."""
        return int(self.fs * 1.5)

    def rect(self, x0: float, y0: float, x1: float, y1: float, fill: Color,
             outline: Color | None = None, radius: int = 0) -> None:
        box = (int(x0), int(y0), int(x1), int(y1))
        if radius:
            self.draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline)
        else:
            self.draw.rectangle(box, fill=fill, outline=outline)

    def text(self, x: float, y: float, s: str, role: str = "text", fill: Color | None = None,
             size: int | None = None) -> tuple[int, int, int, int]:
        """Draw ``s`` at pixel ``(x,y)``; record and return its pixel bbox."""
        font = _font(size or self.fs)
        self.draw.text((int(x), int(y)), s, fill=fill or self.theme.fg, font=font)
        l, t, r, b = self.draw.textbbox((int(x), int(y)), s, font=font)
        l, t, r, b = max(0, l), max(0, t), min(self.w, r), min(self.h, b)
        if s.strip() and r > l and b > t:
            self.elements.append(OCRElement(
                text=s, bbox_xyxy_normalized=(l / self.w, t / self.h, r / self.w, b / self.h),
                confidence=1.0, source="synthetic", ui_role=role))
        return l, t, r, b

    def button(self, x: float, y: float, label: str, primary: bool = False) -> tuple[int, int, int, int]:
        """A bordered button whose element box covers the whole button."""
        font = _font(self.fs)
        l, t, r, b = self.draw.textbbox((0, 0), label, font=font)
        pw, ph = int(self.fs * 1.2), int(self.fs * 0.5)
        x1, y1 = int(x + (r - l) + 2 * pw), int(y + (b - t) + 2 * ph)
        fill = self.theme.accent if primary else self.theme.button
        self.rect(x, y, x1, y1, fill=fill, outline=self.theme.dim, radius=max(2, self.fs // 4))
        tx, ty = x + pw - l, y + ph - t
        color = (255, 255, 255) if primary else self.theme.fg
        self.draw.text((int(tx), int(ty)), label, fill=color, font=font)
        self.elements.append(OCRElement(
            text=label, bbox_xyxy_normalized=(x / self.w, y / self.h, min(x1, self.w) / self.w,
                                              min(y1, self.h) / self.h),
            confidence=1.0, source="synthetic", ui_role="button"))
        return int(x), int(y), x1, y1

    def title_bar(self, title: str) -> int:
        """Window title bar; returns the content top y."""
        th = int(self.lh * 1.4)
        self.rect(0, 0, self.w, th, fill=self.theme.panel)
        self.text(self.fs, (th - self.fs) / 2, title, role="title", fill=self.theme.dim)
        return th


# ----------------------------------------------------------------- content vocabularies

_PROJECTS = ["inventory", "weather_app", "budget_tool", "photo_sort", "todo_api", "garden_planner"]
_MODULES = ["models", "utils", "parser", "service", "handlers", "storage", "report"]
_FUNCS = ["load_items", "parse_row", "compute_total", "fetch_forecast", "save_record",
          "normalize_name", "build_index", "merge_lists"]
_VARS = ["items", "row", "total", "result", "values", "record", "count", "name"]
_ERRORS = [
    ("NameError", "name '{v}' is not defined"),
    ("TypeError", "unsupported operand type(s) for +: 'int' and 'str'"),
    ("KeyError", "'{v}'"),
    ("AttributeError", "'NoneType' object has no attribute '{v}'"),
    ("ZeroDivisionError", "division by zero"),
    ("IndexError", "list index out of range"),
    ("AssertionError", "assert 3 == 4"),
]
_DOC_TOPICS = [
    ("Working with lists", ["Lists are ordered, mutable sequences.",
                            "Use append() to add an item to the end.",
                            "Slicing returns a new list."]),
    ("Reading CSV files", ["The csv module reads and writes tabular data.",
                           "csv.DictReader maps each row to a dict.",
                           "Always open files with newline=''."]),
    ("Dataclasses", ["The dataclass decorator generates __init__ and __repr__.",
                     "Use field(default_factory=list) for mutable defaults.",
                     "Frozen dataclasses are hashable."]),
]


def _code_lines(rng: random.Random, fn: str, var: str, n: int) -> list[str]:
    lines = [f"def {fn}({var}):", f'    """Return the processed {var}."""', "    result = []",
             f"    for x in {var}:", "        if x is None:", "            continue",
             "        result.append(x * 2)", "    return result", "", "",
             "def main():", f"    data = {fn}([1, 2, 3])", "    print(data)", "", "",
             'if __name__ == "__main__":', "    main()"]
    return lines[:n]


# ----------------------------------------------------------------- scenarios


@dataclass
class _Ctx:
    rng: random.Random
    project: str
    module: str
    fn: str
    var: str
    file: str
    line: int


def _ctx(rng: random.Random) -> _Ctx:
    module = rng.choice(_MODULES)
    return _Ctx(rng=rng, project=rng.choice(_PROJECTS), module=module, fn=rng.choice(_FUNCS),
                var=rng.choice(_VARS), file=f"{module}.py", line=rng.randint(3, 120))


def _editor(c: Canvas, x: _Ctx, with_error: bool) -> WindowInfo:
    title = f"{x.file} - {x.project} - Visual Studio Code"
    top = c.title_bar(title)
    side_w = int(c.w * 0.18)
    c.rect(0, top, side_w, c.h, fill=c.theme.panel)
    c.text(c.fs, top + c.fs * 0.5, "EXPLORER", role="header", fill=c.theme.dim)
    files = sorted({x.file, "README.md", "requirements.txt", f"test_{x.module}.py",
                    f"{c.rng.choice(_MODULES)}.py"})
    for i, f in enumerate(files):
        c.text(c.fs * 1.5, top + c.lh * (1.5 + i), f, role="listitem")
    # tabs
    tab_y = top
    c.rect(side_w, tab_y, c.w, tab_y + c.lh * 1.4, fill=c.theme.panel)
    c.text(side_w + c.fs, tab_y + c.lh * 0.3, x.file, role="tab")
    c.text(side_w + c.fs * 10, tab_y + c.lh * 0.3, f"test_{x.module}.py", role="tab", fill=c.theme.dim)
    # code
    code_top = tab_y + c.lh * 1.8
    panel_h = c.lh * 5 if with_error else 0
    status_h = c.lh
    n = max(1, int((c.h - code_top - panel_h - status_h) // c.lh) - 1)
    start = max(1, x.line - c.rng.randint(1, 6))
    for i, src in enumerate(_code_lines(c.rng, x.fn, x.var, n)):
        y = code_top + i * c.lh
        c.text(side_w + c.fs, y, f"{start + i:>3}", role="linenumber", fill=c.theme.dim)
        if src:
            c.text(side_w + c.fs * 4, y, src, role="code")
    if with_error:
        py = c.h - status_h - panel_h
        c.rect(side_w, py, c.w, c.h - status_h, fill=c.theme.panel)
        c.text(side_w + c.fs, py + c.fs * 0.4, "PROBLEMS", role="tab")
        etype, emsg = c.rng.choice(_ERRORS)
        c.text(side_w + c.fs, py + c.lh * 1.5, f"{x.file}:{x.line}:5: error: {etype}: {emsg.format(v=x.var)}",
               role="error", fill=c.theme.error)
        c.text(side_w + c.fs, py + c.lh * 2.6, f'"{x.var}" is possibly unbound  Pylance  [Ln {x.line}, Col 5]',
               role="error", fill=c.theme.error)
    c.rect(0, c.h - status_h, c.w, c.h, fill=c.theme.accent)
    c.text(c.fs, c.h - status_h + (status_h - c.fs) / 3, f"Ln {x.line}, Col 5    UTF-8    Python",
           role="statusbar", fill=(255, 255, 255))
    return WindowInfo(app_name="Code.exe", title=title, window_id="synthetic:editor")


def _terminal(c: Canvas, x: _Ctx, failing: bool) -> WindowInfo:
    title = "Windows PowerShell"
    top = c.title_bar(title)
    root = f"C:\\dev\\{x.project}"
    passed = c.rng.randint(2, 30)
    lines: list[tuple[str, str, Color | None]] = [
        (f"PS {root}> python -m pytest -q", "text", None)]
    if failing:
        etype, emsg = c.rng.choice(_ERRORS)
        emsg = emsg.format(v=x.var)
        lines += [
            ("." * min(passed, 20) + "F", "text", None),
            ("=" * 30 + " FAILURES " + "=" * 30, "text", None),
            (f"___ test_{x.fn} ___", "text", None),
            ("Traceback (most recent call last):", "error", c.theme.error),
            (f'  File "{root}\\tests\\test_{x.module}.py", line {c.rng.randint(5, 60)}, in test_{x.fn}',
             "file_ref", None),
            (f"    assert {x.fn}([1, 2]) == [2, 4]", "code", None),
            (f'  File "{root}\\{x.file}", line {x.line}, in {x.fn}', "file_ref", None),
            (f"    result.append({x.var} * 2)", "code", None),
            (f"{etype}: {emsg}", "error", c.theme.error),
            (f"FAILED tests/test_{x.module}.py::test_{x.fn} - {etype}: {emsg}", "error", c.theme.error),
            (f"1 failed, {passed} passed in 0.{c.rng.randint(10, 99)}s", "error", c.theme.error),
        ]
    else:
        lines += [("." * min(passed, 40), "text", None),
                  (f"{passed} passed in 0.{c.rng.randint(10, 99)}s", "text", c.theme.ok)]
    lines.append((f"PS {root}> ", "text", None))
    for i, (s, role, color) in enumerate(lines):
        c.text(c.fs, top + c.fs * 0.6 + i * c.lh, s, role=role, fill=color)
    return WindowInfo(app_name="WindowsTerminal.exe", title=title, window_id="synthetic:terminal")


def _browser(c: Canvas, x: _Ctx) -> WindowInfo:
    topic, paras = c.rng.choice(_DOC_TOPICS)
    title = f"{topic} - Python documentation - Google Chrome"
    top = c.title_bar(title)
    c.rect(0, top, c.w, top + c.lh * 1.8, fill=c.theme.panel)
    slug = topic.lower().replace(" ", "-")
    c.rect(c.fs * 6, top + c.lh * 0.3, c.w - c.fs * 6, top + c.lh * 1.5, fill=c.theme.bg, radius=c.fs // 2)
    c.text(c.fs * 7, top + c.lh * 0.45, f"https://docs.python.org/3/tutorial/{slug}.html", role="textbox")
    y = top + c.lh * 3
    c.text(c.w * 0.15, y, topic, role="heading", size=int(c.fs * 1.8))
    y += c.lh * 2.5
    for p in paras:
        c.text(c.w * 0.15, y, p, role="text")
        y += c.lh * 1.4
    y += c.lh
    c.rect(c.w * 0.15, y, c.w * 0.75, y + c.lh * 4, fill=c.theme.panel)
    c.text(c.w * 0.16, y + c.lh * 0.5, f"{x.var} = [1, 2, 3]", role="code")
    c.text(c.w * 0.16, y + c.lh * 1.5, f"{x.var}.append(4)", role="code")
    c.button(c.w * 0.68, y + c.lh * 0.3, "Copy")
    y += c.lh * 5.5
    c.text(c.w * 0.15, y, "Next topic: " + c.rng.choice(_DOC_TOPICS)[0], role="link", fill=c.theme.accent)
    return WindowInfo(app_name="chrome.exe", title=title, window_id="synthetic:browser")


def _desktop(c: Canvas, x: _Ctx) -> WindowInfo:
    c.image.paste(tuple(c.rng.randint(20, 90) for _ in range(3)), (0, 0, c.w, c.h))
    names = ["Recycle Bin", "Documents", "Projects", "Notes.txt", "Music", "Photos"]
    for i, n in enumerate(names[: c.rng.randint(3, 6)]):
        y = c.lh + i * c.lh * 3.5
        c.rect(c.fs * 2, y, c.fs * 5, y + c.fs * 2.5, fill=(200, 200, 120))
        c.text(c.fs * 1.2, y + c.fs * 2.8, n, role="listitem", fill=(255, 255, 255))
    bar = c.h - c.lh * 1.8
    c.rect(0, bar, c.w, c.h, fill=(32, 32, 40))
    hh, mm = c.rng.randint(8, 18), c.rng.randint(0, 59)
    c.text(c.w - c.fs * 6, bar + c.lh * 0.4, f"{hh:02d}:{mm:02d}", role="text", fill=(255, 255, 255))
    return WindowInfo(app_name="explorer.exe", title="Program Manager", window_id="synthetic:desktop")


def _settings(c: Canvas, x: _Ctx) -> WindowInfo:
    title = "Settings"
    top = c.title_bar(title)
    y = top + c.lh
    c.text(c.fs * 2, y, c.rng.choice(["Display", "Notifications", "Sound", "Appearance"]),
           role="heading", size=int(c.fs * 1.6))
    y += c.lh * 2.5
    for opt in c.rng.sample(["Enable dark mode", "Show notifications", "Start at login",
                             "Use large text", "Play sounds", "Auto-save files"], 4):
        c.rect(c.fs * 2, y + 2, c.fs * 3.2, y + c.fs * 1.2, fill=c.theme.button, outline=c.theme.dim)
        c.text(c.fs * 4, y, opt, role="checkbox")
        y += c.lh * 1.6
    by = c.h - c.lh * 3
    bx = c.w - c.fs * 26
    for label, primary in [("OK", True), ("Cancel", False), ("Apply", False)]:
        _, _, x1, _ = c.button(bx, by, label, primary=primary)
        bx = x1 + c.fs
    return WindowInfo(app_name="SystemSettings.exe", title=title, window_id="synthetic:settings")


def _explorer(c: Canvas, x: _Ctx) -> WindowInfo:
    title = f"{x.project} - File Explorer"
    top = c.title_bar(title)
    c.rect(0, top, c.w, top + c.lh * 1.8, fill=c.theme.panel)
    c.text(c.fs * 2, top + c.lh * 0.45, f"This PC > Local Disk (C:) > dev > {x.project}", role="textbox")
    y = top + c.lh * 2.5
    for col, xp in [("Name", 0.05), ("Date modified", 0.45), ("Size", 0.7)]:
        c.text(c.w * xp, y, col, role="header", fill=c.theme.dim)
    y += c.lh * 1.4
    files = [x.file, f"test_{x.module}.py", "README.md", "requirements.txt", "data.csv"]
    c.rng.shuffle(files)
    for f in files:
        c.text(c.w * 0.05, y, f, role="listitem")
        c.text(c.w * 0.45, y, f"2026-0{c.rng.randint(1, 9)}-1{c.rng.randint(0, 9)} 10:{c.rng.randint(10, 59)}",
               role="text", fill=c.theme.dim)
        c.text(c.w * 0.7, y, f"{c.rng.randint(1, 90)} KB", role="text", fill=c.theme.dim)
        y += c.lh * 1.3
    return WindowInfo(app_name="explorer.exe", title=title, window_id="synthetic:explorer")


_Renderer = Callable[[Canvas, _Ctx], WindowInfo]

_RENDERERS: dict[str, tuple[_Renderer, ActionType]] = {
    "editor_code": (lambda c, x: _editor(c, x, with_error=False), ActionType.NO_SUGGESTION),
    "editor_error": (lambda c, x: _editor(c, x, with_error=True), ActionType.EXPLAIN_ERROR),
    "terminal_traceback": (lambda c, x: _terminal(c, x, failing=True), ActionType.EXPLAIN_ERROR),
    "terminal_ok": (lambda c, x: _terminal(c, x, failing=False), ActionType.SUGGEST_NEXT_STEP),
    "browser_docs": (_browser, ActionType.COMPARE_WITH_PRIOR_WORK),
    "idle_desktop": (_desktop, ActionType.NO_SUGGESTION),
    "settings_dialog": (_settings, ActionType.NO_SUGGESTION),
    "file_explorer": (_explorer, ActionType.OPEN_RELEVANT_FILE),
}

#: Names accepted by :func:`render_scenario`.
SCENARIOS: tuple[str, ...] = tuple(_RENDERERS)


def expected_action(name: str) -> ActionType:
    """Weak supervision label: the action a helpful assistant would propose."""
    return _RENDERERS[name][1]


def render_scenario(name: str, seed: int = 0, size: tuple[int, int] = (1280, 800),
                    ) -> tuple[np.ndarray, ScreenObservation, str]:
    """Render scenario ``name`` deterministically.

    Returns ``(image [H,W,3] uint8, observation with ground-truth elements,
    label)`` where ``label`` is the scenario name. The image is a
    :class:`SyntheticImage` carrying ``.elements``.
    """
    if name not in _RENDERERS:
        raise KeyError(f"unknown scenario {name!r}; choose from {SCENARIOS}")
    w, h = int(size[0]), int(size[1])
    if w < 64 or h < 64:
        raise ValueError("size must be at least 64x64")
    digest = hashlib.blake2b(f"{name}:{int(seed)}".encode(), digest_size=8).digest()
    rng = random.Random(int.from_bytes(digest, "little"))
    theme = DARK if rng.random() < 0.6 else LIGHT
    canvas = Canvas(w=w, h=h, theme=theme, rng=rng)
    renderer, _ = _RENDERERS[name]
    window = renderer(canvas, _ctx(rng))
    window = replace(window, bounds=(0, 0, w, h))
    elements = [replace(e, window_id=window.window_id) for e in canvas.elements]
    arr = np.asarray(canvas.image, dtype=np.uint8).copy()
    image = arr.view(SyntheticImage)
    image.elements = elements
    _register(arr, elements)
    obs = ScreenObservation(window=window, elements=list(elements), version=int(seed), timestamp_ms=0)
    return image, obs, name
