"""Optional UI Automation (accessibility) tree reader.

Accessibility labels and roles are more reliable than OCR for knowing what is
clickable. This module uses the third-party ``uiautomation`` package (which
wraps ``comtypes``) when it is importable on Windows; otherwise every call
returns ``[]``. It never raises to the caller.

Only element *names* and *control types* are read, never field values, so
password boxes expose at most their label. Names pass through
:func:`redact_text`. The same consent and exclusion gates as screen capture
apply.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from soma.capture.redaction import is_excluded, redact_text
from soma.config import CaptureConfig
from soma.consent import ConsentManager, Scope
from soma.types import OCRElement, WindowInfo

#: UI Automation control type name -> SOMA ``ui_role``.
ROLE_MAP: dict[str, str] = {
    "ButtonControl": "button",
    "SplitButtonControl": "button",
    "HyperlinkControl": "link",
    "EditControl": "textbox",
    "DocumentControl": "document",
    "TextControl": "text",
    "MenuItemControl": "menuitem",
    "MenuBarControl": "menubar",
    "TabItemControl": "tab",
    "CheckBoxControl": "checkbox",
    "RadioButtonControl": "radio",
    "ComboBoxControl": "combobox",
    "ListItemControl": "listitem",
    "TreeItemControl": "treeitem",
    "DataItemControl": "cell",
    "HeaderItemControl": "header",
    "TitleBarControl": "titlebar",
    "ToolBarControl": "toolbar",
    "StatusBarControl": "statusbar",
    "ImageControl": "image",
}


class A11yNode(Protocol):
    """Minimal accessibility node used by :func:`elements_from_tree`."""

    name: str
    control_type: str
    rect: tuple[int, int, int, int]          # left, top, right, bottom (screen px)

    def children(self) -> Iterable["A11yNode"]: ...


@dataclass
class SimpleNode:
    """Plain in-memory node (tests, synthetic data)."""

    name: str
    control_type: str
    rect: tuple[int, int, int, int]
    kids: list["SimpleNode"] = field(default_factory=list)

    def children(self) -> Iterable["SimpleNode"]:
        return self.kids


class _UIANode:  # pragma: no cover - requires Windows + uiautomation
    """Adapter from a ``uiautomation.Control`` to :class:`A11yNode`."""

    def __init__(self, control: Any) -> None:
        self._c = control
        self.name = str(getattr(control, "Name", "") or "")
        self.control_type = str(getattr(control, "ControlTypeName", "") or "")
        r = getattr(control, "BoundingRectangle", None)
        self.rect = (int(r.left), int(r.top), int(r.right), int(r.bottom)) if r is not None else (0, 0, 0, 0)

    def children(self) -> Iterable["_UIANode"]:
        try:
            return [_UIANode(c) for c in self._c.GetChildren()]
        except Exception:  # noqa: BLE001
            return []


def uia_available() -> bool:
    """True when the optional ``uiautomation`` backend can be imported."""
    if sys.platform != "win32":
        return False
    try:
        import uiautomation  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _uia_root() -> A11yNode | None:  # pragma: no cover - requires Windows + uiautomation
    import uiautomation as auto

    ctrl = auto.GetForegroundControl()
    return _UIANode(ctrl) if ctrl is not None else None


def elements_from_tree(root: A11yNode, window: WindowInfo, timestamp_ms: int = 0,
                       max_elements: int = 300, max_depth: int = 12) -> list[OCRElement]:
    """Flatten an accessibility tree into named, on-window :class:`OCRElement` s.

    Boxes are normalized to ``window.bounds`` (falling back to the root's
    rectangle) and clipped to ``[0,1]``; off-window or unnamed nodes are skipped.
    """
    wx, wy, ww, wh = window.bounds
    if ww <= 0 or wh <= 0:
        l, t, r, b = root.rect
        wx, wy, ww, wh = l, t, r - l, b - t
    if ww <= 0 or wh <= 0:
        return []
    out: list[OCRElement] = []
    stack: list[tuple[A11yNode, int]] = [(root, 0)]
    while stack and len(out) < max_elements:
        node, depth = stack.pop()
        name = (node.name or "").strip()
        l, t, r, b = node.rect
        x0, y0 = max(0.0, (l - wx) / ww), max(0.0, (t - wy) / wh)
        x1, y1 = min(1.0, (r - wx) / ww), min(1.0, (b - wy) / wh)
        if name and x1 > x0 and y1 > y0:
            out.append(OCRElement(
                text=redact_text(name),
                bbox_xyxy_normalized=(x0, y0, x1, y1),
                confidence=1.0,
                source="accessibility",
                ui_role=ROLE_MAP.get(node.control_type, node.control_type.replace("Control", "").lower() or None),
                window_id=window.window_id,
                timestamp_ms=timestamp_ms,
            ))
        if depth < max_depth:
            try:
                kids = list(node.children())
            except Exception:  # noqa: BLE001
                kids = []
            stack.extend((k, depth + 1) for k in reversed(kids))
    return out


def read_accessibility(window: WindowInfo, consent: ConsentManager | None = None,
                       cfg: CaptureConfig | None = None, timestamp_ms: int = 0,
                       root_fn: Callable[[], A11yNode | None] | None = None,
                       max_elements: int = 300, max_depth: int = 12) -> list[OCRElement]:
    """Accessibility elements for the foreground ``window``; ``[]`` when unavailable.

    When ``consent`` is given, ``SCREEN_CAPTURE`` must be allowed (not paused).
    Excluded windows always return ``[]``. ``root_fn`` injects a tree for tests.
    """
    if consent is not None and not consent.allowed(Scope.SCREEN_CAPTURE):
        return []
    if is_excluded(window, cfg)[0]:
        return []
    if root_fn is None:
        if not uia_available():
            return []
        root_fn = _uia_root
    try:
        root = root_fn()
        if root is None:
            return []
        return elements_from_tree(root, window, timestamp_ms, max_elements, max_depth)
    except Exception:  # noqa: BLE001 - accessibility is best-effort
        return []
