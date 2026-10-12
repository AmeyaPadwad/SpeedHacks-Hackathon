"""Neutral, minimal palette for the SOMA desktop UI: subtle greys and one accent.

The system font is used throughout; only colours and spacing are set here.
"""

from __future__ import annotations

ACCENT = "#2F5D8A"          # the one accent colour (chosen bar, primary button)
TEXT = "#262626"
MUTED = "#6E6E6E"
BORDER = "#D7D4CE"
SURFACE = "#FFFFFF"
BACKGROUND = "#F7F5F2"
BAR = "#B8B8B8"             # non-chosen bars
STALE = "#A0A0A0"
OK = "#4A6B3A"              # capturing
WARN = "#C99B3A"            # paused
OFF = "#6E6E6E"             # off

STATE_COLORS = {"Capturing": OK, "Paused": WARN, "Off": OFF}

STYLESHEET = f"""
QWidget {{ color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BACKGROUND}; }}
QFrame#card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px; }}
QFrame#card[stale="true"] {{ background: {BACKGROUND}; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#heading {{ font-size: 13pt; font-weight: 600; }}
QPushButton {{ padding: 5px 12px; border: 1px solid {BORDER}; border-radius: 4px; background: {SURFACE}; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {STALE}; background: {BACKGROUND}; }}
QPushButton#primary {{ background: {ACCENT}; color: white; border-color: {ACCENT}; }}
QPushButton#primary:disabled {{ background: {BORDER}; color: {MUTED}; }}
QPushButton#pause {{ font-size: 12pt; padding: 8px 20px; }}
QTableWidget {{ background: {SURFACE}; gridline-color: {BORDER}; border: 1px solid {BORDER}; }}
QHeaderView::section {{ background: {BACKGROUND}; border: none; border-bottom: 1px solid {BORDER};
                        padding: 4px; color: {MUTED}; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; background: {SURFACE}; }}
"""


def state_badge_style(state: str) -> str:
    """Stylesheet for the Capturing / Paused / Off pill."""
    color = STATE_COLORS.get(state, OFF)
    return (f"QLabel {{ color: white; background: {color}; border-radius: 9px;"
            f" padding: 3px 10px; font-weight: 600; }}")
