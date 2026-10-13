"""App-context next steps, so ordinary screens have something worth ranking.

``generate_candidates`` only proposes actions for visible errors, authorized
file references and retrieved memories, so on most real screens (an editor
with clean code, a terminal, a web page) it proposes nothing and SOMA can only
stay silent. With ``PolicyConfig.context_templates`` the runtime appends these
fixed templates, chosen from the foreground app's kind and a few cheap screen
signals. The model still ranks them against ``NO_SUGGESTION`` and the gate
still applies.

Same safety rules as the generator: the text is a fixed template, screen text
appears only as a short sanitized quote, injection-like titles are dropped,
and nothing here executes. ``Candidate.template`` carries the template id so a
companion app (Pette) can map an *accepted* suggestion to its own workflow.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from soma.policy.candidate_generator import (
    ROLE_ERROR,
    _cap,
    _quote,
    detect_role,
    is_suspicious,
    sanitize_snippet,
)
from soma.types import ActionType, Candidate, ScreenObservation

SHARE_CHANGES = "share_changes"
ASK_REVIEW = "ask_review"
TESTS_PASS = "tests_pass"
ASK_ABOUT_ERROR = "ask_about_error"
SEARCH_TOOLS = "search_tools"
TEMPLATES = (SHARE_CHANGES, ASK_REVIEW, TESTS_PASS, ASK_ABOUT_ERROR, SEARCH_TOOLS)

EDITORS = {"code", "code - insiders", "cursor", "windsurf", "pycharm64", "idea64", "webstorm64",
           "devenv", "sublime_text", "notepad++", "zed", "rider64", "clion64"}
TERMINALS = {"windowsterminal", "openconsole", "cmd", "powershell", "pwsh", "wezterm-gui", "alacritty",
             "mintty", "conhost", "tabby", "hyper", "warp"}
BROWSERS = {"chrome", "msedge", "firefox", "brave", "arc", "opera", "vivaldi"}

_FILE_RE = re.compile(r"[\w.-]{1,80}\.[A-Za-z][A-Za-z0-9]{0,7}")
_TESTS_PASS_RE = re.compile(r"\b\d+\s+passed\b|\btests?\s+passed\b|\ball tests pass", re.I)
_BROWSER_SUFFIX_RE = re.compile(
    r"\s+[-—|]\s+(google chrome|microsoft​? edge|mozilla firefox|brave|arc|opera|vivaldi).*$", re.I)


def app_kind(app_name: str) -> str:
    """"editor" | "terminal" | "browser" | "" from a process name like ``Code.exe``."""
    name = (app_name or "").strip().lower()
    name = name[:-4] if name.endswith(".exe") else name
    if name in EDITORS:
        return "editor"
    if name in TERMINALS:
        return "terminal"
    if name in BROWSERS:
        return "browser"
    return ""


def editor_file(title: str) -> str | None:
    """The file name an editor title shows first (``"● main.py - Pette - Visual Studio Code"``)."""
    head = re.split(r"\s+[-—]\s+", (title or "").strip(), maxsplit=1)[0]
    m = _FILE_RE.search(head.lstrip("●*• "))
    return m.group(0) if m else None


def page_topic(title: str) -> str | None:
    """A browser tab title without the browser's name, if it is safe to quote."""
    topic = _BROWSER_SUFFIX_RE.sub("", (title or "").strip())
    topic = sanitize_snippet(topic, 60)
    if len(topic) < 3 or is_suspicious(topic) or topic.lower() in ("new tab", "[redacted]"):
        return None
    return topic


def context_candidates(obs: ScreenObservation) -> list[Candidate]:
    """Template candidates for the foreground app (may be empty)."""
    kind = app_kind(obs.window.app_name)
    if not kind:
        return []
    texts = [e.text for e in obs.elements if not is_suspicious(e.text)]
    has_error = any(detect_role(e) == ROLE_ERROR for e in obs.elements)
    out: list[Candidate] = []

    if kind in ("editor", "terminal") and has_error:
        out.append(Candidate(
            action=ActionType.ASK_CLARIFYING_QUESTION,
            text="Ask the team for help with the error on screen?",
            evidence=["an error is visible"], template=ASK_ABOUT_ERROR))

    if kind == "editor":
        f = editor_file(obs.window.title)
        label = _quote(f, 60) if f else "this file"
        out.append(Candidate(
            action=ActionType.SUGGEST_NEXT_STEP,
            text=_cap(f"Send the team a summary of your changes to {label}"),
            evidence=[f"editing {label}"], template=SHARE_CHANGES))
        out.append(Candidate(
            action=ActionType.SUGGEST_NEXT_STEP,
            text=_cap(f"Ask a teammate to review {label}"),
            evidence=[f"editing {label}"], template=ASK_REVIEW))

    if kind == "terminal" and not has_error and any(_TESTS_PASS_RE.search(t) for t in texts):
        out.append(Candidate(
            action=ActionType.SUGGEST_NEXT_STEP,
            text="Tell the team the tests pass",
            evidence=["test run passed"], template=TESTS_PASS))

    if kind == "browser":
        topic = page_topic(obs.window.title)
        if topic:
            out.append(Candidate(
                action=ActionType.SUGGEST_NEXT_STEP,
                text=_cap(f"Search Darwin for tools related to {_quote(topic, 60)}"),
                evidence=[topic], template=SEARCH_TOOLS))
    return out


def add_context_candidates(obs: ScreenObservation, candidates: Sequence[Candidate],
                           max_candidates: int) -> list[Candidate]:
    """``candidates`` plus context templates, de-duplicated by key and bounded."""
    out = list(candidates)[:max_candidates]
    keys = {c.key for c in out}
    for c in context_candidates(obs):
        if len(out) >= max_candidates:
            break
        if c.key not in keys:
            keys.add(c.key)
            out.append(c)
    return out


__all__ = ["TEMPLATES", "add_context_candidates", "app_kind", "context_candidates", "editor_file",
           "page_topic"]
