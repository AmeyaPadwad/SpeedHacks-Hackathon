"""Darwin Search (https://darwin.so/docs/search/using-results): find agent capabilities for a goal.

"Search Darwin for …" / "find a tool that …" -> POST /api/v3/search -> the top matches become a Slack
draft (pinch or the countdown sends it). Search only finds and ranks; it never connects or acts.
Results are untrusted: names/reasons are shown as text, connection details are never followed.
"""
import re

import requests

from .brain import MAX_MESSAGE, Draft

SEARCH_URL = "https://api.darwin.so/api/v3/search"
TIMEOUT_S = 20
MAX_RESULTS = 5
INTENT = re.compile(r"\b(darwin|search (?:for|darwin)|find (?:me )?(?:an? )?(?:agent|tool|capabilit\w*|service|api))\b",
                    re.I)


class DarwinError(Exception):
    pass


def wants_darwin(command: str) -> bool:
    return bool(INTENT.search(command))


def query_from(command: str) -> str:
    """Strip the trigger words and any 'and send it to slack' tail; what's left is the goal."""
    q = re.sub(r"\b(?:and )?(?:send|post|share)\b.*\b(?:slack|discord|email)\b.*$", "", command, flags=re.I)
    q = re.sub(r"^\s*(?:hey\s+)?(?:pette[,]?\s+)?(?:please\s+)?(?:search\s+darwin\s+for|search\s+for|use\s+darwin\s+to\s+find|"
               r"find\s+(?:me\s+)?(?:an?\s+)?(?:agent|tool|capabilit\w*|service|api)\s+(?:that|to|for)?)\s*", "", q, flags=re.I)
    q = re.sub(r"\b(?:on|with|using|in)\s+darwin\b", "", q, flags=re.I)
    return q.strip(" .,") or command.strip()


def search(query: str, post=requests.post) -> dict:
    try:
        r = post(SEARCH_URL, json={"query": query, "maxResults": MAX_RESULTS},
                 headers={"Content-Type": "application/json"}, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        raise DarwinError(f"Couldn't reach Darwin: {type(e).__name__}") from e
    if r.status_code >= 300:
        raise DarwinError(f"Darwin search failed (HTTP {r.status_code}).")
    try:
        return r.json()
    except ValueError as e:
        raise DarwinError("Darwin returned something that isn't JSON.") from e


def summarize(query: str, data: dict) -> str:
    resp = data.get("response") or {}
    agents = [a for a in resp.get("agents") or [] if isinstance(a, dict)][:MAX_RESULTS]
    if not agents:
        reason = resp.get("noMatchReason") or "no matching capabilities"
        return f"Darwin search for “{query}”: {reason}."
    lines = [f"Darwin search for “{query}” ({resp.get('assessment') or data.get('status') or 'done'}):"]
    for a in agents:
        ready = "ready" if a.get("canStartThread") else (a.get("readiness") or "unknown")
        why = (a.get("reasons") or [""])[0]
        lines.append(f"{a.get('rank', '•')}. {str(a.get('name', '?'))[:80]} ({ready})" + (f": {str(why)[:140]}" if why else ""))
    if data.get("searchId"):
        lines.append(f"searchId: {data['searchId']}")
    return "\n".join(lines)[:MAX_MESSAGE]


def draft(command: str, recipient: str = "team", post=requests.post) -> Draft:
    """Search Darwin and turn the results into a Slack draft (sent by pinch or the countdown)."""
    q = query_from(command)
    return Draft("slack", recipient, "", summarize(q, search(q, post=post)))
