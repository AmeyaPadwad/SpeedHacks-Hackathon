"""World model: what Pette has learned about this user from the cache, and how it changes her behaviour.

Each method is a small, explainable rule over the last 30 days of facts. Defaults apply until there's
enough evidence, so a fresh install behaves exactly like the spec.
"""
import statistics
import time

from .store import Store

DEFAULT_BLINK_ON = 0.6
MIN_BASELINE_SAMPLES = 300
MUTE_MIN_DECIDED = 4        # decisions on one kind of suggestion before Pette may stop showing it
MUTE_DISMISS_RATE = 0.8     # ... and the share of them that were "Not now"
SOMA_PREFIX = "soma:"

KIND_LABELS = {
    "review_changes": "Review changes", "write_tests": "Write a test", "commit_work": "Commit work",
    "take_notes": "Save notes", "explain_error": "Explain error", "open_relevant_file": "Open file",
    "compare_with_prior_work": "Your files", "suggest_next_step": "Next step",
    "ask_clarifying_question": "Question",
}


def kind_label(kind: str) -> str:
    k = kind[len(SOMA_PREFIX):] if kind.startswith(SOMA_PREFIX) else kind
    return KIND_LABELS.get(k, k.replace("_", " ").capitalize())


class WorldModel:
    def __init__(self, store: Store, clock=time.time):
        self.store = store
        self._clock = clock

    # --- webcam calibration -----------------------------------------------
    def blink_threshold(self) -> float:
        """Eyes-closed threshold tuned to this user's open-eye baseline (glasses, lighting, eye shape).

        Threshold sits 0.15 above the open-eye 95th percentile, kept within 0.5..0.85.
        """
        rows = self.store.query("SELECT p95, samples FROM blink_baseline ORDER BY ts DESC LIMIT 20")
        total = sum(s for _, s in rows)
        if total < MIN_BASELINE_SAMPLES:
            return DEFAULT_BLINK_ON
        p95 = sum(p * s for p, s in rows) / total
        return round(min(0.85, max(0.5, p95 + 0.15)), 2)

    # --- SOMA suggestions -------------------------------------------------
    def soma_counts(self) -> dict[str, int]:
        """Suggestion outcomes (pet/soma.py logs `offers` rows with kind 'soma:<kind>')."""
        by = dict(self.store.query("SELECT outcome, COUNT(*) FROM offers WHERE kind LIKE 'soma:%' GROUP BY outcome"))
        return {k: by.get(k, 0) for k in ("shown", "accepted", "dismissed", "ignored")}

    def accept_rate(self) -> float | None:
        c = self.soma_counts()
        decided = c["accepted"] + c["dismissed"]
        return c["accepted"] / decided if decided else None

    def kinds(self) -> list[dict]:
        """Per kind of suggestion: how often it was shown, done or put off."""
        rows = self.store.query("SELECT kind, outcome, COUNT(*) FROM offers WHERE kind LIKE 'soma:%' "
                                "GROUP BY kind, outcome")
        out: dict[str, dict] = {}
        for kind, outcome, n in rows:
            k = out.setdefault(kind, {"kind": kind, "label": kind_label(kind), "shown": 0, "accepted": 0,
                                      "dismissed": 0, "ignored": 0})
            if outcome in k:
                k[outcome] = n
        return sorted(out.values(), key=lambda k: (-k["shown"], k["kind"]))

    def muted_kinds(self) -> set[str]:
        """Kinds you nearly always answer "Not now": Pette stops showing them (Clear memory resets this)."""
        muted = set()
        for k in self.kinds():
            decided = k["accepted"] + k["dismissed"]
            if decided >= MUTE_MIN_DECIDED and k["dismissed"] / decided >= MUTE_DISMISS_RATE:
                muted.add(k["kind"])
        return muted

    # --- presence ---------------------------------------------------------
    def presence_today(self) -> dict:
        start = self._clock() - 86400
        away = [d for (d,) in self.store.query(
            "SELECT duration FROM events WHERE kind = 'away_ended' AND ts >= ? AND duration IS NOT NULL", (start,))]
        count = lambda k: self.store.query("SELECT COUNT(*) FROM events WHERE kind = ? AND ts >= ?", (k, start))[0][0]
        return {
            "away_episodes": len(away),
            "median_away_s": round(statistics.median(away), 1) if away else None,
            "eyes_closed": count("eyes_closed"),
            "pinches": count("pinch"),
        }

    def summary(self) -> str:
        c = self.soma_counts()
        lines = [f"SOMA suggestions: {c['shown']} shown, {c['accepted']} done, {c['dismissed']} not now"]
        muted = self.muted_kinds()
        if muted:
            lines.append("Not shown any more: " + ", ".join(sorted(kind_label(k) for k in muted)))
        thr = self.blink_threshold()
        lines.append(f"Eyes-closed threshold: {thr}" + (" (calibrated)" if thr != DEFAULT_BLINK_ON else " (default)"))
        p = self.presence_today()
        lines.append(f"Last 24h: away {p['away_episodes']}x"
                     + (f" (median {p['median_away_s']}s)" if p["median_away_s"] else "")
                     + f", eyes closed {p['eyes_closed']}x, pinches {p['pinches']}")
        return "\n".join(lines)

    # --- UI ---------------------------------------------------------------
    def snapshot(self) -> dict:
        """Everything the world-model window shows: learned facts plus the behaviour each one drives."""
        samples = self.store.query("SELECT COALESCE(SUM(samples), 0) FROM "
                                   "(SELECT samples FROM blink_baseline ORDER BY ts DESC LIMIT 20)")[0][0]
        thr = self.blink_threshold()
        soma, kinds, muted = self.soma_counts(), self.kinds(), self.muted_kinds()
        rules = [{"when": "SOMA sees a next step on your screen",
                  "then": "Pette shows it as a card; nothing happens until you press “Do it” or pinch",
                  "why": (f"{soma['shown']} shown: {soma['accepted']} done, {soma['dismissed']} not now, "
                          f"{soma['ignored']} ignored") if soma["shown"] else "no suggestions yet"}]
        for k in kinds:
            if k["kind"] in muted:
                rules.append({"when": f"SOMA suggests “{k['label']}”", "then": "Pette doesn't show it any more",
                              "why": f"you said Not now {k['dismissed']} of {k['accepted'] + k['dismissed']} times"})
        rules.append({"when": "Your eyes look closed on camera", "then": f"blink score above {thr}",
                      "why": f"calibrated from {samples} open-eye samples" if thr != DEFAULT_BLINK_ON
                             else "default until the camera has seen enough open-eye frames"})
        recent = [{"ts": ts, "kind": kind_label(k), "outcome": o} for ts, k, o in self.store.query(
            "SELECT ts, kind, outcome FROM offers WHERE kind LIKE 'soma:%' AND outcome != 'shown' "
            "ORDER BY ts DESC LIMIT 8")]
        events = [{"ts": ts, "kind": k} for ts, k in self.store.query(
            "SELECT ts, kind FROM events ORDER BY ts DESC LIMIT 8")]
        return {"soma": soma, "accept_rate": self.accept_rate(), "kinds": kinds[:8],
                "muted": sorted(kind_label(k) for k in muted),
                "blink_threshold": thr, "blink_calibrated": thr != DEFAULT_BLINK_ON, "baseline_samples": samples,
                "presence": self.presence_today(), "rules": rules, "recent": recent, "events": events,
                "updated": self._clock()}
