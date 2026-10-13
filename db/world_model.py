"""World model: what Pette has learned about this user from the cache, and how it changes her behaviour.

Each method is a small, explainable rule over the last 30 days of facts. Defaults apply until there's
enough evidence, so a fresh install behaves exactly like the spec.
"""
import statistics
import time

from .store import Store

DEFAULT_BLINK_ON = 0.6
MIN_BASELINE_SAMPLES = 300
DEFAULT_COOLDOWN_S = 600
DEFAULT_SEND_DELAY_S = 4.0


class WorldModel:
    def __init__(self, store: Store, clock=time.time):
        self.store = store
        self._clock = clock

    # --- messaging --------------------------------------------------------
    def preferred_platform(self, recipient: str) -> str | None:
        """The platform you actually send to this person on most (sent counts double over drafts)."""
        rows = self.store.query(
            "SELECT platform, SUM(CASE outcome WHEN 'sent' THEN 2 WHEN 'drafted' THEN 1 ELSE 0 END) AS score "
            "FROM messages WHERE recipient = ? GROUP BY platform HAVING score > 0 ORDER BY score DESC, platform",
            (recipient.strip().lower(),))
        return rows[0][0] if rows else None

    def known_contacts(self, limit: int = 8) -> list[tuple[str, str]]:
        """Most-messaged people with their usual platform, for the AI prompt and the offline drafter."""
        rows = self.store.query(
            "SELECT recipient, COUNT(*) AS n FROM messages WHERE outcome IN ('sent', 'drafted') "
            "GROUP BY recipient ORDER BY n DESC, recipient LIMIT ?", (limit,))
        return [(r, self.preferred_platform(r) or "") for r, _ in rows]

    def send_rate(self) -> float | None:
        rows = self.store.query("SELECT outcome, COUNT(*) FROM messages WHERE outcome IN ('sent','cancelled') "
                                "GROUP BY outcome")
        d = dict(rows)
        total = d.get("sent", 0) + d.get("cancelled", 0)
        return d.get("sent", 0) / total if total else None

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

    def outcome_counts(self) -> dict[str, int]:
        rows = self.store.query("SELECT outcome, COUNT(*) FROM messages GROUP BY outcome")
        return {k: 0 for k in ("drafted", "sent", "cancelled", "failed")} | dict(rows)

    def send_delay_s(self) -> float:
        """Seconds Pette waits (with Cancel showing) before dispatching. You cancel a lot -> more time to catch it;
        you almost never cancel -> snappier. Needs 5 decided messages before it moves off the default."""
        c = self.outcome_counts()
        decided = c["sent"] + c["cancelled"]
        if decided < 5:
            return DEFAULT_SEND_DELAY_S
        cancel_rate = c["cancelled"] / decided
        if cancel_rate >= 0.3:
            return 6.0
        if cancel_rate <= 0.05:
            return 2.5
        return DEFAULT_SEND_DELAY_S

    # --- interruption -----------------------------------------------------
    def offer_cooldown_s(self, base: float = DEFAULT_COOLDOWN_S) -> float:
        """Back off proactive offers you keep declining; offer sooner if you usually accept."""
        rows = self.store.query("SELECT outcome FROM offers WHERE kind = 'proactive' AND outcome != 'shown' "
                                "ORDER BY ts DESC LIMIT 20")
        if len(rows) < 5:
            return base
        rate = sum(1 for (o,) in rows if o == "accepted") / len(rows)
        if rate < 0.25:
            return base * 3
        if rate < 0.5:
            return base * 1.5
        if rate > 0.75:
            return max(300, base * 0.5)
        return base

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
        lines = []
        contacts = self.known_contacts(5)
        if contacts:
            lines.append("Usual contacts: " + ", ".join(f"{r.title()} ({p or '?'})" for r, p in contacts))
        rate = self.send_rate()
        if rate is not None:
            lines.append(f"You send {rate:.0%} of drafts")
        thr = self.blink_threshold()
        lines.append(f"Eyes-closed threshold: {thr}" + (" (calibrated)" if thr != DEFAULT_BLINK_ON else " (default)"))
        lines.append(f"Send countdown: {self.send_delay_s():g}s")
        p = self.presence_today()
        lines.append(f"Last 24h: away {p['away_episodes']}x"
                     + (f" (median {p['median_away_s']}s)" if p["median_away_s"] else "")
                     + f", eyes closed {p['eyes_closed']}x, pinches {p['pinches']}")
        return "\n".join(lines)

    # --- UI ---------------------------------------------------------------
    def soma_counts(self) -> dict[str, int]:
        """SOMA suggestion outcomes (pet/soma.py logs `offers` rows with kind 'soma:<template>')."""
        rows = self.store.query("SELECT outcome, COUNT(*) FROM offers WHERE kind LIKE 'soma:%' GROUP BY outcome")
        by = dict(rows)
        return {k: by.get(k, 0) for k in ("shown", "accepted", "dismissed", "ignored")}

    def snapshot(self) -> dict:
        """Everything the world-model window shows: learned facts plus the behaviour each one drives."""
        counts = self.outcome_counts()
        contacts = []
        for name, platform in self.known_contacts(8):
            by = dict(self.store.query("SELECT outcome, COUNT(*) FROM messages WHERE recipient = ? GROUP BY outcome",
                                       (name,)))
            contacts.append({"name": name.title(), "platform": platform, "sent": by.get("sent", 0),
                             "drafted": by.get("drafted", 0), "cancelled": by.get("cancelled", 0)})
        samples = self.store.query("SELECT COALESCE(SUM(samples), 0) FROM "
                                   "(SELECT samples FROM blink_baseline ORDER BY ts DESC LIMIT 20)")[0][0]
        thr = self.blink_threshold()
        delay = self.send_delay_s()
        decided = counts["sent"] + counts["cancelled"]
        rules = [{"when": f"You say “message {c['name']}” without a platform",
                  "then": f"Pette uses {c['platform'].title()}",
                  "why": f"{c['sent']} sent, {c['drafted']} drafted to {c['name']} there"}
                 for c in contacts if c["platform"]][:4]
        rules.append({"when": "A draft is ready", "then": f"Pette waits {delay:g}s before sending (Cancel shows)",
                      "why": (f"you cancelled {counts['cancelled']} of {decided} messages" if decided >= 5
                              else "default until 5 messages are decided")})
        rules.append({"when": "Your eyes look closed on camera", "then": f"blink score above {thr}",
                      "why": f"calibrated from {samples} open-eye samples" if thr != DEFAULT_BLINK_ON
                             else "default until the camera has seen enough open-eye frames"})
        soma = self.soma_counts()
        if soma["shown"]:
            rules.append({"when": "SOMA sees a next step on your screen",
                          "then": "Pette offers it as a card; nothing runs without “Do it”",
                          "why": f"{soma['shown']} shown: {soma['accepted']} done, {soma['dismissed']} not now, "
                                 f"{soma['ignored']} ignored"})
        recent = [{"ts": ts, "recipient": r.title(), "platform": p, "outcome": o} for ts, r, p, o in self.store.query(
            "SELECT ts, recipient, platform, outcome FROM messages ORDER BY ts DESC LIMIT 8")]
        events = [{"ts": ts, "kind": k} for ts, k in self.store.query(
            "SELECT ts, kind FROM events ORDER BY ts DESC LIMIT 8")]
        return {"contacts": contacts, "outcomes": counts, "send_rate": self.send_rate(), "send_delay_s": delay,
                "blink_threshold": thr, "blink_calibrated": thr != DEFAULT_BLINK_ON, "baseline_samples": samples,
                "presence": self.presence_today(), "rules": rules, "soma": soma, "recent": recent, "events": events,
                "updated": self._clock()}
