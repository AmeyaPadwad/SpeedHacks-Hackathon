"""Evaluation metrics from build plan section 12 (pure Python, no torch needed).

Retrieval (recall@k, nDCG@k), OCR (CER/WER), UI grounding (point-in-target),
personalization (acceptance/edit rates) and silence policy (unwanted
suggestions per active hour).
"""

from __future__ import annotations

import math
from typing import Hashable, Iterable, Mapping, Sequence

from soma.types import FeedbackOutcome


# ---------------------------------------------------------------- retrieval

def recall_at_k(retrieved: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """Fraction of relevant ids that appear in the top ``k`` retrieved. 1.0 if none relevant."""
    rel = set(relevant)
    if not rel:
        return 1.0
    return len(rel.intersection(retrieved[:k])) / len(rel)


def dcg(gains: Sequence[float]) -> float:
    return sum((2.0 ** g - 1.0) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(retrieved: Sequence[Hashable], relevance: Mapping[Hashable, float] | Iterable[Hashable],
              k: int) -> float:
    """Normalized DCG@k. ``relevance`` maps id -> graded gain (or is a set of relevant ids)."""
    if not isinstance(relevance, Mapping):
        relevance = {r: 1.0 for r in relevance}
    ideal = dcg(sorted(relevance.values(), reverse=True)[:k])
    if ideal <= 0.0:
        return 1.0
    return dcg([float(relevance.get(r, 0.0)) for r in retrieved[:k]]) / ideal


# ---------------------------------------------------------------- OCR

def edit_distance(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    """Levenshtein distance between two sequences (characters or words)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate = edits / len(reference)."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate on whitespace tokens."""
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


# ---------------------------------------------------------------- grounding

def point_in_target_accuracy(points: Sequence[tuple[float, float]],
                             boxes: Sequence[tuple[float, float, float, float]]) -> float:
    """Fraction of predicted ``(x, y)`` points inside their target ``xyxy`` box."""
    if len(points) != len(boxes):
        raise ValueError("points and boxes must align")
    if not points:
        return 0.0
    hits = sum(1 for (x, y), (x0, y0, x1, y1) in zip(points, boxes) if x0 <= x <= x1 and y0 <= y <= y1)
    return hits / len(points)


# ---------------------------------------------------------------- personalization

def acceptance_edit_rates(outcomes: Iterable[FeedbackOutcome | str]) -> dict[str, float]:
    """Rates over all shown suggestions: accepted, edited, rejected, ignored.

    ``acceptance_rate`` counts accepted only; ``useful_rate`` counts accepted
    + edited. Ignored suggestions are reported separately and are ambiguous.
    """
    counts = {o.value: 0 for o in FeedbackOutcome}
    for o in outcomes:
        counts[FeedbackOutcome(o).value] += 1
    n = sum(counts.values())
    if n == 0:
        return {"n": 0, "acceptance_rate": 0.0, "edit_rate": 0.0, "rejection_rate": 0.0,
                "ignore_rate": 0.0, "useful_rate": 0.0}
    acc, ed = counts["accepted"], counts["edited"]
    return {"n": n, "acceptance_rate": acc / n, "edit_rate": ed / n,
            "rejection_rate": counts["rejected"] / n, "ignore_rate": counts["ignored"] / n,
            "useful_rate": (acc + ed) / n}


def pairwise_win_rate(wins: int, losses: int, ties: int = 0) -> float:
    """Preference win rate, counting ties as half a win."""
    n = wins + losses + ties
    return (wins + 0.5 * ties) / n if n else 0.5


# ---------------------------------------------------------------- silence policy

def unwanted_suggestions_per_active_hour(n_unwanted: int, active_seconds: float) -> float:
    """Unwanted (rejected / should-have-been-silent) suggestions per active hour."""
    if active_seconds <= 0:
        return 0.0
    return n_unwanted / (active_seconds / 3600.0)
