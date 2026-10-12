"""Permission-filtered retrieval and reranking of local memories.

Order of operations (plan sections 6 and 10):

1. **Permission prefilter** over metadata only: user-approved, not expired,
   source not blocked, documents/images still inside an allowed folder (which
   also requires FOLDER_INDEXING), feedback memories only with
   FEEDBACK_STORAGE, superseded conflicting preferences dropped.
2. **Vector search** over the allowed ids only (no other row is ever scored).
3. **Rerank** = relevance + recency (half-life) + source trust + preference boost.
4. A final ACL re-check; any failure is counted in ``permission_violations``
   (expected to stay 0) and dropped.

Retrieved summaries are untrusted task data, never instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import torch

from soma.config import MemoryConfig
from soma.consent import ConsentManager, Scope
from soma.memory.sqlite_store import DEFAULT_TRUST, MemoryStore, parse_iso, path_within
from soma.memory.vector_index import VectorIndex
from soma.types import MemoryItem, RetrievedMemory, SourceType

FILE_TYPES = {SourceType.DOCUMENT, SourceType.IMAGE}
CONFLICT_PREFIX = "conflict:"


@dataclass
class RerankWeights:
    relevance: float = 1.0
    recency: float = 0.15
    trust: float = 0.1
    preference: float = 0.1


def conflict_key(item: MemoryItem) -> str | None:
    """The ``conflict:<key>`` tag value of an item, if any."""
    for t in item.tags:
        if t.startswith(CONFLICT_PREFIX) and len(t) > len(CONFLICT_PREFIX):
            return t[len(CONFLICT_PREFIX):]
    return None


def _as_utc(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


class Retriever:
    """Retrieve top-k permitted memories for a query embedding."""

    def __init__(self, store: MemoryStore, index: VectorIndex, consent: ConsentManager,
                 cfg: MemoryConfig | None = None, weights: RerankWeights | None = None,
                 oversample: int = 4) -> None:
        self.store = store
        self.index = index
        self.consent = consent
        self.cfg = cfg or MemoryConfig()
        self.weights = weights or RerankWeights()
        self.oversample = max(1, oversample)
        self.permission_violations = 0
        self.last_allowed_count = 0

    # -- permissions
    def _blocked_roots(self) -> list[str]:
        return [s["root"] for s in self.store.list_sources() if s["blocked"] or not s["allowed"]]

    def is_permitted(self, item: MemoryItem, now: datetime | None = None,
                     blocked_roots: list[str] | None = None) -> bool:
        """Metadata-only ACL check for one item (``blocked_roots`` may be prefetched)."""
        now = _as_utc(now)
        if blocked_roots is None:
            blocked_roots = self._blocked_roots()
        if not item.user_approved:
            return False
        exp = parse_iso(item.expires_at)
        if exp is not None and exp <= now:
            return False
        if item.source_uri and any(path_within(item.source_uri, r) for r in blocked_roots):
            return False
        st = SourceType(item.source_type)
        if st in FILE_TYPES:
            if not item.source_uri or not self.consent.path_allowed(item.source_uri):
                return False
        if st is SourceType.FEEDBACK and not self.consent.allowed(Scope.FEEDBACK_STORAGE):
            return False
        return True

    def allowed_items(self, now: datetime | None = None) -> dict[str, MemoryItem]:
        """All items passing the ACL, with superseded conflicting memories removed."""
        now = _as_utc(now)
        blocked = self._blocked_roots()
        items = [i for i in self.store.list_items(approved_only=True)
                 if self.is_permitted(i, now, blocked)]
        winners: dict[str, MemoryItem] = {}
        for it in items:
            key = conflict_key(it)
            if key is None:
                continue
            cur = winners.get(key)
            if cur is None or (it.updated_at or "", it.created_at or "") > (cur.updated_at or "",
                                                                             cur.created_at or ""):
                winners[key] = it
        out: dict[str, MemoryItem] = {}
        for it in items:
            key = conflict_key(it)
            if key is not None and winners[key].id != it.id:
                continue                        # an older, conflicting preference
            out[it.id] = it
        return out

    # -- scoring
    def _recency(self, item: MemoryItem, now: datetime) -> float:
        t = parse_iso(item.updated_at) or parse_iso(item.created_at)
        if t is None:
            return 0.0
        age_days = max(0.0, (now - t).total_seconds() / 86400.0)
        half = max(1e-6, float(self.cfg.recency_half_life_days))
        return float(0.5 ** (age_days / half))

    def _trust(self, item: MemoryItem) -> float:
        if SourceType(item.source_type) in (SourceType.PREFERENCE, SourceType.FEEDBACK):
            return 1.0
        return self.store.source_trust(item.source_uri, DEFAULT_TRUST) if item.source_uri else DEFAULT_TRUST

    def score(self, item: MemoryItem, relevance: float, now: datetime) -> float:
        w = self.weights
        pref = 1.0 if SourceType(item.source_type) is SourceType.PREFERENCE else 0.0
        return (w.relevance * relevance + w.recency * self._recency(item, now)
                + w.trust * self._trust(item) + w.preference * pref)

    # -- main entry
    def retrieve(self, query_vec: np.ndarray, k: int | None = None,
                 now: datetime | None = None) -> list[RetrievedMemory]:
        """Top-``k`` permitted memories for ``query_vec`` (float ``[D]``), best first."""
        k = int(k if k is not None else self.cfg.top_k)
        if k <= 0:
            return []
        now = _as_utc(now)
        allowed = self.allowed_items(now)
        self.last_allowed_count = len(allowed)
        if not allowed:
            return []
        hits = self.index.search(query_vec, k * self.oversample, allowed_ids=set(allowed))
        blocked = self._blocked_roots()
        results: list[RetrievedMemory] = []
        for item_id, rel in hits:
            item = allowed.get(item_id)
            vec = self.index.get(item_id)
            if item is None or vec is None or not self.is_permitted(item, now, blocked):
                self.permission_violations += 1
                continue
            results.append(RetrievedMemory(item=item, embedding=vec.astype(np.float32),
                                           score=self.score(item, rel, now)))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]


def to_tensors(retrieved: list[RetrievedMemory], K: int, D: int,
               device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack memories into ``(embeddings [K, D] float32, padding_mask [K] bool)``.

    ``padding_mask`` follows the PyTorch convention: ``True`` = padding (ignored).
    Extra memories beyond ``K`` are dropped; missing slots are zero vectors.
    """
    emb = torch.zeros(K, D, dtype=torch.float32)
    mask = torch.ones(K, dtype=torch.bool)
    for i, r in enumerate(retrieved[:K]):
        v = np.asarray(r.embedding, dtype=np.float32).reshape(-1)
        if v.shape[0] != D:
            raise ValueError(f"memory embedding has dim {v.shape[0]}, expected {D}")
        emb[i] = torch.from_numpy(v.copy())
        mask[i] = False
    if device is not None:
        emb, mask = emb.to(device), mask.to(device)
    return emb, mask


Retriever.to_tensors = staticmethod(to_tensors)  # type: ignore[attr-defined]
