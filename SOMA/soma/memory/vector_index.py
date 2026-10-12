"""Cosine vector index: numpy brute force by default, optional FAISS backend.

Permission filtering happens **before** scoring: when ``allowed_ids`` is given,
only those rows are ever compared with the query, so a blocked item can never
influence (or leak through) a score. Filtered searches always use the exact
numpy path; FAISS is used only for unfiltered searches when installed.

Persistence is a single ``.npz`` (ids + float32 matrix, no pickles), written
atomically.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Iterable

import numpy as np


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except Exception:
        return False
    return True


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class VectorIndex:
    """In-memory id -> unit-vector index with cosine search.

    ``backend`` is ``"numpy"`` (default), ``"faiss"`` (requires faiss-cpu) or
    ``"auto"`` (faiss when importable). ``path`` is optional; :meth:`persist`
    writes there.
    """

    def __init__(self, dim: int, path: str | Path | None = None, backend: str = "numpy") -> None:
        self.dim = int(dim)
        self.path = Path(path) if path else None
        if backend == "auto":
            backend = "faiss" if _faiss_available() else "numpy"
        if backend == "faiss" and not _faiss_available():
            raise ImportError("faiss backend requested but faiss is not installed")
        self.backend = backend
        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._mat = np.zeros((0, self.dim), dtype=np.float32)
        self._faiss = None  # lazily rebuilt

    # -- mutation
    def add(self, item_id: str, vec: np.ndarray) -> None:
        """Insert or replace ``item_id``'s vector (stored L2-normalised)."""
        v = _unit(vec)
        if v.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {v.shape[0]}")
        with self._lock:
            if item_id in self._pos:
                self._mat[self._pos[item_id]] = v
            else:
                self._pos[item_id] = len(self._ids)
                self._ids.append(item_id)
                self._mat = np.vstack([self._mat, v[None, :]])
            self._faiss = None

    def remove(self, item_id: str) -> bool:
        """Delete ``item_id``; returns False when absent."""
        with self._lock:
            i = self._pos.pop(item_id, None)
            if i is None:
                return False
            self._ids.pop(i)
            self._mat = np.delete(self._mat, i, axis=0)
            self._pos = {k: j for j, k in enumerate(self._ids)}
            self._faiss = None
            return True

    def rebuild(self, items: Iterable[tuple[str, np.ndarray]]) -> None:
        """Replace all contents with ``(id, vector)`` pairs."""
        latest: dict[str, np.ndarray] = {}
        for item_id, vec in items:
            v = _unit(vec)
            if v.shape[0] != self.dim:
                raise ValueError(f"expected dim {self.dim}, got {v.shape[0]}")
            latest[str(item_id)] = v
        with self._lock:
            self._ids = list(latest)
            self._pos = {k: j for j, k in enumerate(self._ids)}
            self._mat = (np.stack(list(latest.values())).astype(np.float32) if latest
                         else np.zeros((0, self.dim), dtype=np.float32))
            self._faiss = None

    def clear(self) -> None:
        self.rebuild([])

    # -- queries
    def __len__(self) -> int:
        with self._lock:
            return len(self._ids)

    def __contains__(self, item_id: object) -> bool:
        with self._lock:
            return item_id in self._pos

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._ids)

    def get(self, item_id: str) -> np.ndarray | None:
        """Stored unit vector for ``item_id`` (a copy) or None."""
        with self._lock:
            i = self._pos.get(item_id)
            return None if i is None else self._mat[i].copy()

    def search(self, query: np.ndarray, k: int,
               allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        """Top-``k`` ``(id, cosine)`` pairs, best first.

        With ``allowed_ids`` the candidate rows are selected first and only
        those are scored.
        """
        if k <= 0:
            return []
        q = _unit(query)
        if q.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {q.shape[0]}")
        with self._lock:
            if allowed_ids is not None:
                rows = [self._pos[i] for i in allowed_ids if i in self._pos]
                if not rows:
                    return []
                rows.sort()
                ids = [self._ids[r] for r in rows]
                mat = self._mat[rows]                 # only permitted rows are scored
                return self._topk(ids, mat @ q, k)
            if not self._ids:
                return []
            if self.backend == "faiss":
                return self._faiss_search(q, k)
            return self._topk(self._ids, self._mat @ q, k)

    @staticmethod
    def _topk(ids: list[str], scores: np.ndarray, k: int) -> list[tuple[str, float]]:
        k = min(k, len(ids))
        if k < len(ids):
            part = np.argpartition(-scores, k - 1)[:k]
        else:
            part = np.arange(len(ids))
        order = part[np.argsort(-scores[part], kind="stable")]
        return [(ids[i], float(scores[i])) for i in order]

    def _faiss_search(self, q: np.ndarray, k: int) -> list[tuple[str, float]]:
        import faiss

        if self._faiss is None:
            index = faiss.IndexFlatIP(self.dim)
            index.add(self._mat)
            self._faiss = index
        scores, idx = self._faiss.search(q[None, :], min(k, len(self._ids)))
        return [(self._ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]

    # -- persistence
    def save(self, path: str | Path) -> Path:
        """Atomically write ``ids`` and ``vectors`` to an ``.npz`` file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        with self._lock:
            ids = np.array(self._ids, dtype=np.str_) if self._ids else np.zeros((0,), dtype="<U1")
            with open(tmp, "wb") as fh:
                np.savez(fh, ids=ids, vectors=self._mat, dim=np.array([self.dim]))
        os.replace(tmp, p)
        return p

    def persist(self) -> Path | None:
        """Save to :attr:`path` if one was configured."""
        return self.save(self.path) if self.path else None

    @classmethod
    def load(cls, path: str | Path, backend: str = "numpy") -> "VectorIndex":
        """Load an index written by :meth:`save` (``path`` becomes its persist target)."""
        with np.load(Path(path), allow_pickle=False) as data:
            dim = int(data["dim"][0])
            idx = cls(dim, path=path, backend=backend)
            idx.rebuild(zip([str(i) for i in data["ids"]], data["vectors"]))
        return idx

    @classmethod
    def open(cls, path: str | Path, dim: int, backend: str = "numpy") -> "VectorIndex":
        """Load ``path`` if it exists, else create an empty index that persists there."""
        p = Path(path)
        if p.exists():
            idx = cls.load(p, backend=backend)
            if idx.dim != dim:
                raise ValueError(f"index at {p} has dim {idx.dim}, expected {dim}")
            return idx
        return cls(dim, path=p, backend=backend)
