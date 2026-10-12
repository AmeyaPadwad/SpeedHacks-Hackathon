"""Bounded, stratified replay buffer for continual updates (Stage C/D).

* Per-(app, task) reservoir sampling keeps every stratum an unbiased sample of
  what it has seen, so one busy app cannot crowd out the others.
* A global ``capacity`` bound evicts from the largest stratum first.
* "Anchor" examples from prior competencies (curated / synthetic, never raw
  user content) live in their own bounded reservoir and are mixed into every
  sample to limit forgetting.
* Ambiguous (ignored) and non-consented examples are refused at ``add``.
"""

from __future__ import annotations

import random
from typing import Any

import torch

from soma.learning.feedback import LabeledExample

_SEP = "\x1f"
StratumKey = tuple[str, str]


def _key_str(key: StratumKey) -> str:
    return f"{key[0]}{_SEP}{key[1]}"


def _key_tuple(s: str) -> StratumKey:
    app, _, task = s.partition(_SEP)
    return app, task


class ReplayBuffer:
    """Reservoir buffer stratified by ``(app, task)`` with an anchor pool."""

    def __init__(self, capacity: int = 2000, *, per_stratum_capacity: int | None = None,
                 anchor_capacity: int = 500, anchor_fraction: float = 0.25, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.per_stratum_capacity = int(per_stratum_capacity or capacity)
        self.anchor_capacity = int(anchor_capacity)
        self.anchor_fraction = float(anchor_fraction)
        self._rng = random.Random(seed)
        self._strata: dict[StratumKey, list[LabeledExample]] = {}
        self._seen: dict[StratumKey, int] = {}
        self._anchors: list[LabeledExample] = []
        self._anchor_seen = 0
        self.rejected = 0

    # ------------------------------------------------------------ adding
    @staticmethod
    def accepts(ex: LabeledExample) -> bool:
        """True only for consented, unambiguous (non-ignored) examples."""
        return ex.usable_for_training

    def _reservoir(self, pool: list[LabeledExample], seen: int, cap: int, ex: LabeledExample) -> None:
        if len(pool) < cap:
            pool.append(ex)
            return
        j = self._rng.randrange(seen)
        if j < cap:
            pool[j] = ex

    def add(self, ex: LabeledExample) -> bool:
        """Add one example; returns False when it is refused (ambiguous / no consent)."""
        if not self.accepts(ex):
            self.rejected += 1
            return False
        key = (ex.app, ex.task)
        pool = self._strata.setdefault(key, [])
        self._seen[key] = self._seen.get(key, 0) + 1
        self._reservoir(pool, self._seen[key], self.per_stratum_capacity, ex)
        while len(self) > self.capacity:
            biggest = max(self._strata, key=lambda k: (len(self._strata[k]), _key_str(k)))
            victims = self._strata[biggest]
            victims.pop(self._rng.randrange(len(victims)))
        return True

    def extend(self, examples: list[LabeledExample]) -> int:
        """Add many; returns how many were accepted."""
        return sum(self.add(e) for e in examples)

    def add_anchor(self, ex: LabeledExample) -> bool:
        """Add a prior-competency anchor (must be unambiguous and consented/curated)."""
        if not self.accepts(ex):
            self.rejected += 1
            return False
        self._anchor_seen += 1
        self._reservoir(self._anchors, self._anchor_seen, self.anchor_capacity, ex)
        return True

    # ------------------------------------------------------------ queries
    def __len__(self) -> int:
        return sum(len(v) for v in self._strata.values())

    @property
    def n_anchors(self) -> int:
        return len(self._anchors)

    def strata(self) -> dict[StratumKey, int]:
        return {k: len(v) for k, v in sorted(self._strata.items(), key=lambda kv: _key_str(kv[0]))}

    def examples(self) -> list[LabeledExample]:
        return [e for k in sorted(self._strata, key=_key_str) for e in self._strata[k]]

    def anchors(self) -> list[LabeledExample]:
        return list(self._anchors)

    @staticmethod
    def _take(pool: list[LabeledExample], count: int, g: torch.Generator) -> list[LabeledExample]:
        """``count`` picks without replacement, cycling fresh permutations if needed."""
        out: list[LabeledExample] = []
        while len(out) < count and pool:
            perm = torch.randperm(len(pool), generator=g).tolist()
            out.extend(pool[i] for i in perm[: count - len(out)])
        return out

    def sample(self, n: int, generator: torch.Generator | None = None, *,
               anchor_fraction: float | None = None) -> list[LabeledExample]:
        """Stratified sample of ``n`` examples with anchors mixed in.

        Strata get equal shares (remainder to random strata). Deterministic for
        a given buffer state and generator seed.
        """
        g = generator if generator is not None else torch.Generator().manual_seed(0)
        frac = self.anchor_fraction if anchor_fraction is None else anchor_fraction
        keys = sorted(self._strata, key=_key_str)
        keys = [k for k in keys if self._strata[k]]
        n_anchor = int(round(n * frac)) if self._anchors else 0
        if not keys:
            n_anchor = n if self._anchors else 0
        n_main = n - n_anchor
        out: list[LabeledExample] = []
        if keys and n_main > 0:
            share, rem = divmod(n_main, len(keys))
            extra = set(torch.randperm(len(keys), generator=g)[:rem].tolist())
            for i, k in enumerate(keys):
                out.extend(self._take(self._strata[k], share + (i in extra), g))
        out.extend(self._take(self._anchors, n_anchor, g))
        order = torch.randperm(len(out), generator=g).tolist()
        return [out[i] for i in order]

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, Any]:
        """Tensors + scalars only (safe for ``torch.save`` / ``weights_only`` loads)."""
        return {
            "capacity": self.capacity, "per_stratum_capacity": self.per_stratum_capacity,
            "anchor_capacity": self.anchor_capacity, "anchor_fraction": self.anchor_fraction,
            "strata": {_key_str(k): [e.to_state() for e in v] for k, v in self._strata.items()},
            "seen": {_key_str(k): v for k, v in self._seen.items()},
            "anchors": [e.to_state() for e in self._anchors],
            "anchor_seen": self._anchor_seen, "rejected": self.rejected,
            "rng_state": repr(self._rng.getstate()),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        import ast

        self.capacity = int(state["capacity"])
        self.per_stratum_capacity = int(state["per_stratum_capacity"])
        self.anchor_capacity = int(state["anchor_capacity"])
        self.anchor_fraction = float(state["anchor_fraction"])
        self._strata = {_key_tuple(k): [LabeledExample.from_state(e) for e in v]
                        for k, v in state["strata"].items()}
        self._seen = {_key_tuple(k): int(v) for k, v in state["seen"].items()}
        self._anchors = [LabeledExample.from_state(e) for e in state["anchors"]]
        self._anchor_seen = int(state["anchor_seen"])
        self.rejected = int(state.get("rejected", 0))
        self._rng.setstate(ast.literal_eval(state["rng_state"]))

    def clear(self) -> None:
        """Forget everything except anchors (e.g. when training consent is revoked)."""
        self._strata.clear()
        self._seen.clear()
