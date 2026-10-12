"""Synthetic, non-sensitive GUI scenarios with a known best action (Stage A/B smoke data).

.. warning::
   Synthetic performance is **not** real-world quality. These scenarios exist
   so the training/eval/rollout machinery can be tested end to end and so that
   ablations have a measurable signal. Never report numbers on this data as
   evidence that SOMA helps real users.

Each item is a :class:`~soma.learning.datasets.SomaItem` whose label is
genuinely learnable from the tensors:

* OCR tokens contain scenario keywords (e.g. ``error traceback``) plus filler
  words and the app's title token; boxes put keywords in scenario regions.
* Candidates carry fixed per-action-type prototype embeddings (plus noise) and
  their ``candidate_types``; slot order is shuffled so position is not a cue.
* ``prior_work`` depends on memory: with a relevant memory the answer is
  ``COMPARE_WITH_PRIOR_WORK``, without any memory it is silence. Masking all
  memories at eval time (``evaluate_no_memory``) therefore costs accuracy.
* Many items are silence: idle screens, ``prior_work`` without memory, and
  hard negatives (error keywords but the matching candidate is absent or
  blocked).

Everything is deterministic for a given ``seed`` and model config.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from soma.config import ModelConfig
from soma.learning.datasets import SomaItem
from soma.models.tokenizer import encode, hash_bucket
from soma.types import ACTION_TYPES, ActionType

GENERIC_APPS: tuple[str, ...] = ("code.exe", "chrome.exe", "winword.exe", "windowsterminal.exe")
_PERSONAL_APP_CHOICES: tuple[str, ...] = ("notes.exe", "obsidian.exe", "notepad.exe",
                                          "slack.exe", "excel.exe", "figma.exe", "zoom.exe")

FILLER_WORDS: tuple[str, ...] = (
    "file", "edit", "view", "window", "help", "menu", "home", "page", "settings", "tab",
    "search", "open", "save", "panel", "toolbar", "status", "zoom", "share", "tools", "layout",
)

TYPE_INDEX: dict[ActionType, int] = {a: i for i, a in enumerate(ACTION_TYPES)}
SPEAKING_TYPES: tuple[ActionType, ...] = tuple(a for a in ACTION_TYPES
                                               if a is not ActionType.NO_SUGGESTION)


@dataclass(frozen=True)
class Scenario:
    """Template for one synthetic situation."""

    name: str
    keywords: tuple[str, ...]
    action: ActionType                       # best action when conditions hold
    region: tuple[float, float]              # (y0, y1) band where keywords appear
    apps: tuple[str, ...] = GENERIC_APPS


SCENARIOS: dict[str, Scenario] = {
    "error": Scenario("error", ("error", "traceback", "exception", "failed"),
                      ActionType.EXPLAIN_ERROR, (0.7, 0.95), ("code.exe", "windowsterminal.exe")),
    "missing_file": Scenario("missing_file", ("import", "module", "missing", "path"),
                             ActionType.OPEN_RELEVANT_FILE, (0.4, 0.7), ("code.exe", "chrome.exe")),
    "prior_work": Scenario("prior_work", ("draft", "report", "revision", "outline"),
                           ActionType.COMPARE_WITH_PRIOR_WORK, (0.1, 0.4), ("winword.exe", "chrome.exe")),
    "next_step": Scenario("next_step", ("passed", "complete", "done", "success"),
                          ActionType.SUGGEST_NEXT_STEP, (0.7, 0.95), ("windowsterminal.exe", "code.exe")),
    "unclear": Scenario("unclear", ("which", "option", "choose", "unsure"),
                        ActionType.ASK_CLARIFYING_QUESTION, (0.1, 0.4), ("chrome.exe", "winword.exe")),
    "idle": Scenario("idle", (), ActionType.NO_SUGGESTION, (0.0, 1.0)),
}

# Kind -> (scenario, variant). Variants: "normal", "with_memory", "no_memory", "hard_negative".
DEFAULT_MIX: dict[str, float] = {
    "error": 0.13,
    "missing_file": 0.10,
    "prior_work_with_memory": 0.12,
    "prior_work_no_memory": 0.10,
    "next_step": 0.10,
    "unclear": 0.10,
    "idle": 0.20,
    "hard_negative": 0.15,
}

_PROTO_SEED = 7919


def _prototypes(n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(n, d, generator=g)
    return p / p.norm(dim=-1, keepdim=True) * 2.0


def action_prototypes(cfg: ModelConfig) -> torch.Tensor:
    """Fixed ``[n_action_types, D]`` candidate prototype embeddings."""
    return _prototypes(max(cfg.n_action_types, len(ACTION_TYPES)), cfg.d_model, _PROTO_SEED)


def memory_prototype(cfg: ModelConfig) -> torch.Tensor:
    """Fixed ``[D]`` embedding of the 'relevant prior work' memory."""
    return _prototypes(1, cfg.d_model, _PROTO_SEED + 1)[0]


def personal_app(n_apps: int) -> str:
    """An app name whose hash bucket does not collide with :data:`GENERIC_APPS`."""
    used = {hash_bucket(a, n_apps) for a in GENERIC_APPS}
    for name in _PERSONAL_APP_CHOICES:
        if hash_bucket(name, n_apps) not in used:
            return name
    return _PERSONAL_APP_CHOICES[0]


def _app_token(app: str) -> str:
    return app.lower().removesuffix(".exe")


class SyntheticScenarioGenerator:
    """Deterministic generator of synthetic :class:`SomaItem` scenarios.

    ``preferences`` remaps the best action per scenario name (used to model a
    user's personal preferences, e.g. ``{"error": NO_SUGGESTION}`` in a quiet
    app). A preferred speaking action is guaranteed to be among the candidates.
    """

    def __init__(self, cfg: ModelConfig, seed: int = 0, *, ocr_len: int | None = None,
                 noise: float = 0.3) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.g = torch.Generator().manual_seed(seed)
        self.L = min(cfg.max_ocr_tokens, ocr_len or 12)
        self.noise = noise
        self.protos = action_prototypes(cfg)
        self.mem_proto = memory_prototype(cfg)

    # ------------------------------------------------------------ pieces
    def _randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=self.g)

    def _ocr(self, scenario: Scenario, app: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg, rng = self.cfg, self.rng
        words: list[tuple[str, tuple[float, float]]] = [(_app_token(app), (0.0, 0.05))]
        if scenario.keywords:
            for w in rng.sample(scenario.keywords, k=min(len(scenario.keywords), rng.randint(2, 3))):
                words.append((w, scenario.region))
        n_tokens = rng.randint(max(len(words) + 1, self.L // 2), self.L)
        while len(words) < n_tokens:
            words.append((rng.choice(FILLER_WORDS), (0.05, 1.0)))
        head, rest = words[:1], words[1:]
        rng.shuffle(rest)
        words = head + rest
        ids, boxes = [], []
        for w, (y0, y1) in words:
            ids.append(encode(w, cfg.vocab_size)[0])
            x = rng.uniform(0.0, 0.8)
            y = rng.uniform(y0, max(y0, y1 - 0.04))
            boxes.append([x, y, x + rng.uniform(0.05, 0.2), y + 0.04])
        ids_t = torch.full((self.L,), 0, dtype=torch.long)
        boxes_t = torch.zeros(self.L, 4)
        mask = torch.ones(self.L, dtype=torch.bool)
        n = len(ids)
        ids_t[:n] = torch.tensor(ids, dtype=torch.long)
        boxes_t[:n] = torch.tensor(boxes).clamp(0.0, 1.0)
        mask[:n] = False
        return ids_t, boxes_t, mask

    def _image(self, scenario_name: str) -> torch.Tensor:
        s = self.cfg.image_size
        img = 0.1 * self._randn(3, s, s)
        # Weak visual cue: a coloured band where the scenario's text lives.
        sc = SCENARIOS[scenario_name]
        y0, y1 = int(sc.region[0] * s), max(int(sc.region[0] * s) + 1, int(sc.region[1] * s))
        channel = list(SCENARIOS).index(scenario_name) % 3
        img[channel, y0:y1, :] += 0.5
        return img

    def _memories(self, relevant: bool, allow_distractors: bool) -> tuple[torch.Tensor, torch.Tensor]:
        K, D = self.cfg.max_memories, self.cfg.d_model
        emb = torch.zeros(K, D)
        mask = torch.ones(K, dtype=torch.bool)
        n = 0
        if relevant:
            n = self.rng.randint(1, K)
        elif allow_distractors and self.rng.random() < 0.5:
            n = self.rng.randint(1, K)
        for k in range(n):
            emb[k] = self._randn(D)
            mask[k] = False
        if relevant:
            slot = self.rng.randrange(n)
            emb[slot] = self.mem_proto + self.noise * self._randn(D)
        return emb, mask

    def _candidates(self, best: ActionType, include_best: bool, block_best: bool,
                    also: ActionType | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return (embeddings, types, mask, label) with label in logits indexing.

        ``also`` forces a hard-negative type (e.g. the generic answer that a
        personal preference overrides) into the candidate set.
        """
        A, D = self.cfg.max_candidates, self.cfg.d_model
        n = self.rng.randint(min(3 if also else 2, A), A)
        types: list[ActionType] = []
        if include_best and best is not ActionType.NO_SUGGESTION:
            types.append(best)
        if also is not None and also is not best and also is not ActionType.NO_SUGGESTION:
            types.append(also)
        pool = [t for t in SPEAKING_TYPES if t not in types and t is not best]
        types += self.rng.sample(pool, k=min(len(pool), n - len(types)))
        self.rng.shuffle(types)
        emb = torch.zeros(A, D)
        type_ids = torch.zeros(A, dtype=torch.long)
        mask = torch.ones(A, dtype=torch.bool)
        label = 0
        for slot, t in enumerate(types):
            emb[slot] = self.protos[TYPE_INDEX[t]] + self.noise * self._randn(D)
            type_ids[slot] = TYPE_INDEX[t]
            mask[slot] = False
            if t is best and best is not ActionType.NO_SUGGESTION:
                if block_best:
                    mask[slot] = True
                else:
                    label = slot + 1
        return emb, type_ids, mask, label

    # ------------------------------------------------------------ items
    def make(self, kind: str, *, app: str | None = None,
             preferences: dict[str, ActionType] | None = None) -> SomaItem:
        """Build one item of ``kind`` (a key of :data:`DEFAULT_MIX`)."""
        cfg = self.cfg
        variant = "normal"
        if kind == "hard_negative":
            scenario = SCENARIOS[self.rng.choice(["error", "missing_file", "next_step", "unclear"])]
            variant = "hard_negative"
        elif kind.startswith("prior_work"):
            scenario = SCENARIOS["prior_work"]
            variant = "with_memory" if kind.endswith("with_memory") else "no_memory"
        else:
            scenario = SCENARIOS[kind]
        app = app or self.rng.choice(scenario.apps)
        best = scenario.action
        if preferences and scenario.name in preferences and variant != "hard_negative":
            best = preferences[scenario.name]
        if variant == "no_memory":
            best = ActionType.NO_SUGGESTION

        ids, boxes, ocr_mask = self._ocr(scenario, app)
        T, D = cfg.history_len, cfg.d_model
        history = 0.1 * self._randn(T, D)
        hist_mask = torch.zeros(T, dtype=torch.bool)
        hist_mask[: self.rng.randint(0, T - 1)] = True
        mem, mem_mask = self._memories(relevant=(variant == "with_memory"),
                                       allow_distractors=(scenario.name != "prior_work"))

        if variant == "hard_negative":
            block = self.rng.random() < 0.5
            emb, types, cmask, _ = self._candidates(scenario.action, include_best=block, block_best=block)
            label = 0
        elif variant == "no_memory":
            # The compare candidate is offered but there is nothing to compare with.
            emb, types, cmask, _ = self._candidates(ActionType.COMPARE_WITH_PRIOR_WORK, True, False)
            label = 0
        elif best is ActionType.NO_SUGGESTION:
            emb, types, cmask, _ = self._candidates(scenario.action, include_best=True, block_best=False)
            label = 0
        else:
            emb, types, cmask, label = self._candidates(best, include_best=True, block_best=False,
                                                        also=scenario.action)

        return SomaItem(
            images=self._image(scenario.name),
            ocr_ids=ids, ocr_boxes=boxes, ocr_padding_mask=ocr_mask,
            history=history, history_padding_mask=hist_mask,
            app_ids=torch.tensor(hash_bucket(app, cfg.n_apps), dtype=torch.long),
            memory_embeddings=mem, memory_padding_mask=mem_mask,
            candidate_embeddings=emb, candidate_types=types, candidate_mask=cmask,
            target=label, weight=1.0, app=app, task=kind,
            meta={"synthetic": True, "variant": variant},
        )

    def generate(self, n: int, mix: dict[str, float] | None = None, *, app: str | None = None,
                 preferences: dict[str, ActionType] | None = None) -> list[SomaItem]:
        """``n`` items with kinds drawn from ``mix`` (default :data:`DEFAULT_MIX`)."""
        mix = mix or DEFAULT_MIX
        kinds, weights = list(mix), [mix[k] for k in mix]
        return [self.make(self.rng.choices(kinds, weights)[0], app=app, preferences=preferences)
                for _ in range(n)]


def generate_synthetic(n: int, cfg: ModelConfig, seed: int = 0,
                       mix: dict[str, float] | None = None) -> list[SomaItem]:
    """Generic prior-competency scenarios across :data:`GENERIC_APPS`."""
    return SyntheticScenarioGenerator(cfg, seed).generate(n, mix)


# A plausible personal preference: in one quiet app the user never wants error
# explanations and prefers a clarifying question over a next-step suggestion.
DEFAULT_PERSONAL_PREFERENCES: dict[str, ActionType] = {
    "error": ActionType.NO_SUGGESTION,
    "next_step": ActionType.ASK_CLARIFYING_QUESTION,
}
PERSONAL_MIX: dict[str, float] = {"error": 0.35, "next_step": 0.35, "missing_file": 0.1,
                                  "unclear": 0.1, "idle": 0.1}


def generate_personalized(n: int, cfg: ModelConfig, seed: int = 0, *, app: str | None = None,
                          preferences: dict[str, ActionType] | None = None,
                          mix: dict[str, float] | None = None) -> list[SomaItem]:
    """Items from one user's 'personal' app whose labels follow ``preferences``.

    A model trained only on :func:`generate_synthetic` gets these wrong; the
    controlled rollout must learn them without forgetting the generic set.
    """
    app = app or personal_app(cfg.n_apps)
    gen = SyntheticScenarioGenerator(cfg, seed)
    return gen.generate(n, mix or PERSONAL_MIX, app=app,
                        preferences=preferences or DEFAULT_PERSONAL_PREFERENCES)


def label_distribution(items: list[SomaItem]) -> dict[str, float]:
    """Fraction of silence vs speaking labels (handy for chance baselines)."""
    n = max(1, len(items))
    silent = sum(1 for i in items if i.target == 0)
    return {"silence": silent / n, "speak": 1 - silent / n}
