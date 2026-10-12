"""M3 acceptance on scripted GUI scenarios (build plan section 11).

A tiny ``SomaModel`` is trained for a few hundred steps on batches produced by
the *real* runtime pipeline (synthetic screens -> synthetic OCR -> parser ->
retrieval -> candidates), labelled with each scenario's expected action.
Held-out seeds then check that the runtime:

* suggests a grounded next step where one is useful,
* stays silent on known low-value screens (even though retrieval offers a
  "compare with prior work" candidate there),
* does not repeat a dismissed suggestion,
* never executes anything,
* and reloads from a checkpoint with identical scores.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from soma.config import SomaConfig, TrainConfig, small_model_config
from soma.consent import ConsentManager, Scope
from soma.learning.datasets import collate_soma
from soma.learning.trainer import Trainer
from soma.perception.ocr import SyntheticOCREngine
from soma.perception.synthetic import SCENARIOS, expected_action, render_scenario
from soma.policy.grounding import verify_target
from soma.runtime.factory import build_runtime, open_memory, save_runtime_checkpoint
from soma.runtime.pipeline import SomaRuntime
from soma.types import ActionType, FeedbackOutcome, ScreenFrame

SIZE = (640, 400)
SPEAK = {"editor_error": ActionType.EXPLAIN_ERROR, "terminal_traceback": ActionType.EXPLAIN_ERROR,
         "browser_docs": ActionType.COMPARE_WITH_PRIOR_WORK}
SILENT = ("editor_code", "idle_desktop", "settings_dialog")
TRAIN_SEEDS = range(0, 6)
EVAL_SEEDS = (100, 101)


class Clock:
    def __init__(self) -> None:
        self.t = 10_000.0

    def __call__(self) -> float:
        return self.t


def frame(name: str, seed: int, version: int) -> ScreenFrame:
    image, obs, _ = render_scenario(name, seed=seed, size=SIZE)
    return ScreenFrame(image=image, window=obs.window, version=version)


def target_index(rt_candidates, name: str) -> int:
    want = expected_action(name)
    if want is ActionType.NO_SUGGESTION:
        return 0
    for i, c in enumerate(rt_candidates):
        if c.action is want:
            return i + 1
    return 0


def make_env(tmp: Path):
    cfg = SomaConfig(device="cpu", model=small_model_config())
    consent = ConsentManager()
    for s in (Scope.SCREEN_CAPTURE, Scope.FOLDER_INDEXING, Scope.FEEDBACK_STORAGE):
        consent.grant(s)
    notes = tmp / "notes"
    notes.mkdir()
    (notes / "http_notes.md").write_text(
        "My notes on HTTP client sessions, retries and timeouts from the last project.",
        encoding="utf-8")
    consent.add_folder(notes)
    mem = open_memory(cfg, consent, tmp / "data")
    mem.ingestor.ingest_folder(notes)
    mem.ingestor.add_preference("When a traceback appears, inspect the last frame first")
    return cfg, consent, mem


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    torch.manual_seed(0)
    tmp = tmp_path_factory.mktemp("e2e")
    cfg, consent, mem = make_env(tmp)
    clock = Clock()
    rt = SomaRuntime(cfg, consent, ocr=SyntheticOCREngine(), memory=mem, clock=clock, model_seed=0)

    # collect runtime-produced training items, pushing history as the live loop does
    items, version = [], 0
    rng = random.Random(0)
    for seed in TRAIN_SEEDS:
        order = list(SCENARIOS)
        rng.shuffle(order)
        for name in order:
            version += 1
            p = rt.prepare(frame(name, seed, version))
            items.append(p.to_item(target_index(p.candidates, name), task=name, keep_images=True))
            rt.history.push(p.screen_state[0])
    rt.history.clear()

    trainer = Trainer(rt.model, TrainConfig(lr=3e-3, weight_decay=0.0, use_amp=False, grad_clip=1.0),
                      device="cpu")
    g = random.Random(1)
    for _ in range(150):
        batch, targets, weights = collate_soma(g.sample(items, 16))
        trainer.train_step(batch, targets, weights)
    rt.model.eval()
    yield rt, clock, tmp
    mem.close()


def run(rt: SomaRuntime, clock: Clock, name: str, seed: int, version: int):
    clock.t += 700.0          # past cooldown and the duplicate window
    return rt.process_frame(frame(name, seed, version))


def test_suggests_grounded_next_step_where_useful(trained):
    rt, clock, _ = trained
    v = 1000
    for seed in EVAL_SEEDS:
        for name, action in SPEAK.items():
            v += 1
            r = run(rt, clock, name, seed, v)
            s = r.suggestion
            assert not s.is_silent, (name, seed, s.reason)
            assert s.candidate.action is action, (name, s.candidate.action)
            assert s.screen_version == v and s.confidence >= rt.cfg.policy.relevance_threshold
            assert verify_target(s.candidate, r.observation)
            if action is ActionType.EXPLAIN_ERROR:
                assert s.candidate.target is not None and s.candidate.evidence
            else:
                assert s.candidate.memory_ref in {m.item.id for m in r.retrieved}


def test_stays_silent_on_low_value_screens(trained):
    rt, clock, _ = trained
    v = 2000
    for seed in EVAL_SEEDS:
        for name in SILENT:
            v += 1
            r = run(rt, clock, name, seed, v)
            assert r.candidates, name                  # retrieval did offer something ...
            assert r.suggestion.is_silent, (name, r.suggestion.reason)   # ... and SOMA abstained
            assert r.suggestion.distribution["NO_SUGGESTION"] > 0.5


def test_dismissed_suggestion_not_repeated(trained):
    rt, clock, _ = trained
    first = run(rt, clock, "browser_docs", 200, 3001)
    assert not first.suggestion.is_silent
    key = first.suggestion.candidate.key
    rt.feedback(first.suggestion.id, FeedbackOutcome.REJECTED)
    for i in range(3):
        clock.t += 60.0       # past cooldown, inside the duplicate window
        r = rt.process_frame(frame("browser_docs", 201 + i, 3002 + i))
        assert key in {c.key for c in r.candidates}
        assert r.suggestion.is_silent or r.suggestion.candidate.key != key


def test_never_executes(trained):
    rt, clock, _ = trained
    for i, name in enumerate(SCENARIOS):
        r = run(rt, clock, name, 300, 4000 + i)
        if not r.suggestion.is_silent:
            rt.feedback(r.suggestion.id, FeedbackOutcome.ACCEPTED)
    assert rt.ledger.executors == {}
    assert rt.ledger.executed_count == 0 and rt.ledger.unapproved_execution_count == 0


def test_checkpoint_roundtrip_through_factory_scores_identically(trained, tmp_path):
    rt, _, tmp = trained
    ckpt = save_runtime_checkpoint(rt, tmp_path / "soma.pt")
    rt2 = build_runtime(rt.cfg, rt.consent, data_dir=tmp_path / "data2", checkpoint=ckpt, synthetic=True)
    own_memory = rt2.memory
    rt2.memory = rt.memory                    # same retrieval context for both runtimes
    saved_ranker, rt.ranker, rt2.ranker = rt.ranker, None, None
    try:
        for name in ("terminal_traceback", "idle_desktop", "browser_docs"):
            f = frame(name, 500, 1)
            rt.history.clear()
            rt2.history.clear()
            p1, p2 = rt.prepare(f), rt2.prepare(f)
            assert [c.key for c in p1.candidates] == [c.key for c in p2.candidates]
            torch.testing.assert_close(rt.score(p1), rt2.score(p2))
    finally:
        rt.ranker = saved_ranker
        own_memory.close()
    assert list((tmp_path / "data2").glob("*.png")) == []
