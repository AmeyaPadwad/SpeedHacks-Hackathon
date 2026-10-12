import torch

from soma.consent import ConsentManager, Scope
from soma.learning.feedback import (FeedbackRecorder, FeedbackSink, InMemoryFeedbackSink,
                                    LabeledExample, as_examples, example_from_feedback)
from soma.learning.replay_buffer import ReplayBuffer
from soma.learning.synthetic_data import generate_synthetic
from soma.types import FeedbackEvent, FeedbackOutcome


def _consent(*scopes):
    c = ConsentManager()
    for s in scopes:
        c.grant(s)
    return c


def _item(small_cfg, seed=0, app="code.exe", task="error"):
    return generate_synthetic(1, small_cfg, seed=seed)[0].clone(app=app, task=task)


def test_nothing_recorded_without_feedback_storage(small_cfg):
    sink = InMemoryFeedbackSink()
    rec = FeedbackRecorder(_consent(Scope.MODEL_TRAINING), sink)
    out = rec.record(FeedbackEvent("s1", FeedbackOutcome.ACCEPTED, consented_for_training=True),
                     item=_item(small_cfg), shown_index=1)
    assert out is None and sink.events == [] and rec.examples == []
    assert isinstance(sink, FeedbackSink)


def test_training_flag_follows_model_training_consent(small_cfg):
    sink = InMemoryFeedbackSink()
    rec = FeedbackRecorder(_consent(Scope.FEEDBACK_STORAGE), sink)
    ev = FeedbackEvent("s1", FeedbackOutcome.ACCEPTED, consented_for_training=True)
    ex = rec.record(ev, item=_item(small_cfg), shown_index=1)
    assert sink.events == [ev] and ev.consented_for_training is False
    assert ex is not None and not ex.usable_for_training and rec.training_examples() == []

    rec.consent.grant(Scope.MODEL_TRAINING)
    ev2 = FeedbackEvent("s2", FeedbackOutcome.ACCEPTED)
    ex2 = rec.record(ev2, item=_item(small_cfg), shown_index=2)
    assert ev2.consented_for_training and ex2.usable_for_training
    assert rec.training_examples() == [ex2]
    rec.consent.revoke(Scope.MODEL_TRAINING)
    assert rec.training_examples() == []


def test_ignored_is_recorded_but_never_a_negative(small_cfg):
    sink = InMemoryFeedbackSink()
    rec = FeedbackRecorder(_consent(Scope.FEEDBACK_STORAGE, Scope.MODEL_TRAINING), sink)
    ex = rec.record(FeedbackEvent("s", FeedbackOutcome.IGNORED), item=_item(small_cfg), shown_index=2)
    assert len(sink.events) == 1
    assert ex.ambiguous and ex.rejected_index is None and ex.item.weight == 0.0
    assert not ex.usable_for_training and rec.training_examples() == []
    buf = ReplayBuffer(10)
    assert buf.add(ex) is False and len(buf) == 0


def test_outcomes_convert_to_labels(small_cfg):
    item = _item(small_cfg)
    acc = example_from_feedback(item, FeedbackOutcome.ACCEPTED, 2, consented=True)
    ed = example_from_feedback(item, FeedbackOutcome.EDITED, 2, consented=True)
    rej = example_from_feedback(item, FeedbackOutcome.REJECTED, 2, consented=True)
    assert (acc.item.target, acc.item.weight) == (2, 1.0)
    assert (ed.item.target, ed.item.weight) == (2, 0.5)
    assert (rej.item.target, rej.item.weight, rej.rejected_index) == (0, 0.5, 2)
    # Pixels dropped by default, kept on request; ids preserved.
    assert acc.item.images.abs().sum() == 0 and torch.equal(acc.item.ocr_ids, item.ocr_ids)
    kept = example_from_feedback(item, FeedbackOutcome.ACCEPTED, 2, consented=True, keep_images=True)
    assert torch.equal(kept.item.images, item.images)
    restored = LabeledExample.from_state(rej.to_state())
    assert restored.rejected_index == 2 and restored.outcome is FeedbackOutcome.REJECTED


def _examples(small_cfg, n, app, task, seed=0, consented=True):
    items = [i.clone(app=app, task=task) for i in generate_synthetic(n, small_cfg, seed=seed)]
    exs = as_examples(items, source="feedback")
    for e in exs:
        e.consented = consented
    return exs


def test_replay_bounds_and_stratification(small_cfg):
    buf = ReplayBuffer(capacity=30, per_stratum_capacity=20, seed=0)
    buf.extend(_examples(small_cfg, 100, "code.exe", "error"))
    buf.extend(_examples(small_cfg, 5, "winword.exe", "draft", seed=1))
    assert len(buf) <= 30
    assert buf.strata()[("winword.exe", "draft")] == 5        # minority not crowded out
    assert buf.strata()[("code.exe", "error")] <= 20
    assert buf.add(_examples(small_cfg, 1, "x", "y", consented=False)[0]) is False
    sample = buf.sample(20, torch.Generator().manual_seed(0), anchor_fraction=0.0)
    n_word = sum(1 for e in sample if e.app == "winword.exe")
    assert len(sample) == 20 and n_word == 10                  # equal share per stratum
    assert len({id(e) for e in sample if e.app == "winword.exe"}) == 5


def test_replay_anchor_mixin_and_determinism(small_cfg):
    def build():
        buf = ReplayBuffer(capacity=50, anchor_capacity=10, anchor_fraction=0.25, seed=4)
        buf.extend(_examples(small_cfg, 40, "code.exe", "error"))
        for a in as_examples(generate_synthetic(30, small_cfg, seed=9), source="anchor"):
            buf.add_anchor(a)
        return buf

    b1, b2 = build(), build()
    assert b1.n_anchors == 10
    s1 = b1.sample(16, torch.Generator().manual_seed(1))
    s2 = b2.sample(16, torch.Generator().manual_seed(1))
    assert [torch.equal(x.item.ocr_ids, y.item.ocr_ids) for x, y in zip(s1, s2)] == [True] * 16
    assert sum(e.source == "anchor" for e in s1) == 4
    only_anchor = ReplayBuffer(5)
    only_anchor.add_anchor(as_examples(generate_synthetic(1, small_cfg), "anchor")[0])
    assert len(only_anchor.sample(3)) == 3


def test_replay_state_dict_roundtrip(small_cfg, tmp_path):
    buf = ReplayBuffer(capacity=20, seed=2)
    buf.extend(_examples(small_cfg, 25, "code.exe", "error"))
    buf.add_anchor(as_examples(generate_synthetic(1, small_cfg), "anchor")[0])
    torch.save(buf.state_dict(), tmp_path / "replay.pt")
    other = ReplayBuffer(1)
    other.load_state_dict(torch.load(tmp_path / "replay.pt", weights_only=True))
    assert other.strata() == buf.strata() and other.n_anchors == 1
    a = buf.sample(8, torch.Generator().manual_seed(3))
    b = other.sample(8, torch.Generator().manual_seed(3))
    assert [torch.equal(x.item.candidate_embeddings, y.item.candidate_embeddings) for x, y in zip(a, b)] == [True] * 8
    # Reservoir RNG state also survives, so future adds evolve identically.
    extra = _examples(small_cfg, 10, "code.exe", "error", seed=7)
    buf.extend(extra)
    other.extend(extra)
    assert [e.feedback_id + str(e.created_ms) for e in buf.examples()] ==         [e.feedback_id + str(e.created_ms) for e in other.examples()]
