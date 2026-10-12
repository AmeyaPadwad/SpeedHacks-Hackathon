"""Safety / interruptibility gate: NO_SUGGESTION must often win."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from soma.config import PolicyConfig
from soma.policy.safety_gate import SafetyGate, distribution_labels, to_probs
from soma.types import (
    ActionType,
    Candidate,
    MemoryItem,
    OCRElement,
    RetrievedMemory,
    ScreenObservation,
    SourceType,
    WindowInfo,
)

ERR = "ZeroDivisionError: division by zero"
BOX = (0.1, 0.5, 0.6, 0.54)


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def obs(*elements: OCRElement, app: str = "Code.exe", version: int = 7) -> ScreenObservation:
    return ScreenObservation(window=WindowInfo(app, "t"), elements=list(elements) or [OCRElement(ERR, BOX)],
                             version=version)


def explain() -> Candidate:
    return Candidate(ActionType.EXPLAIN_ERROR, f'Explain the error "{ERR}"', target=OCRElement(ERR, BOX))


def ask() -> Candidate:
    return Candidate(ActionType.ASK_CLARIFYING_QUESTION, "Which error first?")


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def gate(clock):
    cfg = PolicyConfig(relevance_threshold=0.55, margin_over_abstain=0.05, cooldown_s=30.0,
                       duplicate_window_s=600.0, quiet_apps=["Spotify.exe", "zoom"])
    return SafetyGate(cfg, clock=clock)


CONFIDENT = [0.1, 0.8, 0.1]       # NO_SUGGESTION, explain, ask


def test_speaks_when_confident(gate):
    s = gate.decide(CONFIDENT, [explain(), ask()], obs())
    assert not s.is_silent and s.candidate.action is ActionType.EXPLAIN_ERROR
    assert s.confidence == pytest.approx(0.8)
    assert s.screen_version == 7
    assert set(s.distribution) == {"NO_SUGGESTION", "EXPLAIN_ERROR", "ASK_CLARIFYING_QUESTION"}
    assert sum(s.distribution.values()) == pytest.approx(1.0)
    assert "EXPLAIN_ERROR" in s.reason and "threshold" in s.reason


def test_accepts_logits_tensor_with_batch_dim_and_padding(gate):
    logits = torch.tensor([[0.0, 4.0, -1.0, float("-inf")]])  # last slot is padding
    s = gate.decide(logits, [explain(), ask()], obs())
    assert not s.is_silent
    assert sum(s.distribution.values()) == pytest.approx(1.0)


def test_silent_below_threshold(gate):
    s = gate.decide([0.3, 0.4, 0.3], [explain(), ask()], obs())
    assert s.is_silent and "low relevance" in s.reason
    assert s.confidence == pytest.approx(0.3)
    assert s.distribution["NO_SUGGESTION"] == pytest.approx(0.3)


def test_silent_when_not_beating_abstain_by_margin(clock):
    g = SafetyGate(PolicyConfig(relevance_threshold=0.4, margin_over_abstain=0.05), clock=clock)
    s = g.decide([0.48, 0.52], [explain()], obs())
    assert s.is_silent and "does not beat NO_SUGGESTION" in s.reason


def test_cooldown(gate, clock):
    assert not gate.decide(CONFIDENT, [explain(), ask()], obs()).is_silent
    clock.advance(10)
    s = gate.decide([0.1, 0.1, 0.8], [explain(), ask()], obs())
    assert s.is_silent and "cooldown" in s.reason
    clock.advance(25)
    s = gate.decide([0.1, 0.1, 0.8], [explain(), ask()], obs())
    assert not s.is_silent and s.candidate.action is ActionType.ASK_CLARIFYING_QUESTION


def test_duplicate_suppression_after_dismissal(gate, clock):
    s = gate.decide(CONFIDENT, [explain()], obs())
    assert not s.is_silent
    gate.record_dismissed(s.candidate.key)
    clock.advance(60)  # past cooldown, inside duplicate window
    s2 = gate.decide(CONFIDENT[:2], [explain()], obs())
    assert s2.is_silent and "dismissed" in s2.reason
    clock.advance(600)  # duplicate window over
    assert not gate.decide(CONFIDENT[:2], [explain()], obs()).is_silent


def test_shown_suggestion_not_repeated_but_next_best_can_speak(gate, clock):
    assert gate.decide([0.05, 0.9, 0.05], [explain(), ask()], obs()).candidate.action is ActionType.EXPLAIN_ERROR
    clock.advance(60)
    s = gate.decide([0.1, 0.6, 0.3], [explain(), ask()], obs())
    assert s.is_silent  # ask is below threshold, explain was already shown
    assert "already shown" in s.reason
    s = gate.decide([0.05, 0.3, 0.65], [explain(), ask()], obs())
    assert s.candidate.action is ActionType.ASK_CLARIFYING_QUESTION


@pytest.mark.parametrize("app", ["Spotify.exe", "spotify", "Zoom.exe"])
def test_quiet_app(gate, app):
    s = gate.decide(CONFIDENT, [explain(), ask()], obs(app=app))
    assert s.is_silent and "quiet" in s.reason
    s = gate.decide(CONFIDENT, [explain(), ask()], obs(), app_name=app)
    assert s.is_silent


def test_grounding_failure(gate):
    moved_away = obs(OCRElement("All tests passed", BOX))
    s = gate.decide([0.1, 0.9], [explain()], moved_away)
    assert s.is_silent and "target no longer on screen" in s.reason
    assert not gate.in_cooldown()  # silence does not start a cooldown


def test_grounding_failure_falls_back_to_untargeted(gate):
    moved_away = obs(OCRElement("All tests passed", BOX))
    s = gate.decide([0.05, 0.6, 0.35], [explain(), ask()], moved_away)
    assert s.is_silent  # ask is below threshold
    s = gate.decide([0.0, 0.6, 0.4], [explain(), ask()], moved_away, is_logits=False)
    assert s.is_silent
    g2 = SafetyGate(PolicyConfig(relevance_threshold=0.3), clock=FakeClock())
    s = g2.decide([0.05, 0.55, 0.4], [explain(), ask()], moved_away)
    assert s.candidate.action is ActionType.ASK_CLARIFYING_QUESTION
    assert "skipped" in s.reason


def test_no_candidates_is_silent(gate):
    s = gate.decide([5.0], [], obs())
    assert s.is_silent and s.distribution == {"NO_SUGGESTION": 1.0}


def test_record_false_does_not_start_cooldown(gate):
    assert not gate.decide(CONFIDENT, [explain(), ask()], obs(), record=False).is_silent
    assert not gate.in_cooldown()
    assert not gate.is_duplicate(explain().key)


def test_blocked_candidate_is_skipped(gate):
    open_file = Candidate(ActionType.OPEN_RELEVANT_FILE, "Open secret.py", file_ref="C:/private/secret.py")
    blocked = SafetyGate.blocked_mask([open_file, ask()], authorized_file=lambda p: p.startswith("src/"))
    assert blocked == [True, False]
    s = gate.decide([0.05, 0.9, 0.05], [open_file, ask()], obs(), blocked=blocked)
    assert s.is_silent and "blocked" in s.reason


def test_blocked_mask_rules():
    item = MemoryItem(id="m1", source_type=SourceType.DOCUMENT, user_approved=True, source_uri="C:/docs/a.md")
    unapproved = MemoryItem(id="m2", source_type=SourceType.DOCUMENT, user_approved=False)
    mems = {"m1": RetrievedMemory(item, np.zeros(4, np.float32), 0.9),
            "m2": RetrievedMemory(unapproved, np.zeros(4, np.float32), 0.9)}
    cmp1 = Candidate(ActionType.COMPARE_WITH_PRIOR_WORK, "c1", memory_ref="m1")
    cmp2 = Candidate(ActionType.COMPARE_WITH_PRIOR_WORK, "c2", memory_ref="m2")
    cmp3 = Candidate(ActionType.COMPARE_WITH_PRIOR_WORK, "c3", memory_ref="missing")
    f = Candidate(ActionType.OPEN_RELEVANT_FILE, "o", file_ref="src/a.py")
    assert SafetyGate.blocked_mask([cmp1, cmp2, cmp3, f], memories_by_id=mems) == [False, True, True, True]
    assert SafetyGate.blocked_mask([cmp1, f], authorized_file=lambda p: True,
                                   blocked_sources={"C:/docs/a.md"}, memories_by_id=mems) == [True, False]
    assert SafetyGate.blocked_mask([cmp1], blocked_sources={"m1"}) == [True]


def test_to_probs_and_labels():
    p = to_probs([0.2, 0.8])
    assert p.tolist() == pytest.approx([0.2, 0.8])
    p = to_probs([0.0, 2.0])  # logits (sum != 1)
    assert p.sum() == pytest.approx(1.0) and p[1] > p[0]
    assert to_probs([float("-inf"), float("-inf")]).tolist() == [1.0, 0.0]
    assert distribution_labels([explain(), explain(), ask()]) == [
        "NO_SUGGESTION", "EXPLAIN_ERROR", "EXPLAIN_ERROR#2", "ASK_CLARIFYING_QUESTION"]
    with pytest.raises(ValueError):
        SafetyGate(PolicyConfig()).decide([1.0], [explain()], obs())


def test_abstain_wins_on_typical_uncertain_scores(gate):
    """With a uniform-ish scorer (untrained), the gate stays silent."""
    rng = np.random.default_rng(0)
    spoke = 0
    for _ in range(50):
        logits = rng.normal(0, 0.5, size=4)
        spoke += not gate.decide(logits, [explain(), ask(), ask()], obs(), record=False).is_silent
    assert spoke == 0


# ---------------------------------------------------------------- end to end (M3 scripted scenario)


def test_scripted_scenario_generator_encoder_gate_ledger(small_cfg):
    from soma.policy import (
        ApprovalLedger,
        CandidateEncoder,
        NoExecutorError,
        SafetyGate,
        generate_candidates,
    )

    lines = ["Traceback (most recent call last):",
             '  File "src/app/calc.py", line 40, in divide',
             ERR]
    screen = ScreenObservation(
        window=WindowInfo("Code.exe", "calc.py"),
        elements=[OCRElement(t, (0.05, 0.1 + 0.05 * i, 0.7, 0.14 + 0.05 * i)) for i, t in enumerate(lines)],
        version=11)
    pref = RetrievedMemory(MemoryItem("pref", SourceType.PREFERENCE, True,
                                      "prefers inspecting tracebacks before changing code"),
                           np.ones(8, np.float32), 0.9)
    authorized = lambda p: p.startswith("src/")  # noqa: E731
    cands = generate_candidates(screen, [pref], small_cfg.max_candidates, authorized)
    blocked = SafetyGate.blocked_mask(cands, authorized_file=authorized, memories_by_id={"pref": pref})
    assert not any(blocked)

    enc = CandidateEncoder.from_config(small_cfg, memory_dim=8)
    emb, types, mask = enc.encode(cands, {"pref": pref}, A=small_cfg.max_candidates, blocked=blocked)
    assert emb.shape == (small_cfg.max_candidates, small_cfg.d_model)

    # stand-in scorer: favour the preference-shaped next step, mask pads with -inf
    logits = torch.full((1, small_cfg.max_candidates + 1), float("-inf"))
    logits[0, 0] = 0.0
    for i, c in enumerate(cands):
        logits[0, i + 1] = 3.0 if c.action is ActionType.SUGGEST_NEXT_STEP else -1.0
    logits[0, 1:][mask] = float("-inf")

    clock = FakeClock()
    gate = SafetyGate(PolicyConfig(), clock=clock)
    s = gate.decide(logits, cands, screen)
    assert s.candidate is not None
    assert s.candidate.text == "Inspect the traceback at src/app/calc.py:40 before editing"
    assert s.screen_version == 11

    # dismissed: not repeated even after the cooldown
    gate.record_dismissed(s.candidate.key)
    clock.advance(PolicyConfig().cooldown_s + 1)
    s2 = gate.decide(logits, cands, screen)
    assert s2.candidate is None or s2.candidate.key != s.candidate.key

    # idle screen: nothing to say
    idle = ScreenObservation(window=WindowInfo("Code.exe", "x"), elements=[OCRElement("def f():", BOX)], version=12)
    assert gate.decide([1.0], generate_candidates(idle, [pref], 4, authorized), idle).is_silent

    # accepting the suggestion never executes anything in v0.1
    ledger = ApprovalLedger()
    req = ledger.propose(s)
    tok = ledger.approve(req.id, user_confirmed=True)
    with pytest.raises(NoExecutorError):
        ledger.request_execution(req.id, token=tok.token, current_version=11)
    assert ledger.unapproved_execution_count == 0
