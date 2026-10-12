import torch

from soma.learning.online_ranker import OnlinePreferenceRanker
from soma.types import ActionType, Candidate, FeedbackOutcome


def _cands():
    return [Candidate(ActionType.EXPLAIN_ERROR, "Explain this error"),
            Candidate(ActionType.SUGGEST_NEXT_STEP, "Run the tests"), None]


def test_no_evidence_means_zero_bias():
    r = OnlinePreferenceRanker()
    b = r.bias(_cands(), "code.exe", n_slots=4)
    assert b.shape == (5,) and torch.count_nonzero(b) == 0


def test_rejections_raise_abstain_and_lower_candidate():
    r = OnlinePreferenceRanker()
    c = _cands()
    for _ in range(5):
        r.update_candidate(c[0], "code.exe", FeedbackOutcome.REJECTED)
    b = r.bias(c, "code.exe")
    assert b[0] > 0 and b[1] < 0 and b[2] == 0 and b[3] == 0
    # Other apps are unaffected.
    assert torch.count_nonzero(r.bias(c, "chrome.exe")) == 0


def test_accepts_raise_candidate_and_lower_abstain():
    r = OnlinePreferenceRanker()
    c = _cands()
    for _ in range(4):
        r.update_candidate(c[1], "code.exe", FeedbackOutcome.ACCEPTED)
    b = r.bias(c, "code.exe")
    assert b[2] > 0 and b[0] < 0
    edited = OnlinePreferenceRanker()
    for _ in range(4):
        edited.update_candidate(c[1], "code.exe", FeedbackOutcome.EDITED)
    assert 0 < edited.bias(c, "code.exe")[2] < b[2]


def test_ignored_is_not_counted():
    r = OnlinePreferenceRanker()
    assert r.update("code.exe", ActionType.EXPLAIN_ERROR, FeedbackOutcome.IGNORED) is False
    assert r.updates == 0 and torch.count_nonzero(r.bias(_cands(), "code.exe")) == 0


def test_bias_is_bounded_and_decays():
    r = OnlinePreferenceRanker(max_bias=1.5, decay=0.5)
    for _ in range(200):
        r.update("code.exe", ActionType.EXPLAIN_ERROR, FeedbackOutcome.REJECTED)
    b = r.bias(_cands(), "code.exe")
    assert b.abs().max() <= 1.5 + 1e-6
    before = r.abstain_bias("code.exe")
    for _ in range(10):  # evidence elsewhere decays this app's arms
        r.update("chrome.exe", ActionType.SUGGEST_NEXT_STEP, FeedbackOutcome.ACCEPTED)
    assert abs(r.abstain_bias("code.exe")) < before


def test_apply_keeps_blocked_and_does_not_touch_base_weights():
    logits = torch.tensor([[0.0, 1.0, -1e4, float("-inf")]])
    bias = torch.tensor([0.5, -0.5, 2.0, 2.0])
    out = OnlinePreferenceRanker.apply(logits, bias)
    assert out.tolist() == [[0.5, 0.5, -1e4, float("-inf")]]
    net = torch.nn.Linear(2, 2)
    snap = {k: v.clone() for k, v in net.state_dict().items()}
    r = OnlinePreferenceRanker()
    r.update("a", ActionType.EXPLAIN_ERROR, FeedbackOutcome.ACCEPTED)
    OnlinePreferenceRanker.apply(net(torch.zeros(1, 2)), torch.zeros(2))
    assert all(torch.equal(snap[k], v) for k, v in net.state_dict().items())


def test_state_dict_roundtrip_and_reset():
    r = OnlinePreferenceRanker()
    c = _cands()
    r.update_candidate(c[0], "code.exe", FeedbackOutcome.REJECTED)
    r.update_candidate(c[1], "chrome.exe", FeedbackOutcome.ACCEPTED)
    other = OnlinePreferenceRanker(prior=5.0)
    other.load_state_dict(r.state_dict())
    assert torch.equal(other.bias(c, "code.exe"), r.bias(c, "code.exe"))
    r.reset("code.exe")
    assert torch.count_nonzero(r.bias(c, "code.exe")) == 0
    assert torch.count_nonzero(r.bias(c, "chrome.exe")) > 0
    t = r.bias_for_types([ActionType.SUGGEST_NEXT_STEP, None], "chrome.exe")
    assert t.shape == (3,) and t[1] > 0 and t[2] == 0
