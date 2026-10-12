"""SOMA learning package: stages A-D of build plan section 8 (milestone M4).

* ``datasets``       SomaItem, padding collate, seeded split / loaders
* ``synthetic_data`` deterministic non-sensitive scenarios (not real-world quality)
* ``losses``         policy CE (with NO_SUGGESTION), InfoNCE retrieval, pairwise ranking
* ``feedback``       consent-gated feedback -> labeled examples (ignored != negative)
* ``replay_buffer``  stratified reservoir replay with prior-competency anchors
* ``online_ranker``  Stage C bounded logit biases, never touches base weights
* ``trainer``        AdamW / clipping / autocast / early stopping / checkpoints
* ``evaluator``      top-k, abstention, unwanted suggestions, ECE, no-memory ablation
* ``rollout``        Stage D gated deploy-or-rollback
* ``metrics``        retrieval, OCR, grounding, personalization metrics
"""

from soma.learning.datasets import (SomaDataset, SomaItem, collate_soma, item_from_batch,
                                    make_loader, seeded_split)
from soma.learning.evaluator import compare, evaluate, evaluate_no_memory
from soma.learning.feedback import (FeedbackRecorder, FeedbackSink, InMemoryFeedbackSink,
                                    LabeledExample, as_examples)
from soma.learning.losses import (pairwise_ranking_loss, policy_loss, retrieval_contrastive_loss,
                                  total_loss)
from soma.learning.online_ranker import OnlinePreferenceRanker
from soma.learning.replay_buffer import ReplayBuffer
from soma.learning.rollout import ControlledRollout, RolloutDecision
from soma.learning.synthetic_data import generate_personalized, generate_synthetic
from soma.learning.trainer import Trainer, TrainResult, set_trainable

__all__ = [
    "SomaDataset", "SomaItem", "collate_soma", "item_from_batch", "make_loader", "seeded_split",
    "compare", "evaluate", "evaluate_no_memory",
    "FeedbackRecorder", "FeedbackSink", "InMemoryFeedbackSink", "LabeledExample", "as_examples",
    "pairwise_ranking_loss", "policy_loss", "retrieval_contrastive_loss", "total_loss",
    "OnlinePreferenceRanker", "ReplayBuffer", "ControlledRollout", "RolloutDecision",
    "generate_personalized", "generate_synthetic", "Trainer", "TrainResult", "set_trainable",
]
