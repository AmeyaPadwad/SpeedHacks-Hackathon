import torch

from soma.learning.datasets import (SomaDataset, SomaItem, collate_soma, item_from_batch,
                                    make_loader, seeded_split)
from soma.learning.synthetic_data import (DEFAULT_MIX, GENERIC_APPS, generate_personalized,
                                          generate_synthetic, label_distribution, personal_app)
from soma.models.batch import random_batch
from soma.models.tokenizer import hash_bucket


def _item(L, K, A, T=2, D=8, target=0):
    return SomaItem(
        images=torch.zeros(3, 8, 8), ocr_ids=torch.arange(1, L + 1), ocr_boxes=torch.rand(L, 4),
        ocr_padding_mask=torch.zeros(L, dtype=torch.bool), history=torch.randn(T, D),
        history_padding_mask=torch.zeros(T, dtype=torch.bool), app_ids=torch.tensor(3),
        memory_embeddings=torch.randn(K, D), memory_padding_mask=torch.zeros(K, dtype=torch.bool),
        candidate_embeddings=torch.randn(A, D), candidate_types=torch.ones(A, dtype=torch.long),
        candidate_mask=torch.zeros(A, dtype=torch.bool), target=target)


def test_collate_pads_to_batch_max_and_masks_padding():
    batch, targets, weights = collate_soma([_item(3, 1, 2, target=2), _item(5, 0, 4, T=3, target=4)])
    assert batch.ocr_ids.shape == (2, 5) and batch.ocr_boxes.shape == (2, 5, 4)
    assert batch.ocr_padding_mask[0].tolist() == [False] * 3 + [True] * 2
    assert batch.ocr_ids[0, 3:].tolist() == [0, 0]
    assert batch.memory_embeddings.shape == (2, 1, 8) and batch.memory_padding_mask[1].all()
    assert batch.candidate_embeddings.shape == (2, 4, 8)
    assert batch.candidate_mask[0].tolist() == [False, False, True, True]
    assert batch.history_padding_mask[0].tolist() == [False, False, True]
    assert batch.app_ids.shape == (2,)
    assert targets.tolist() == [2, 4] and targets.dtype == torch.long
    assert weights.tolist() == [1.0, 1.0]


def test_collate_rejects_out_of_range_target():
    try:
        collate_soma([_item(2, 1, 2, target=3)])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_item_from_batch_roundtrip():
    b = random_batch(3, D=8)
    item = item_from_batch(b, 1, target=2, app="code.exe")
    again, t, _ = collate_soma([item])
    assert torch.equal(again.ocr_ids[0], b.ocr_ids[1]) and t.item() == 2
    # Snapshot is independent of the source batch.
    b.ocr_ids[1, 0] = 999
    assert item.ocr_ids[0] != 999
    state = item.to_state()
    assert torch.equal(SomaItem.from_state(state).candidate_embeddings, item.candidate_embeddings)


def test_seeded_split_and_loader_are_deterministic():
    ds = SomaDataset([_item(2, 1, 2, target=i % 3) for i in range(20)])
    tr1, va1 = seeded_split(ds, 0.25, seed=3)
    tr2, va2 = seeded_split(ds, 0.25, seed=3)
    assert list(va1.indices) == list(va2.indices) and len(va1) == 5 and len(tr1) == 15
    assert set(tr1.indices).isdisjoint(va1.indices)
    order = lambda: [t.tolist() for _, t, _ in make_loader(ds, 4, shuffle=True, seed=7)]
    assert order() == order()


def test_synthetic_is_deterministic_and_well_formed(small_cfg):
    a = generate_synthetic(60, small_cfg, seed=5)
    b = generate_synthetic(60, small_cfg, seed=5)
    assert all(torch.equal(x.ocr_ids, y.ocr_ids) and torch.equal(x.candidate_embeddings, y.candidate_embeddings)
               and x.target == y.target for x, y in zip(a, b))
    batch, targets, _ = collate_soma(a)
    assert batch.candidate_embeddings.shape[-1] == small_cfg.d_model
    assert batch.ocr_ids.max() < small_cfg.vocab_size
    # A label always points to an unblocked candidate.
    for i, t in enumerate(targets.tolist()):
        if t > 0:
            assert not batch.candidate_mask[i, t - 1]
    dist = label_distribution(generate_synthetic(400, small_cfg, seed=1))
    assert 0.3 < dist["silence"] < 0.6
    assert {x.task for x in generate_synthetic(400, small_cfg, seed=1)} == set(DEFAULT_MIX)


def test_memory_matters_for_prior_work(small_cfg):
    items = generate_synthetic(300, small_cfg, seed=2)
    with_mem = [i for i in items if i.task == "prior_work_with_memory"]
    no_mem = [i for i in items if i.task == "prior_work_no_memory"]
    assert with_mem and no_mem
    assert all(i.target > 0 and not i.memory_padding_mask.all() for i in with_mem)
    assert all(i.target == 0 and i.memory_padding_mask.all() for i in no_mem)


def test_personalized_uses_non_colliding_app(small_cfg):
    app = personal_app(small_cfg.n_apps)
    assert hash_bucket(app, small_cfg.n_apps) not in {hash_bucket(a, small_cfg.n_apps) for a in GENERIC_APPS}
    items = generate_personalized(80, small_cfg, seed=0)
    assert all(i.app == app for i in items)
    errors = [i for i in items if i.task == "error"]
    assert errors and all(i.target == 0 for i in errors)
