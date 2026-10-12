from pathlib import Path

import pytest
import torch

from soma.config import SomaConfig, load_config
from soma.consent import ConsentError, ConsentManager, Scope
from soma.device import select_device, seed_everything
from soma.models.batch import random_batch
from soma.models.tokenizer import encode, hash_bucket


def test_device_falls_back_to_cpu():
    assert select_device("cpu").type == "cpu"
    assert select_device("auto").type in ("cpu", "cuda")


def test_seed_is_reproducible():
    seed_everything(3); a = torch.rand(3)
    seed_everything(3); b = torch.rand(3)
    assert torch.equal(a, b)


def test_load_configs():
    assert isinstance(load_config(None), SomaConfig)
    dev = load_config("configs/dev.yaml")
    assert dev.model.d_model == 256 and dev.capture.persist_screenshots is False
    train = load_config("configs/train.yaml")
    assert train.train.epochs == 8


def test_consent_defaults_off_and_pause(tmp_path: Path):
    c = ConsentManager(tmp_path / "consent.json")
    for s in Scope:
        assert not c.allowed(s)
    with pytest.raises(ConsentError):
        c.require(Scope.SCREEN_CAPTURE)
    c.grant(Scope.SCREEN_CAPTURE)
    assert c.allowed(Scope.SCREEN_CAPTURE)
    c.pause()
    assert not c.allowed(Scope.SCREEN_CAPTURE)
    assert ConsentManager(tmp_path / "consent.json").paused  # persisted


def test_consent_refuses_home_and_scopes_paths(tmp_path: Path):
    c = ConsentManager()
    with pytest.raises(ConsentError):
        c.add_folder(Path.home())
    c.grant(Scope.FOLDER_INDEXING)
    c.add_folder(tmp_path / "notes")
    assert c.path_allowed(tmp_path / "notes" / "a.md")
    assert not c.path_allowed(tmp_path / "other" / "a.md")
    c.revoke(Scope.FOLDER_INDEXING)
    assert not c.path_allowed(tmp_path / "notes" / "a.md")


def test_tokenizer_is_stable():
    assert encode("Run tests", 512) == encode("run TESTS", 512)
    assert all(i >= 2 for i in encode("Traceback (most recent call last)", 512))
    assert hash_bucket("Code.exe", 16) == hash_bucket("code.exe", 16)


def test_random_batch_shapes():
    b = random_batch(B=3, A=5, K=2)
    assert b.candidate_embeddings.shape == (3, 5, 64)
    assert b.batch_size == 3
