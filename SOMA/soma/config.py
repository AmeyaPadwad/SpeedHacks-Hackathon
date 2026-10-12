"""Typed configuration loaded from YAML (``configs/dev.yaml``, ``configs/train.yaml``)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

SOMA_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelConfig:
    d_model: int = 512                 # fusion dimension D
    n_heads: int = 8
    dropout: float = 0.1
    image_size: int = 448
    patch_size: int = 32               # 448/32 = 14 -> N = 196 visual tokens
    vision_layers: int = 2
    vision_backbone: str = "scratch"   # "scratch" | "torchvision_vit_b_16" (frozen, pretrained)
    vocab_size: int = 8192             # hashed OCR token vocabulary
    max_ocr_tokens: int = 256          # L
    text_layers: int = 2
    fusion_layers: int = 2
    temporal_kind: str = "gru"         # "gru" | "transformer"
    history_len: int = 8               # T
    n_apps: int = 64                   # hashed app-id embedding buckets
    max_memories: int = 8              # K
    max_candidates: int = 8            # A (NO_SUGGESTION is added on top)
    n_action_types: int = 6


@dataclass
class CaptureConfig:
    min_interval_s: float = 1.0        # rate limit
    change_threshold: float = 0.01     # fraction of changed pixels to count as "meaningful"
    excluded_apps: list[str] = field(default_factory=lambda: [
        "1password", "keepass", "bitwarden", "lastpass", "dashlane",
        "credentialuibroker", "consent.exe", "logonui",
    ])
    excluded_title_patterns: list[str] = field(default_factory=lambda: [
        "inprivate", "incognito", "private browsing", "password", "sign in",
        "bank", "login",
    ])
    persist_screenshots: bool = False


@dataclass
class MemoryConfig:
    db_path: str = "data/soma_memory.sqlite"
    index_path: str = "data/soma_index.npz"
    top_k: int = 8
    recency_half_life_days: float = 30.0
    allowed_extensions: list[str] = field(default_factory=lambda: [
        ".txt", ".md", ".py", ".json", ".csv", ".png", ".jpg", ".jpeg",
    ])
    max_file_bytes: int = 2_000_000


@dataclass
class PolicyConfig:
    relevance_threshold: float = 0.55  # min prob of best non-abstain action to speak
    margin_over_abstain: float = 0.05
    cooldown_s: float = 30.0
    duplicate_window_s: float = 600.0
    quiet_apps: list[str] = field(default_factory=list)


@dataclass
class TrainConfig:
    seed: int = 0
    batch_size: int = 16
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    val_fraction: float = 0.2
    lambda_retrieval: float = 0.0
    use_amp: bool = True
    checkpoint_dir: str = "checkpoints"
    max_forgetting: float = 0.02       # allowed drop on prior holdout before rollback
    min_improvement: float = 0.0


@dataclass
class SomaConfig:
    device: str = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(dc: Any, data: dict[str, Any]) -> Any:
    """Recursively build dataclass ``dc`` (a type) from a dict, ignoring unknown keys."""
    kwargs = {}
    for f in fields(dc):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = f.type if not isinstance(f.type, str) else globals().get(f.type)
        if ftype is not None and is_dataclass(ftype) and isinstance(value, dict):
            value = _merge(ftype, value)
        kwargs[f.name] = value
    return dc(**kwargs)


def load_config(path: str | Path | None = None) -> SomaConfig:
    """Load a ``SomaConfig`` from YAML; missing keys keep defaults."""
    if path is None:
        return SomaConfig()
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = SOMA_ROOT / p
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return _merge(SomaConfig, data)


def small_model_config() -> ModelConfig:
    """Tiny model dims for fast CPU tests."""
    return ModelConfig(d_model=64, n_heads=4, dropout=0.0, image_size=64, patch_size=16,
                       vision_layers=1, vocab_size=512, max_ocr_tokens=32, text_layers=1,
                       fusion_layers=1, history_len=4, n_apps=16, max_memories=4,
                       max_candidates=4)
