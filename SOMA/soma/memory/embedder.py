"""Local PyTorch embedders for memory items and retrieval queries.

Both embedders produce float32 ``[D]`` unit vectors and are initialised
deterministically from a seed (via a private ``torch.Generator``), so the same
text or image maps to the same vector in every process without shipping
weights. They are untrained bootstrap encoders: the text embedder is a hashed
bag-of-words / character-trigram random projection (which preserves lexical
overlap as cosine similarity); the image embedder is a small conv net over a
downsampled image. A pretrained torchvision backbone can be enabled for images
behind a lazy import.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from soma.config import ModelConfig
from soma.models.tokenizer import token_id, tokenize

DEFAULT_SEED = 1234


def _default_dim() -> int:
    return ModelConfig().d_model


def _features(text: str, vocab_size: int, max_features: int) -> list[int]:
    """Hashed word tokens plus character trigrams of each word (robust to inflection)."""
    ids: list[int] = []
    for tok in tokenize(text):
        ids.append(token_id(tok, vocab_size))
        if tok.isalpha() and len(tok) > 3:
            padded = f"<{tok}>"
            ids.extend(token_id("#" + padded[i:i + 3], vocab_size) for i in range(len(padded) - 2))
        if len(ids) >= max_features:
            break
    return ids[:max_features]


class TextEmbedder(nn.Module):
    """Hashed tokens -> ``nn.EmbeddingBag`` (mean) -> ``nn.Linear`` -> ``LayerNorm`` -> L2 norm."""

    def __init__(self, dim: int | None = None, vocab_size: int = 16384, seed: int = DEFAULT_SEED,
                 max_features: int = 4096) -> None:
        super().__init__()
        self._dim = int(dim or _default_dim())
        self.vocab_size = vocab_size
        self.max_features = max_features
        self.seed = seed
        with torch.random.fork_rng(devices=[]):   # default inits must not touch global RNG
            self.bag = nn.EmbeddingBag(vocab_size, self._dim, mode="mean")
            self.proj = nn.Linear(self._dim, self._dim)
            self.norm = nn.LayerNorm(self._dim)
        self.reset_parameters()
        self.eval()

    def reset_parameters(self) -> None:
        """Deterministic init from ``seed`` that does not touch the global RNG."""
        g = torch.Generator().manual_seed(self.seed)
        with torch.no_grad():
            self.bag.weight.normal_(0.0, 1.0, generator=g)
            # Near-orthogonal projection (identity + small noise) keeps cosine structure.
            noise = torch.empty(self._dim, self._dim).normal_(0.0, 0.1 / self._dim ** 0.5, generator=g)
            self.proj.weight.copy_(torch.eye(self._dim) + noise)
            self.proj.bias.zero_()
            self.norm.reset_parameters()

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """``ids`` flat ``[N]`` long, ``offsets`` ``[B]`` -> unit vectors ``[B, D]``."""
        x = self.bag(ids, offsets)
        x = self.norm(self.proj(x))
        return F.normalize(x, dim=-1)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Batch of strings -> float32 ``[B, D]`` unit vectors (empty text -> UNK vector)."""
        flat: list[int] = []
        offsets: list[int] = []
        for t in texts:
            offsets.append(len(flat))
            feats = _features(t, self.vocab_size, self.max_features) or [1]
            flat.extend(feats)
        with torch.inference_mode():
            out = self.forward(torch.tensor(flat, dtype=torch.long),
                               torch.tensor(offsets, dtype=torch.long))
        return out.cpu().numpy().astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """One string -> float32 ``[D]`` unit vector."""
        return self.embed_texts([text])[0]


class ImageEmbedder(nn.Module):
    """Small conv net over a downsampled RGB image -> float32 ``[D]`` unit vector.

    With ``pretrained=True`` a frozen torchvision ResNet-18 (lazy import, weights
    downloaded by torchvision) replaces the conv trunk; the projection stays
    deterministic.
    """

    def __init__(self, dim: int | None = None, input_size: int = 64, seed: int = DEFAULT_SEED + 1,
                 pretrained: bool = False) -> None:
        super().__init__()
        self._dim = int(dim or _default_dim())
        self.input_size = input_size
        self.seed = seed
        self.pretrained = pretrained
        self.backbone: nn.Module | None = None
        with torch.random.fork_rng(devices=[]):   # default inits must not touch global RNG
            if pretrained:
                self.backbone, feat = self._load_pretrained()
            else:
                self.trunk = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
                )
                feat = 128 * 2  # mean + max pooled
            self.proj = nn.Linear(feat, self._dim)
            self.norm = nn.LayerNorm(self._dim)
        self.reset_parameters()
        self.eval()

    @staticmethod
    def _load_pretrained() -> tuple[nn.Module, int]:
        from torchvision.models import ResNet18_Weights, resnet18  # lazy, optional

        net = resnet18(weights=ResNet18_Weights.DEFAULT)
        net.fc = nn.Identity()
        for p in net.parameters():
            p.requires_grad_(False)
        return net.eval(), 512

    def reset_parameters(self) -> None:
        g = torch.Generator().manual_seed(self.seed)
        with torch.no_grad():
            if not self.pretrained:
                for m in self.trunk.modules():
                    if isinstance(m, nn.Conv2d):
                        fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                        m.weight.normal_(0.0, (2.0 / fan_in) ** 0.5, generator=g)
                        m.bias.zero_()
            self.proj.weight.normal_(0.0, 1.0 / self.proj.in_features ** 0.5, generator=g)
            self.proj.bias.zero_()
            self.norm.reset_parameters()

    @property
    def dim(self) -> int:
        return self._dim

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """``uint8``/float ``[H,W]``, ``[H,W,3]`` or ``[H,W,4]`` -> ``[1,3,S,S]`` in ``[-1, 1]``."""
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = arr[..., :3]
        x = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))
        if arr.dtype == np.uint8 or x.max() > 1.0:
            x = x / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0)
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear",
                          align_corners=False, antialias=True)
        if self.pretrained:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            return (x - mean) / std
        return x * 2.0 - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B,3,S,S]`` -> unit vectors ``[B, D]``."""
        if self.backbone is not None:
            h = self.backbone(x)
        else:
            f = self.trunk(x)
            h = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=-1)
        return F.normalize(self.norm(self.proj(h)), dim=-1)

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """One image array -> float32 ``[D]`` unit vector."""
        with torch.inference_mode():
            out = self.forward(self.preprocess(image))
        return out[0].cpu().numpy().astype(np.float32)
