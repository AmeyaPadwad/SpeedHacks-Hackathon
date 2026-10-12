"""Device selection and reproducible seeding."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def select_device(preference: str = "auto") -> torch.device:
    """Return CUDA when available (and allowed), otherwise CPU.

    ``preference`` is ``"auto"``, ``"cuda"`` or ``"cpu"``. Asking for CUDA on a
    machine without it falls back to CPU instead of raising.
    """
    pref = (preference or "auto").lower()
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int = 0) -> None:
    """Seed Python, NumPy and torch RNGs for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_dtype(device: torch.device) -> torch.dtype:
    """Mixed-precision dtype for ``torch.autocast`` on the given device."""
    return torch.float16 if device.type == "cuda" else torch.bfloat16
