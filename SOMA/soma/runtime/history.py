"""Rolling, in-memory buffer of recent screen states for the temporal encoder.

The buffer holds at most ``T`` detached ``[D]`` screen-state vectors (the
``SomaModel.encode_screen`` output of previous frames), oldest first. It is
never persisted, holds no pixels or text, and is cleared on pause so a resumed
session never mixes with what was on screen before the pause.
"""

from __future__ import annotations

import threading
from collections import deque

import torch


class ScreenHistory:
    """Bounded FIFO of previous screen states.

    ``tensors()`` returns ``(history [T, D], padding_mask [T])`` with the most
    recent state last and left padding (``True`` = PAD), matching
    ``SomaBatch.history`` / ``history_padding_mask`` without the batch dim.
    """

    def __init__(self, max_len: int, dim: int, *, reset_on_app_switch: bool = False) -> None:
        if max_len < 1:
            raise ValueError("max_len must be >= 1")
        self.max_len = int(max_len)
        self.dim = int(dim)
        self.reset_on_app_switch = reset_on_app_switch
        self._states: deque[torch.Tensor] = deque(maxlen=self.max_len)
        self._app: str | None = None
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._states)

    def clear(self) -> None:
        with self._lock:
            self._states.clear()
            self._app = None

    def observe_app(self, app: str) -> None:
        """Note the foreground app; clears the buffer on a switch when configured."""
        app = (app or "").lower()
        with self._lock:
            if self.reset_on_app_switch and self._app is not None and app != self._app:
                self._states.clear()
            self._app = app

    def push(self, state: torch.Tensor) -> None:
        """Append one ``[D]`` (or ``[1, D]``) state as a detached CPU float copy."""
        s = state.detach().reshape(-1).to("cpu", torch.float32).clone()
        if s.shape[0] != self.dim:
            raise ValueError(f"state has dim {s.shape[0]}, expected {self.dim}")
        with self._lock:
            self._states.append(s)

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(history [T, D] float32, padding_mask [T] bool)``; True = PAD."""
        hist = torch.zeros(self.max_len, self.dim, dtype=torch.float32)
        mask = torch.ones(self.max_len, dtype=torch.bool)
        with self._lock:
            states = list(self._states)
        n = len(states)
        if n:
            hist[self.max_len - n:] = torch.stack(states)
            mask[self.max_len - n:] = False
        return hist, mask
