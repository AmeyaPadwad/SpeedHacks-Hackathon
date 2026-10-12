"""SOMA runtime: the consent-gated perception -> memory -> policy loop,
a background inference scheduler, opt-in telemetry and component wiring."""

from soma.runtime.factory import (MemoryBundle, SyntheticScreenSource, build_runtime, open_memory,
                                  save_runtime_checkpoint)
from soma.runtime.history import ScreenHistory
from soma.runtime.pipeline import PreparedStep, SomaRuntime, StepResult, fit_dim
from soma.runtime.scheduler import InferenceScheduler
from soma.runtime.telemetry import LatencyStats, Telemetry, percentile

__all__ = [
    "InferenceScheduler", "LatencyStats", "MemoryBundle", "PreparedStep", "ScreenHistory", "SomaRuntime", "StepResult",
    "SyntheticScreenSource", "Telemetry", "build_runtime", "fit_dim", "open_memory", "percentile",
    "save_runtime_checkpoint",
]
