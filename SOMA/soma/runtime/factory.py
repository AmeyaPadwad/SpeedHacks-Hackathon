"""Wire real SOMA components into a :class:`~soma.runtime.pipeline.SomaRuntime`.

* Memory lives under ``data_dir`` (SQLite + ``.npz`` vector index), embedded
  with ``TextEmbedder(dim=cfg.model.d_model)`` so retrieved vectors feed the
  model's memory slots without projection.
* The model comes from a checkpoint when given; otherwise ``build_model`` with
  a fixed seed: a *dummy but real* PyTorch model (untrained weights, genuine
  forward pass), which is what the v0.1 desktop shows before training.
* ``synthetic=True`` swaps real capture for a :class:`SyntheticScreenSource`
  that cycles :func:`~soma.perception.synthetic.render_scenario` frames and
  the synthetic OCR engine, so the app runs without screen access. Consent
  gates still apply exactly as for real capture.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from soma.capture.accessibility import read_accessibility
from soma.capture.screen import ScreenCapturer
from soma.config import SomaConfig
from soma.consent import ConsentManager
from soma.device import select_device
from soma.learning.feedback import FeedbackRecorder
from soma.learning.online_ranker import OnlinePreferenceRanker
from soma.memory.embedder import ImageEmbedder, TextEmbedder
from soma.memory.ingestion import Ingestor
from soma.memory.retrieval import Retriever
from soma.memory.sqlite_store import MemoryStore
from soma.memory.vector_index import VectorIndex
from soma.models.checkpoint import load_checkpoint, save_checkpoint
from soma.models.soma_model import build_model
from soma.perception.ocr import get_ocr_engine
from soma.perception.synthetic import SCENARIOS, render_scenario
from soma.policy.candidate_encoder import CandidateEncoder
from soma.policy.safety_gate import SafetyGate
from soma.runtime.pipeline import SomaRuntime, seeded_candidate_encoder
from soma.runtime.telemetry import Telemetry
from soma.types import WindowInfo

DB_NAME = "soma_memory.sqlite"
INDEX_NAME = "soma_index.npz"
CANDIDATE_ENCODER_KEY = "candidate_encoder"


@dataclass
class MemoryBundle:
    """Everything the runtime, API and scripts need to use local memory."""

    store: MemoryStore
    index: VectorIndex
    text_embedder: TextEmbedder
    image_embedder: ImageEmbedder
    retriever: Retriever
    ingestor: Ingestor

    def close(self) -> None:
        self.index.persist()
        self.store.close()


def open_memory(cfg: SomaConfig, consent: ConsentManager, data_dir: str | Path) -> MemoryBundle:
    """Open (or create) the memory store and vector index inside ``data_dir``.

    File names come from ``cfg.memory`` (only the base names; everything is
    placed under ``data_dir``). Raises ``ValueError`` when an existing index
    was built with a different embedding dimension than ``cfg.model.d_model``.
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    D = int(cfg.model.d_model)
    db_path = root / (Path(cfg.memory.db_path).name or DB_NAME)
    index_path = root / (Path(cfg.memory.index_path).name or INDEX_NAME)
    index = VectorIndex.open(index_path, dim=D)
    if index.dim != D:
        raise ValueError(f"memory index {index_path} has dim {index.dim} but the model uses "
                         f"d_model={D}; rebuild memory (scripts.build_memory) for this config")
    store = MemoryStore(db_path)
    text = TextEmbedder(dim=D)
    image = ImageEmbedder(dim=D)
    retriever = Retriever(store, index, consent, cfg.memory)
    ingestor = Ingestor(store, index, consent, text, image, cfg.memory)
    return MemoryBundle(store=store, index=index, text_embedder=text, image_embedder=image,
                        retriever=retriever, ingestor=ingestor)


class SyntheticScreenSource:
    """Cycle synthetic GUI scenarios as a fake foreground window + grab function.

    Each scenario stays on screen for ``dwell_s`` seconds (re-grabs of the
    same scenario are pixel-identical, so change detection skips them), then
    the next one is shown. Use :meth:`window` as ``window_fn`` and
    :meth:`grab` as ``grab`` for :class:`ScreenCapturer`.
    """

    def __init__(self, scenarios: Sequence[str] = SCENARIOS, *, dwell_s: float = 4.0,
                 size: tuple[int, int] = (1280, 800), seed: int = 0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not scenarios:
            raise ValueError("need at least one scenario")
        self.scenarios = list(scenarios)
        self.dwell_s = float(dwell_s)
        self.size = size
        self.seed = int(seed)
        self.clock = clock
        self._t0 = clock()
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[np.ndarray, WindowInfo]] = {}
        self.current = self.scenarios[0]

    def _render(self, name: str) -> tuple[np.ndarray, WindowInfo]:
        if name not in self._cache:
            image, obs, _ = render_scenario(name, seed=self.seed, size=self.size)
            win = dataclasses.replace(obs.window, window_id=f"synthetic:{name}")
            self._cache[name] = (image, win)
        return self._cache[name]

    def window(self) -> WindowInfo:
        with self._lock:
            i = int(max(0.0, self.clock() - self._t0) // max(self.dwell_s, 1e-6)) % len(self.scenarios)
            self.current = self.scenarios[i]
            return self._render(self.current)[1]

    def grab(self, window: WindowInfo) -> np.ndarray:
        with self._lock:
            return self._render(self.current)[0]


def _load_candidate_encoder(cfg: SomaConfig, extra: dict) -> CandidateEncoder:
    enc = seeded_candidate_encoder(cfg)
    state = extra.get(CANDIDATE_ENCODER_KEY) if extra else None
    if state:
        enc.load_state_dict(state)
    return enc.eval()


def save_runtime_checkpoint(runtime: SomaRuntime, path: str | Path, extra: dict | None = None) -> Path:
    """Save the runtime's model *and* candidate encoder so a reload scores identically."""
    payload = dict(extra or {})
    payload[CANDIDATE_ENCODER_KEY] = {k: v.detach().cpu()
                                      for k, v in runtime.candidate_encoder.state_dict().items()}
    return save_checkpoint(runtime.model, path, runtime.cfg.model, extra=payload)


def build_runtime(cfg: SomaConfig, consent: ConsentManager, *, data_dir: Path,
                  checkpoint: Path | None = None, synthetic: bool = False,
                  telemetry: bool = False, synthetic_dwell_s: float = 4.0,
                  clock: Callable[[], float] = time.monotonic) -> SomaRuntime:
    """Build a runtime from real components (see module docstring).

    ``telemetry=True`` opts in to local latency/rate metrics.
    """
    device = select_device(cfg.device)
    extra: dict = {}
    if checkpoint is not None:
        model, mcfg, extra = load_checkpoint(checkpoint, map_location=device)
        cfg = dataclasses.replace(cfg, model=mcfg)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.train.seed)
            model = build_model(cfg.model, pretrained=cfg.model.vision_backbone != "scratch")
    encoder = _load_candidate_encoder(cfg, extra)
    memory = open_memory(cfg, consent, data_dir)

    accessibility = None
    if synthetic:
        source = SyntheticScreenSource(dwell_s=synthetic_dwell_s, clock=clock)
        capturer = ScreenCapturer(consent, cfg.capture, grab=source.grab, window_fn=source.window,
                                  clock=clock)
        ocr = get_ocr_engine("synthetic")
    else:
        capturer = ScreenCapturer(consent, cfg.capture, clock=clock)
        ocr = get_ocr_engine("auto")
        accessibility = lambda w: read_accessibility(w, consent, cfg.capture)  # noqa: E731

    runtime = SomaRuntime(
        cfg, consent, capturer=capturer, ocr=ocr, model=model, memory=memory,
        gate=SafetyGate(cfg.policy, clock=clock), candidate_encoder=encoder, device=device,
        clock=clock, accessibility=accessibility, ranker=OnlinePreferenceRanker(),
        recorder=FeedbackRecorder(consent, memory.store),
        telemetry=Telemetry(enabled=telemetry, clock=clock),
    )
    runtime.synthetic = synthetic  # type: ignore[attr-defined]
    return runtime
