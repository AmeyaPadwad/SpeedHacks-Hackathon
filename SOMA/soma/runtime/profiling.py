"""Per-stage latency profiling of the SOMA pipeline on the current machine.

Measures wall-clock time with ``time.perf_counter`` for each stage of one
observation -> suggestion pass (batch size 1, inference mode):

``render`` (synthetic capture stand-in) -> ``ocr_synthetic`` / ``ocr_rapidocr``
-> ``parse`` -> ``preprocess`` + ``featurize`` -> ``visual_encode`` ->
``text_encode`` -> ``screen_fusion`` -> ``temporal`` -> ``retrieval_N<n>`` ->
``memory_fusion`` -> ``candidates`` -> ``policy`` -> ``gate``; plus
``model_forward`` (the full ``SomaModel`` forward in one call).

Results are P50/P95/mean in milliseconds. They describe *this* machine and
configuration only; nothing here is a latency promise. ``torch.profiler`` is
optional (:func:`torch_profile_forward`).
"""

from __future__ import annotations

import os
import platform
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from soma.config import ModelConfig, PolicyConfig
from soma.device import seed_everything, select_device
from soma.models.batch import SomaBatch
from soma.models.soma_model import build_model


def percentile(values: Sequence[float], q: float) -> float:
    """``q``-th percentile (0-100) with linear interpolation; NaN when empty."""
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


class StageTimer:
    """Collects ``perf_counter`` samples per named stage (milliseconds)."""

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)
        self._order: list[str] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - start) * 1000.0)

    def add(self, name: str, ms: float) -> None:
        if name not in self.samples:
            self._order.append(name)
        self.samples[name].append(float(ms))

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name in self._order:
            v = self.samples[name]
            out[name] = {"n": len(v), "p50_ms": percentile(v, 50), "p95_ms": percentile(v, 95),
                         "mean_ms": float(np.mean(v)) if v else float("nan")}
        return out


def _cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def hardware_info() -> dict[str, Any]:
    """CPU / OS / torch / CUDA facts for the benchmark record (no personal data)."""
    info: dict[str, Any] = {
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    return info


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _candidates_for(obs: Any, cfg: ModelConfig) -> list[Any]:
    from soma.policy.candidate_generator import generate_candidates

    return generate_candidates(obs, None, max_candidates=cfg.max_candidates)


@torch.inference_mode()
def profile_pipeline(cfg: ModelConfig, *, iters: int = 30, warmup: int = 3,
                     n_memories: Sequence[int] = (1_000, 10_000), rapidocr: bool | None = None,
                     rapidocr_iters: int = 5, device: str = "cpu", seed: int = 0,
                     scenarios: Sequence[str] | None = None,
                     render_size: tuple[int, int] = (1280, 800)) -> dict[str, Any]:
    """Time each pipeline stage ``iters`` times (after ``warmup`` untimed passes).

    ``rapidocr=None`` profiles RapidOCR only when it is installed. Returns
    ``{"config", "device", "stages": {name: {n, p50_ms, p95_ms, mean_ms}}, ...}``.
    """
    from soma.memory.vector_index import VectorIndex
    from soma.perception import (SCENARIOS, frame_to_tensor, get_ocr_engine, parse_screen,
                                 rapidocr_available, render_scenario)
    from soma.perception.featurize import batch_observations
    from soma.policy.safety_gate import SafetyGate
    from soma.types import ScreenFrame

    dev = select_device(device)
    seed_everything(seed)
    model = build_model(cfg, pretrained=False).to(dev).eval()
    D, T, K, A = cfg.d_model, cfg.history_len, cfg.max_memories, cfg.max_candidates
    names = list(scenarios or SCENARIOS)
    synth_ocr = get_ocr_engine("synthetic")
    use_rapid = rapidocr_available() if rapidocr is None else bool(rapidocr)
    rapid = get_ocr_engine("rapidocr") if use_rapid else None
    gate = SafetyGate(PolicyConfig(), clock=lambda: 0.0)
    rng = np.random.default_rng(seed)
    indexes = {}
    for n in n_memories:
        idx = VectorIndex(D)
        for i, v in enumerate(rng.standard_normal((int(n), D)).astype(np.float32)):
            idx.add(f"m{i}", v)
        indexes[int(n)] = (idx, set(idx.ids()))
    history = torch.randn(1, T, D, device=dev)
    hist_mask = torch.zeros(1, T, dtype=torch.bool, device=dev)
    timer = StageTimer()

    def one_pass(i: int, record: bool) -> None:
        t = timer if record else StageTimer()
        name = names[i % len(names)]
        with t.stage("render"):
            image, obs, _ = render_scenario(name, seed=i, size=render_size)
        with t.stage("ocr_synthetic"):
            elements = synth_ocr.recognize(image)
        if rapid is not None and (not record or i < rapidocr_iters):
            with t.stage("ocr_rapidocr"):
                rapid.recognize(np.asarray(image))
        with t.stage("parse"):
            frame = ScreenFrame(image=np.asarray(image), window=obs.window, version=i)
            parsed = parse_screen(frame, elements)
        with t.stage("preprocess"):
            img_t, _ = frame_to_tensor(np.asarray(image), cfg.image_size)
            images = img_t.unsqueeze(0).to(dev)
        with t.stage("featurize"):
            feats = {k: v.to(dev) for k, v in batch_observations([parsed], cfg).items()}
        with t.stage("visual_encode"):
            visual = model.vision_encoder(images)
            _sync(dev)
        with t.stage("text_encode"):
            text = model.text_layout_encoder(feats["ocr_ids"], feats["ocr_boxes"],
                                             feats["ocr_padding_mask"])
            _sync(dev)
        with t.stage("screen_fusion"):
            screen_state = model.screen_fusion(visual, text, feats["ocr_padding_mask"])
            _sync(dev)
        with t.stage("temporal"):
            temporal = model.temporal_encoder(screen_state, history, hist_mask, feats["app_ids"])
            _sync(dev)
        query = torch.nn.functional.normalize(temporal, dim=-1)[0].float().cpu().numpy()
        hits: list[tuple[str, float]] = []
        for n, (idx, allowed) in indexes.items():
            with t.stage(f"retrieval_N{n}"):
                hits = idx.search(query, K, allowed_ids=allowed)
        mem = torch.from_numpy(np.stack([idx.get(h) for h, _ in hits])) if hits else torch.zeros(0, D)
        mem_emb = torch.zeros(1, K, D, device=dev)
        mem_mask = torch.ones(1, K, dtype=torch.bool, device=dev)
        mem_emb[0, :mem.shape[0]] = mem.to(dev)
        mem_mask[0, :mem.shape[0]] = False
        with t.stage("memory_fusion"):
            fused = model.memory_fusion(temporal, mem_emb, mem_mask)
            _sync(dev)
        with t.stage("candidates"):
            cands = _candidates_for(parsed, cfg)
        n_c = max(1, min(A, len(cands)))
        cand_emb = torch.randn(1, n_c, D, device=dev)
        cand_types = torch.ones(1, n_c, dtype=torch.long, device=dev)
        cand_mask = torch.zeros(1, n_c, dtype=torch.bool, device=dev)
        with t.stage("policy"):
            logits = model.action_policy(fused.context, cand_emb, cand_types, cand_mask)
            _sync(dev)
        with t.stage("gate"):
            gate.decide(logits[0, : len(cands[:n_c]) + 1].float().cpu().numpy(), cands[:n_c], parsed,
                        is_logits=True, record=False)
        batch = SomaBatch(images=images, ocr_ids=feats["ocr_ids"], ocr_boxes=feats["ocr_boxes"],
                          ocr_padding_mask=feats["ocr_padding_mask"], history=history,
                          history_padding_mask=hist_mask, app_ids=feats["app_ids"],
                          memory_embeddings=mem_emb, memory_padding_mask=mem_mask,
                          candidate_embeddings=cand_emb, candidate_types=cand_types,
                          candidate_mask=cand_mask)
        with t.stage("model_forward"):
            model(batch)
            _sync(dev)

    for i in range(warmup):
        one_pass(i, record=False)
    for i in range(iters):
        one_pass(i, record=True)
    n_params = sum(p.numel() for p in model.parameters())
    return {"config": asdict(cfg), "device": str(dev), "iters": int(iters), "warmup": int(warmup),
            "render_size": list(render_size), "n_params": int(n_params),
            "rapidocr_profiled": rapid is not None,
            "rapidocr_iters": int(min(rapidocr_iters, iters)) if rapid is not None else 0,
            "stages": timer.summary()}


@torch.inference_mode()
def torch_profile_forward(cfg: ModelConfig, *, device: str = "cpu", iters: int = 5,
                          row_limit: int = 12, seed: int = 0) -> str:
    """Optional ``torch.profiler`` table for the full model forward on a random batch."""
    from torch.profiler import ProfilerActivity, profile

    from soma.models.batch import random_batch

    dev = select_device(device)
    seed_everything(seed)
    model = build_model(cfg, pretrained=False).to(dev).eval()
    batch = random_batch(1, image_size=cfg.image_size, L=cfg.max_ocr_tokens, T=cfg.history_len,
                         K=cfg.max_memories, A=cfg.max_candidates, D=cfg.d_model,
                         vocab_size=cfg.vocab_size, n_apps=cfg.n_apps,
                         n_action_types=cfg.n_action_types).to(dev)
    model(batch)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    with profile(activities=activities) as prof:
        for _ in range(iters):
            model(batch)
    sort = "cuda_time_total" if dev.type == "cuda" else "cpu_time_total"
    return prof.key_averages().table(sort_by=sort, row_limit=row_limit)


def format_stages(result: dict[str, Any]) -> str:
    """Markdown table of one :func:`profile_pipeline` result."""
    lines = ["| stage | n | P50 ms | P95 ms | mean ms |", "|---|---|---|---|---|"]
    for name, s in result["stages"].items():
        lines.append(f"| {name} | {s['n']} | {s['p50_ms']:.2f} | {s['p95_ms']:.2f} | {s['mean_ms']:.2f} |")
    return "\n".join(lines)
