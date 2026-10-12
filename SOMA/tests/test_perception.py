"""Perception: synthetic screens, OCR engines, screen parser, preprocessing, featurization."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from soma.capture.screen import ScreenCapturer
from soma.capture.window import static_window
from soma.config import CaptureConfig, ModelConfig
from soma.consent import ConsentManager, Scope
from soma.models.tokenizer import PAD_ID
from soma.perception import ocr as ocr_mod
from soma.perception.featurize import app_bucket, batch_observations, observation_to_tensors
from soma.perception.ocr import (
    FallbackOCREngine,
    NullOCREngine,
    OCRUnavailableError,
    RapidOCREngine,
    SyntheticOCREngine,
    get_ocr_engine,
    rapidocr_available,
)
from soma.perception.preprocess import change_ratio, frame_to_tensor
from soma.perception.screen_parser import (
    extract_file_refs,
    find_buttons,
    find_errors,
    find_file_refs,
    infer_role,
    looks_like_error,
    parse_screen,
)
from soma.perception.synthetic import (
    SCENARIOS,
    expected_action,
    ground_truth_for,
    render_scenario,
)
from soma.types import ActionType, OCRElement, ScreenFrame, ScreenObservation, WindowInfo


def frame_of(name: str, seed: int = 0, size=(1280, 800)) -> tuple[ScreenFrame, ScreenObservation]:
    img, gt, _ = render_scenario(name, seed, size)
    return ScreenFrame(image=img, window=gt.window, version=7, timestamp_ms=5), gt


# ----------------------------------------------------------------- synthetic

@pytest.mark.parametrize("name", SCENARIOS)
def test_every_scenario_renders(name):
    img, obs, label = render_scenario(name, seed=3, size=(640, 400))
    assert label == name
    assert img.shape == (400, 640, 3) and img.dtype == np.uint8
    assert obs.elements, name
    for e in obs.elements:
        x0, y0, x1, y1 = e.bbox_xyxy_normalized
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
        assert e.source == "synthetic" and e.ui_role
        assert e.window_id == obs.window.window_id
    assert obs.window.bounds == (0, 0, 640, 400)
    assert isinstance(expected_action(name), ActionType)


def test_synthetic_is_deterministic_per_seed():
    a, oa, _ = render_scenario("terminal_traceback", 11)
    b, ob, _ = render_scenario("terminal_traceback", 11)
    c, oc, _ = render_scenario("terminal_traceback", 12)
    assert np.array_equal(a, b)
    assert [e.text for e in oa.elements] == [e.text for e in ob.elements]
    assert not np.array_equal(a, c) or oa.text != oc.text


def test_unknown_scenario():
    with pytest.raises(KeyError):
        render_scenario("nope", 0)


def test_ground_truth_survives_plain_copy():
    img, gt, _ = render_scenario("browser_docs", 2)
    plain = np.array(img)                    # plain ndarray, no attribute
    assert type(plain) is np.ndarray
    found = ground_truth_for(plain)
    assert found is not None and [e.text for e in found] == [e.text for e in gt.elements]
    assert ground_truth_for(np.zeros((10, 10, 3), np.uint8)) is None


def test_synthetic_screens_have_no_secrets():
    from soma.capture.redaction import contains_secret
    for name in SCENARIOS:
        for seed in range(3):
            _, obs, _ = render_scenario(name, seed, (640, 400))
            assert not any(contains_secret(e.text) for e in obs.elements), name


# ----------------------------------------------------------------- OCR engines

def test_synthetic_ocr_drops_roles_and_sets_ids():
    img, gt, _ = render_scenario("settings_dialog", 1)
    els = SyntheticOCREngine().recognize(img, window_id="w1", timestamp_ms=99)
    assert len(els) == len(gt.elements)
    assert all(e.ui_role is None and e.window_id == "w1" and e.timestamp_ms == 99 for e in els)
    kept = SyntheticOCREngine(keep_roles=True).recognize(img)
    assert {e.ui_role for e in kept} >= {"button", "checkbox"}


def test_synthetic_ocr_noise():
    img, gt, _ = render_scenario("editor_code", 1)
    noisy = SyntheticOCREngine(drop_prob=0.5, jitter=0.01, seed=1).recognize(img)
    assert 0 < len(noisy) < len(gt.elements)
    for e in noisy:
        x0, y0, x1, y1 = e.bbox_xyxy_normalized
        assert 0 <= x0 <= x1 <= 1 and 0 <= y0 <= y1 <= 1


def test_ocr_output_is_redacted(monkeypatch):
    class Leaky:
        name = "leaky"

        def recognize(self, image, window_id="", timestamp_ms=0):
            return ocr_mod._finalize([OCRElement("password=hunter2", (0, 0, 1, 1)),
                                      OCRElement("   ", (0, 0, 1, 1))])

    els = FallbackOCREngine(Leaky()).recognize(np.zeros((8, 8, 3), np.uint8))
    assert len(els) == 1 and "hunter2" not in els[0].text


def test_fallback_without_rapidocr(monkeypatch):
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)   # import fails
    assert not rapidocr_available()
    eng = get_ocr_engine("auto")
    assert isinstance(eng, FallbackOCREngine) and isinstance(eng.primary, NullOCREngine)
    assert eng.recognize(np.zeros((32, 32, 3), np.uint8)) == []
    img, gt, _ = render_scenario("terminal_ok", 0)
    assert len(eng.recognize(img)) == len(gt.elements)          # synthetic still works
    with pytest.raises(OCRUnavailableError):
        get_ocr_engine("rapidocr")
    with pytest.raises(OCRUnavailableError):
        RapidOCREngine().recognize(np.zeros((32, 32, 3), np.uint8))


def test_fallback_swallows_backend_errors():
    class Broken:
        name = "broken"

        def recognize(self, image, window_id="", timestamp_ms=0):
            raise RuntimeError("onnx crashed")

    eng = FallbackOCREngine(Broken())
    assert eng.recognize(np.zeros((16, 16, 3), np.uint8)) == []
    assert "onnx crashed" in eng.last_error


def test_engine_names():
    assert isinstance(get_ocr_engine("null"), NullOCREngine)
    assert isinstance(get_ocr_engine("synthetic"), SyntheticOCREngine)
    with pytest.raises(ValueError):
        get_ocr_engine("tesseract9000")


@pytest.mark.skipif(not rapidocr_available(), reason="rapidocr-onnxruntime not installed")
def test_rapidocr_reads_synthetic_terminal():
    img, _, _ = render_scenario("terminal_traceback", 0)
    els = RapidOCREngine().recognize(np.array(img), window_id="t", timestamp_ms=1)
    assert len(els) >= 5
    assert all(e.source == "ocr" and 0 <= e.bbox_xyxy_normalized[0] < e.bbox_xyxy_normalized[2] <= 1
               for e in els)
    obs = parse_screen(ScreenFrame(np.array(img), WindowInfo("WindowsTerminal.exe", "t"), 1), els)
    assert find_errors(obs)
    assert any(r.line is not None for r in find_file_refs(obs))


# ----------------------------------------------------------------- parser

def test_parser_finds_errors_and_file_refs_in_terminal():
    for seed in range(5):
        frame, gt = frame_of("terminal_traceback", seed)
        obs = parse_screen(frame, SyntheticOCREngine().recognize(frame.image))
        assert obs.version == 7 and obs.timestamp_ms == 5 and obs.window.app_name == "WindowsTerminal.exe"
        errors = [e.text for e in find_errors(obs)]
        assert any(t.startswith("Traceback") for t in errors)
        assert any(t.startswith("FAILED") for t in errors)
        refs = find_file_refs(obs)
        with_lines = [r for r in refs if r.line is not None]
        assert len(with_lines) >= 2
        assert any(r.path.endswith(".py") and r.path.startswith("C:\\dev\\") for r in with_lines)
        assert any(r.path.startswith("tests/test_") for r in refs)       # pytest node id
        assert all(r.element is not None for r in refs)
        roles = {e.text: e.ui_role for e in obs.elements}
        assert any(role == "file_ref" for role in roles.values())


def test_parser_matches_ground_truth_roles():
    for name in SCENARIOS:
        for seed in range(3):
            frame, gt = frame_of(name, seed)
            obs = parse_screen(frame, SyntheticOCREngine().recognize(frame.image))
            pred = {e.text: e.ui_role for e in obs.elements}
            for g in gt.elements:
                key = g.text.strip()
                assert (g.ui_role == "button") == (pred[key] == "button"), (name, key)
                assert (g.ui_role == "error") == (pred[key] == "error"), (name, key)


def test_terminal_ok_has_no_errors_and_settings_has_buttons():
    frame, _ = frame_of("terminal_ok", 4)
    assert find_errors(parse_screen(frame, SyntheticOCREngine().recognize(frame.image))) == []
    frame, _ = frame_of("settings_dialog", 4)
    obs = parse_screen(frame, SyntheticOCREngine().recognize(frame.image))
    assert {e.text for e in find_buttons(obs)} >= {"OK", "Cancel", "Apply"}


def test_parser_joins_accessibility_roles_by_iou():
    win = WindowInfo("App.exe", "x")
    ocr = [OCRElement("Run tests", (0.10, 0.10, 0.20, 0.12)),        # inside the a11y button
           OCRElement("hello world", (0.5, 0.5, 0.7, 0.52)),
           OCRElement("ValueError: bad", (0.1, 0.8, 0.4, 0.82))]
    a11y = [OCRElement("Run tests", (0.09, 0.09, 0.21, 0.13), source="accessibility", ui_role="button"),
            OCRElement("console", (0.0, 0.7, 1.0, 1.0), source="accessibility", ui_role="document"),
            OCRElement("Settings", (0.9, 0.0, 1.0, 0.05), source="accessibility", ui_role="menuitem")]
    obs = parse_screen(None, ocr, a11y, window=win)
    roles = {(e.text, e.source): e.ui_role for e in obs.elements}
    assert roles[("Run tests", "ocr")] == "button"
    assert roles[("ValueError: bad", "ocr")] == "error"        # generic a11y role yields to heuristic
    assert roles[("hello world", "ocr")] == "text"
    assert roles[("Settings", "accessibility")] == "menuitem"   # unmatched a11y kept
    assert ("Run tests", "accessibility") not in roles          # matched a11y not duplicated
    assert ("console", "accessibility") not in roles
    assert obs.window is win
    ys = [e.bbox_xyxy_normalized[1] for e in obs.elements]
    assert ys == sorted(ys)                                     # reading order


@pytest.mark.parametrize("text,expected", [
    ("Traceback (most recent call last):", True),
    ("Traceback(mostrecentcalllast):", True),                   # OCR without spaces
    ("ModuleNotFoundError: No module named 'x'", True),
    ("src/app.ts:4:2 - error TS2322: Type", True),
    ("FAILED tests/test_a.py::test_b - assert 1 == 2", True),
    ("1failed,6passedin0.81s", True),
    ("3 passed in 0.12s", False),
    ("def handle_error(e):", False),
    ("Run tests", False),
])
def test_error_heuristic(text, expected):
    assert looks_like_error(text) is expected


def test_file_ref_extraction():
    refs = extract_file_refs('File "C:\\dev\\app\\main.py", line 12, in run')
    assert refs[0].path == "C:\\dev\\app\\main.py" and refs[0].line == 12
    refs = extract_file_refs("src/utils.py:40:7: E501 line too long")
    assert (refs[0].path, refs[0].line, refs[0].column) == ("src/utils.py", 40, 7)
    refs = extract_file_refs("Program.cs(12,5): error CS1002")
    assert (refs[0].path, refs[0].line, refs[0].column) == ("Program.cs", 12, 5)
    assert extract_file_refs("see https://example.org/guide/index.html") == []
    assert extract_file_refs("nothing here") == []


def test_infer_role_without_image():
    assert infer_role(OCRElement("Cancel", (0, 0, 0.1, 0.1))) == "button"
    assert infer_role(OCRElement("continue", (0, 0, 0.1, 0.1))) == "text"
    assert infer_role(OCRElement("https://docs.python.org", (0, 0, 0.1, 0.1))) == "link"
    assert infer_role(OCRElement("x.py:3", (0, 0, 0.1, 0.1))) == "file_ref"


# ----------------------------------------------------------------- preprocess

def test_frame_to_tensor_shapes_and_normalization():
    img = np.full((300, 500, 3), 128, np.uint8)
    t, tf = frame_to_tensor(img, 64)
    assert t.shape == (3, 64, 64) and t.dtype == torch.float32 and torch.isfinite(t).all()
    t2, _ = frame_to_tensor(img[:, :, 0], 32, normalize=False)      # grayscale accepted
    assert t2.shape == (3, 32, 32) and 0 <= t2.min() and t2.max() <= 1
    with pytest.raises(ValueError):
        frame_to_tensor(np.zeros((4, 4, 1), np.uint8), 8)


def test_change_ratio():
    a, _, _ = render_scenario("editor_code", 0)
    b, _, _ = render_scenario("editor_error", 0)
    assert change_ratio(a, a) == 0.0
    assert change_ratio(a, b) > 0.01
    assert change_ratio(None, a) == 1.0
    assert change_ratio(a, np.zeros((10, 10, 3), np.uint8)) == 1.0
    tiny = np.zeros((5, 7, 3), np.uint8)
    assert change_ratio(tiny, tiny + 50) == 1.0


# ----------------------------------------------------------------- featurize

def test_featurize_shapes_and_padding(small_cfg):
    _, obs, _ = render_scenario("settings_dialog", 0)
    ids, boxes, mask = observation_to_tensors(obs, ModelConfig(max_ocr_tokens=256, vocab_size=512))
    assert ids.shape == (256,) and boxes.shape == (256, 4) and mask.shape == (256,)
    assert ids.dtype == torch.long and boxes.dtype == torch.float32 and mask.dtype == torch.bool
    n = int((~mask).sum())
    assert 0 < n < 256
    assert (ids[mask] == PAD_ID).all() and (ids[~mask] != PAD_ID).all()
    assert (boxes[mask] == 0).all()
    assert ((boxes[~mask] >= 0) & (boxes[~mask] <= 1)).all()
    assert (ids < 512).all()
    # padding is a suffix
    assert not mask[:n].any() and mask[n:].all()


def test_featurize_one_entry_per_word_with_element_box(small_cfg):
    obs = ScreenObservation(WindowInfo("a", "b"), [OCRElement("Run tests", (0.1, 0.2, 0.3, 0.4))], 1)
    ids, boxes, mask = observation_to_tensors(obs, small_cfg)
    assert int((~mask).sum()) == 2
    assert torch.allclose(boxes[0], torch.tensor([0.1, 0.2, 0.3, 0.4]))
    assert torch.allclose(boxes[1], boxes[0])


def test_featurize_truncates_and_prioritizes_errors(small_cfg):
    frame, _ = frame_of("terminal_traceback", 0)
    obs = parse_screen(frame, SyntheticOCREngine().recognize(frame.image))
    ids, boxes, mask = observation_to_tensors(obs, small_cfg)          # L = 32
    assert ids.shape == (small_cfg.max_ocr_tokens,) and not mask.any()
    first_error = find_errors(obs)[0]
    assert torch.allclose(boxes[0], torch.tensor(first_error.bbox_xyxy_normalized, dtype=torch.float32))


def test_featurize_empty(small_cfg):
    for obs in (None, ScreenObservation(WindowInfo("", ""), [], 0)):
        ids, boxes, mask = observation_to_tensors(obs, small_cfg)
        assert mask.all() and (ids == PAD_ID).all() and torch.isfinite(boxes).all()


def test_batch_and_app_bucket(small_cfg):
    _, o1, _ = render_scenario("editor_code", 0)
    _, o2, _ = render_scenario("browser_docs", 0)
    b = batch_observations([o1, o2, None], small_cfg)
    L = small_cfg.max_ocr_tokens
    assert b["ocr_ids"].shape == (3, L) and b["ocr_boxes"].shape == (3, L, 4)
    assert b["ocr_padding_mask"].shape == (3, L) and b["app_ids"].shape == (3,)
    assert (b["app_ids"] < small_cfg.n_apps).all()
    assert app_bucket(WindowInfo("Code.exe", ""), 64) == app_bucket(WindowInfo("code.EXE", "z"), 64)
    assert 0 <= app_bucket(None, 16) < 16


# ----------------------------------------------------------------- end to end

def test_capture_to_tensors_pipeline(small_cfg):
    img, gt, _ = render_scenario("terminal_traceback", 5, (640, 400))
    consent = ConsentManager()
    consent.grant(Scope.SCREEN_CAPTURE)
    cap = ScreenCapturer(consent, CaptureConfig(), grab=lambda w: np.array(img),
                         window_fn=static_window("WindowsTerminal.exe", "pwsh", (0, 0, 640, 400)))
    frame = cap.capture()
    assert frame is not None
    els = get_ocr_engine("auto").recognize(frame.image, frame.window.window_id, frame.timestamp_ms)
    obs = parse_screen(frame, els)
    assert obs.version == frame.version and find_errors(obs)
    x, tf = frame_to_tensor(frame.image, small_cfg.image_size)
    assert x.shape == (3, small_cfg.image_size, small_cfg.image_size)
    ids, boxes, mask = observation_to_tensors(obs, small_cfg)
    assert ids.shape == (small_cfg.max_ocr_tokens,)
