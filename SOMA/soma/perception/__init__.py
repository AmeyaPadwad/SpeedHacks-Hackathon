"""SOMA perception package: preprocessing, OCR, screen parsing, featurization
and synthetic GUI screens."""

from soma.perception.featurize import app_bucket, batch_observations, observation_to_tensors
from soma.perception.ocr import (
    FallbackOCREngine,
    NullOCREngine,
    OCREngine,
    OCRUnavailableError,
    RapidOCREngine,
    SyntheticOCREngine,
    get_ocr_engine,
    rapidocr_available,
)
from soma.perception.preprocess import LetterboxTransform, change_ratio, frame_to_tensor
from soma.perception.screen_parser import (
    FileRef,
    draw_observation,
    find_buttons,
    find_errors,
    find_file_refs,
    iou,
    parse_screen,
    screen_to_window,
    to_normalized,
    to_pixels,
    window_to_screen,
)
from soma.perception.synthetic import SCENARIOS, expected_action, ground_truth_for, render_scenario

__all__ = [
    "frame_to_tensor", "change_ratio", "LetterboxTransform",
    "OCREngine", "RapidOCREngine", "SyntheticOCREngine", "NullOCREngine", "FallbackOCREngine",
    "OCRUnavailableError", "get_ocr_engine", "rapidocr_available",
    "parse_screen", "draw_observation", "find_errors", "find_file_refs", "find_buttons", "FileRef",
    "to_pixels", "to_normalized", "window_to_screen", "screen_to_window", "iou",
    "observation_to_tensors", "batch_observations", "app_bucket",
    "SCENARIOS", "render_scenario", "expected_action", "ground_truth_for",
]
