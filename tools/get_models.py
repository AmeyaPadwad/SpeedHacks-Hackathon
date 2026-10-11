"""Download the MediaPipe face and hand landmarker models into models/ (gitignored)."""
import urllib.request
from pathlib import Path

MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

out = Path(__file__).resolve().parent.parent / "models"
out.mkdir(exist_ok=True)
for name, url in MODELS.items():
    dest = out / name
    if dest.exists():
        print(f"have {dest}")
        continue
    urllib.request.urlretrieve(url, dest)
    print(f"wrote {dest} ({dest.stat().st_size:,} bytes)")
