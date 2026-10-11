import cv2
import numpy as np
from PySide6.QtGui import QColor, QImage

from pet.screen_cv import analyze, qimage_to_bgr

RED = (60, 60, 230)     # BGR, like a VS Code error squiggle / red error text


def ide_screen(dark=True):
    img = np.full((900, 1600, 3), 30 if dark else 245, np.uint8)
    for y in range(80, 860, 22):                                  # neutral "code" lines
        cv2.line(img, (60, y), (60 + (y * 7) % 700, y), (200, 200, 200) if dark else (40, 40, 40), 2)
    return img


def squiggle(img, x, y, length=180):
    for i in range(0, length, 6):
        cv2.line(img, (x + i, y + (i // 6 % 2) * 3), (x + i + 3, y + ((i // 6 + 1) % 2) * 3), RED, 2)


def test_clean_screen_has_no_errors():
    f = analyze(ide_screen())
    assert not f.has_errors and f.crop_jpeg is None and f.red_fraction < 0.001


def test_squiggle_found_and_cropped():
    img = ide_screen()
    squiggle(img, 400, 500)
    f = analyze(img)
    assert f.has_errors
    r = f.regions[0]
    assert 380 <= r.x <= 400 and 490 <= r.y <= 505
    crop = cv2.imdecode(np.frombuffer(f.crop_jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert crop.shape[0] < 300 and crop.shape[1] < img.shape[1]   # focused, not the whole screen


def test_red_text_on_light_theme_and_fingerprint_stable():
    img = ide_screen(dark=False)
    cv2.putText(img, "AssertionError: expected 3 got 2", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
    a, b = analyze(img), analyze(img.copy())
    assert a.has_errors and a.fingerprint() == b.fingerprint() != ""


def test_large_red_banner_is_ignored():
    img = ide_screen()
    img[50:850, 900:1500] = RED      # a big red image/banner, not an error
    assert not analyze(img).has_errors


def test_badges_buttons_and_excluded_areas_ignored():
    img = ide_screen()
    cv2.circle(img, (1500, 60), 9, RED, -1)                       # notification badge
    cv2.rectangle(img, (1300, 800), (1356, 840), RED, -1)         # solid red button
    squiggle(img, 200, 860)                                       # error, but inside an excluded strip
    assert not analyze(img, exclude=[(0, 850, 1600, 50)]).has_errors
    assert analyze(img).has_errors                                 # without exclusion the squiggle is found


def test_qimage_conversion_preserves_colour(qapp):
    q = QImage(37, 11, QImage.Format_RGB32)    # odd width exercises row padding
    q.fill(QColor(230, 60, 60))
    bgr = qimage_to_bgr(q)
    assert bgr.shape == (11, 37, 3) and tuple(bgr[5, 5]) == (60, 60, 230)
