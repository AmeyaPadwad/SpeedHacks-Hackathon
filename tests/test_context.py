import json

from PySide6.QtGui import QColor, QPixmap

from pet.context import MAX_EDGE, MAX_FILE_CHARS, encode_jpeg, read_active_file


def test_read_active_file(tmp_path):
    f = tmp_path / "cur.json"
    f.write_text(json.dumps({"path": "a.py", "language": "python", "content": "x" * (MAX_FILE_CHARS + 5)}))
    af = read_active_file(f)
    assert af.path == "a.py" and af.language == "python"
    assert len(af.content) == MAX_FILE_CHARS


def test_read_active_file_missing_or_malformed(tmp_path):
    assert read_active_file(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert read_active_file(bad) is None
    bad.write_text(json.dumps({"content": "no path"}))
    assert read_active_file(bad) is None


def test_encode_jpeg_downscales(qapp):
    pm = QPixmap(2560, 1440)
    pm.fill(QColor("white"))
    data = encode_jpeg(pm)
    assert data[:2] == b"\xff\xd8"
    out = QPixmap()
    out.loadFromData(data)
    assert max(out.width(), out.height()) == MAX_EDGE


def test_read_diagnostics_and_workspace(tmp_path):
    f = tmp_path / "cur.json"
    f.write_text(json.dumps({"path": "a.py", "workspace": "C:/repo", "diagnostics": [
        {"line": 3, "severity": "error", "message": "x is undefined", "source": "Pylance", "code": "reportUndefinedVariable"},
        {"line": 9, "severity": "warning", "message": "unused import"},
        "garbage",
    ]}))
    af = read_active_file(f)
    assert af.workspace == "C:/repo" and len(af.diagnostics) == 2
    assert [str(d) for d in af.errors] == ["line 3 error [Pylance reportUndefinedVariable]: x is undefined"]
