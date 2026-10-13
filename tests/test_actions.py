""""Do it" never runs a file: VS Code at the line, an image viewer, or Explorer's select."""
import pytest

from pet import actions


class Recorder:
    def __init__(self):
        self.calls = []

    def popen(self, args, **kw):
        self.calls.append(("popen", list(args)))

    def startfile(self, path):
        self.calls.append(("startfile", path))


@pytest.fixture
def files(tmp_path):
    py = tmp_path / "calc.py"
    py.write_text("x = 1\n")
    png = tmp_path / "shot.png"
    png.write_bytes(b"\x89PNG")
    return py, png


def test_code_cli_opens_at_the_line(files):
    py, _ = files
    r = Recorder()
    msg = actions.open_file(str(py), 40, which=lambda name: "C:/code.cmd", popen=r.popen, startfile=r.startfile)
    assert r.calls == [("popen", ["C:/code.cmd", "-g", f"{py}:40"])] and "line 40" in msg


def test_without_code_a_script_is_only_selected_in_explorer(files, monkeypatch):
    py, _ = files
    monkeypatch.setattr(actions.sys, "platform", "win32")
    r = Recorder()
    actions.open_file(str(py), None, which=lambda name: None, popen=r.popen, startfile=r.startfile)
    assert r.calls == [("popen", ["explorer", f"/select,{py}"])]       # never startfile on a .py


def test_images_open_in_their_viewer(files):
    _, png = files
    r = Recorder()
    actions.open_file(str(png), None, which=lambda name: "C:/code.cmd", popen=r.popen, startfile=r.startfile)
    assert r.calls == [("startfile", str(png))]


def test_missing_file(tmp_path):
    with pytest.raises(actions.ActionError, match="isn't there"):
        actions.open_file(str(tmp_path / "gone.py"), which=lambda n: None, popen=None, startfile=None)


def test_copy_text():
    got = []
    assert "Copied" in actions.copy_text("hello", got.append) and got == ["hello"]
