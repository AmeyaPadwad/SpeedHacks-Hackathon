import json
import shutil
import subprocess

import pytest

from pet.context import MAX_FILE_CHARS, ActiveFile, Context, git_changes, read_active_file


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


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_changes_diff_untracked_and_clean(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init")
    f = tmp_path / "app.py"
    f.write_text("x = 1\n")
    af = ActiveFile(str(f), "python", "x = 1\n")
    assert git_changes(af).startswith("new file app.py (untracked)") and "+x = 1" in git_changes(af)
    _git(tmp_path, "add", "app.py")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "add")
    assert git_changes(af) == ""                                  # clean
    f.write_text("x = 2\n")
    diff = git_changes(af)
    assert "-x = 1" in diff and "+x = 2" in diff


def test_git_changes_outside_a_repo_or_without_git(tmp_path):
    af = ActiveFile(str(tmp_path / "a.py"), "python", "")
    assert git_changes(None) == "" and git_changes(ActiveFile("Z:/nope/a.py", "python", "")) == ""

    def no_git(*a, **k):
        raise OSError("git missing")
    assert git_changes(af, run=no_git) == ""


def test_capture_has_no_screenshot(tmp_path, monkeypatch):
    import pet.context as c
    monkeypatch.setattr(c, "read_active_file", lambda: None)
    ctx = c.capture()
    assert ctx == Context(None, "") and not hasattr(ctx, "screenshot_jpeg")


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
