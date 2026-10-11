from pathlib import Path

import pytest

from pet.inputs.project import ProjectError, ProjectReader, inventory


def write(root, name, content="code\n"):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def test_inventory_respects_nested_gitignore_and_exclusions(tmp_path):
    write(tmp_path, ".gitignore", "ignored/\n*.tmp\n!keep.tmp\n")
    write(tmp_path, "ignored/a.py")
    write(tmp_path, "lost.tmp")
    write(tmp_path, "keep.tmp")
    write(tmp_path, "src/.gitignore", "local.py\n")
    write(tmp_path, "src/local.py")
    write(tmp_path, "src/auth.py")
    write(tmp_path, ".env", "SECRET=123")
    write(tmp_path, ".env.example")
    write(tmp_path, "private.pem")
    write(tmp_path, "node_modules/lib/index.js")
    (tmp_path / "binary.dat").write_bytes(b"\x00\xff")
    (tmp_path / "too_big.py").write_text("x" * 256001)
    paths = inventory(tmp_path).paths
    assert "keep.tmp" in paths and "src/auth.py" in paths
    assert not set(paths) & {"ignored/a.py", "lost.tmp", "src/local.py", ".env", ".env.example",
                             "private.pem", "binary.dat", "too_big.py", "node_modules/lib/index.js"}


def test_reader_rechecks_paths_and_rejects_symlinks(tmp_path):
    outside = write(tmp_path.parent, "outside.py", "secret")
    inside = write(tmp_path, "a.py")
    catalog = inventory(tmp_path)
    (tmp_path / "linked.py").symlink_to(outside)
    assert "linked.py" not in inventory(tmp_path).paths
    reader = ProjectReader(catalog)
    for path in (str(outside), "../outside.py", "linked.py", "missing.py"):
        with pytest.raises(ProjectError):
            reader.read(path)
    inside.unlink()
    inside.symlink_to(outside)
    with pytest.raises(ProjectError):
        reader.read("a.py")


def test_read_and_search_share_budget_and_keep_line_ranges(tmp_path):
    for i in range(15):
        write(tmp_path, f"src/{i:02}.py", "line one\nAUTH_FAILURE\nline three\n")
    reader = ProjectReader(inventory(tmp_path))
    excerpt = reader.read("src/00.py", 2, 2)
    assert excerpt.content == "AUTH_FAILURE\n" and excerpt.start_line == excerpt.end_line == 2
    assert excerpt.truncated
    reader.search("AUTH_FAILURE", "src/")
    reader.search("no match", "src/")
    assert reader.reads <= 12
    with pytest.raises(ProjectError):
        reader.read("src/14.py")


def test_source_budget_and_large_excerpt_notice(tmp_path):
    for i in range(5):
        write(tmp_path, f"{i}.py", "x" * 16000)
    reader = ProjectReader(inventory(tmp_path))
    for i in range(4):
        assert reader.read(f"{i}.py").truncated
    assert reader.source_chars == 48000
    with pytest.raises(ProjectError):
        reader.read("4.py")


def test_invalid_range_and_missing_root(tmp_path):
    write(tmp_path, "a.py")
    reader = ProjectReader(inventory(tmp_path))
    for start, end in ((0, 3), (3, 2), (1, 502), (99, 100)):
        with pytest.raises(ProjectError):
            reader.read("a.py", start, end)
    with pytest.raises(ProjectError):
        inventory(tmp_path / "absent")
