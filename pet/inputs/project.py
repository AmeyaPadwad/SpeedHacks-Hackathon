"""Read-only, bounded project access shared by manual and LLM file selection."""
import os
import codecs
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pathspec

from .packet import FileExcerpt

MAX_INVENTORY = 2_000
MAX_ENTRIES = 20_000
MAX_FILE_BYTES = 256_000
MAX_READS = 12
MAX_SOURCE_CHARS = 48_000
EXCLUDED_DIRS = {".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
                 "dist", "build", ".next", ".cache", ".aws", ".ssh", ".codex", ".agents"}
EXCLUDED_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".pyc", ".png", ".jpg", ".jpeg",
                     ".gif", ".webp", ".pdf", ".zip", ".sqlite", ".db", ".wav", ".mp4",
                     ".lock", ".so", ".dylib", ".exe", ".ttf", ".woff", ".woff2"}


class ProjectError(Exception):
    pass


def sensitive(path: str) -> bool:
    p = PurePosixPath(path)
    name = p.name.lower()
    return (any(part.lower() in EXCLUDED_DIRS for part in p.parts)
            or name.startswith(".env") or name.startswith("id_rsa") or name.startswith("id_ed25519")
            or name in {"credentials", "credentials.json", "secrets.json", "secrets.yaml", "secrets.yml"}
            or p.suffix.lower() in EXCLUDED_SUFFIXES)


@dataclass(frozen=True)
class Inventory:
    root: Path
    paths: tuple[str, ...]
    truncated: bool = False


def inventory(root: Path) -> Inventory:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ProjectError("Choose an existing project directory.")
    paths, entries = [], 0
    rules_by_dir = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        directory = Path(directory)
        inherited = list(rules_by_dir.get(directory.parent, []))
        ignore = directory / ".gitignore"
        try:
            if not ignore.is_symlink() and ignore.stat().st_size <= MAX_FILE_BYTES:
                inherited.append((directory, pathspec.GitIgnoreSpec.from_lines(ignore.read_text().splitlines())))
        except (OSError, UnicodeError):
            pass
        rules_by_dir[directory] = inherited

        def ignored(p, is_dir=False):
            ignored_here = False
            for base, spec in inherited:
                match = spec.check_file(p.relative_to(base).as_posix() + ("/" if is_dir else ""))
                if match.include is not None:
                    ignored_here = match.include
            return ignored_here

        entries += len(dirs) + len(files)
        if entries > MAX_ENTRIES:
            return Inventory(root, tuple(paths), True)
        dirs[:] = sorted(d for d in dirs if not (directory / d).is_symlink()
                         and not sensitive((directory / d).relative_to(root).as_posix())
                         and not ignored(directory / d, True))
        for name in sorted(files):
            p = directory / name
            rel = p.relative_to(root).as_posix()
            if p.is_symlink() or sensitive(rel) or ignored(p):
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
                with p.open("rb") as f:
                    head = f.read(4096)
                if b"\x00" in head:
                    continue
                codecs.getincrementaldecoder("utf-8")().decode(head, final=False)
            except (OSError, UnicodeError):
                continue
            paths.append(rel)
            if len(paths) >= MAX_INVENTORY:
                return Inventory(root, tuple(paths), True)
    return Inventory(root, tuple(paths))


class ProjectReader:
    def __init__(self, catalog: Inventory):
        self.catalog = catalog
        self.reads = 0
        self.source_chars = 0
        self.excerpts: dict[tuple[str, int, int], FileExcerpt] = {}

    def read(self, path: str, start_line: int = 1, end_line: int = 200,
             reason: str = "Selected manually") -> FileExcerpt:
        if (not isinstance(path, str) or path not in self.catalog.paths or sensitive(path)
                or not isinstance(start_line, int) or not isinstance(end_line, int)
                or start_line < 1 or end_line < start_line or end_line - start_line > 500):
            raise ProjectError("Choose an eligible project file and a range of at most 501 lines.")
        key = (path, start_line, end_line)
        if key in self.excerpts:
            return self.excerpts[key]
        if self.reads >= MAX_READS or self.source_chars >= MAX_SOURCE_CHARS:
            raise ProjectError("File selection reached its read/content budget.")
        file = self.catalog.root / path
        # Re-check at read time, since files can change after inventory creation.
        if any(p.is_symlink() for p in (file, *file.parents) if p != self.catalog.root):
            raise ProjectError("Symlink files and directories are excluded.")
        if not file.resolve().is_relative_to(self.catalog.root):
            raise ProjectError("File is outside the connected project.")
        try:
            with file.open("rb") as stream:
                raw = stream.read(MAX_FILE_BYTES + 1)
            if len(raw) > MAX_FILE_BYTES or b"\x00" in raw:
                raise ProjectError("File is too large or contains binary data.")
            lines = raw.decode("utf-8").splitlines(keepends=True)
        except (OSError, UnicodeError) as e:
            raise ProjectError("File is unavailable or is not UTF-8 text.") from e
        if start_line > max(1, len(lines)):
            raise ProjectError("Requested range is beyond the end of the file.")
        content = "".join(lines[start_line - 1:end_line])
        limit = min(12_000, MAX_SOURCE_CHARS - self.source_chars)
        truncated = len(content) > limit or end_line < len(lines) or start_line > 1
        content = content[:limit]
        actual_end = start_line + max(0, len(content.splitlines()) - 1)
        result = FileExcerpt(path, start_line, actual_end, content, reason[:300], truncated)
        self.reads += 1
        self.source_chars += len(content)
        self.excerpts[key] = result
        return result

    def search(self, term: str, prefix: str = "") -> list[dict]:
        """Literal search only. Reading for search consumes the same twelve-read budget."""
        if not isinstance(term, str) or not 1 <= len(term) <= 100 or not isinstance(prefix, str):
            raise ProjectError("Search needs a short literal term and a relative prefix.")
        matches = []
        for path in self.catalog.paths:
            if not path.startswith(prefix):
                continue
            if self.reads >= MAX_READS or self.source_chars >= MAX_SOURCE_CHARS:
                break
            excerpt = self.read(path, reason="Read during text search")
            for offset, line in enumerate(excerpt.content.splitlines()):
                if term.casefold() in line.casefold():
                    matches.append({"path": path, "line": excerpt.start_line + offset, "text": line[:300]})
                    break
            if len(matches) >= 6:
                break
        return matches
