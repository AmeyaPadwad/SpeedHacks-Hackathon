"""Consent-gated ingestion of a user-selected folder, preferences and UI events.

Rules enforced here:

- A folder is indexed only when ``consent.path_allowed(folder)`` (which also
  requires the FOLDER_INDEXING scope); home directories and drive roots are
  refused even if somehow allowed.
- The walk never leaves the folder: hidden/system directories are skipped,
  symlinks are not followed, and every file is re-checked against consent.
- Only allowed extensions under ``max_file_bytes`` are read. Raw contents stay
  in memory; only a short redacted summary, an embedding and provenance
  (path, mtime, size, sha256, ingest time) are stored.
- Re-ingesting an unchanged file (same sha256) is a no-op.
- Image thumbnails are written only when ``save_thumbnails=True``.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from soma.config import MemoryConfig
from soma.consent import ConsentError, ConsentManager, Scope
from soma.memory.embedder import ImageEmbedder, TextEmbedder
from soma.memory.sqlite_store import MemoryStore, norm_path, path_within, utc_now_iso
from soma.memory.vector_index import VectorIndex
from soma.types import MemoryItem, SourceType, new_id

TEXT_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv"}
OPTIONAL_TEXT_EXTENSIONS = {".pdf", ".docx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
SKIP_DIR_NAMES = {"__pycache__", "node_modules", ".git", ".venv", "venv", "$recycle.bin",
                  "system volume information", ".svn", ".hg", ".idea", ".vscode"}
EMBED_CHARS = 20_000          # text (redacted) used for the embedding; never stored
THUMB_SIZE = 128

# ---------------------------------------------------------------- redaction

_FALLBACK_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\b(?:sk|pk|rk|ghp|gho|ghs|xox[abprs]|AKIA|AIza)[-_A-Za-z0-9]{12,}\b"),
    re.compile(r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)\b\s*[:=]\s*\S+"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),
]


def _fallback_redact(text: str) -> str:
    for pat in _FALLBACK_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def redact(text: str) -> str:
    """Redact secrets with ``soma.capture.redaction`` when available, plus a local fallback."""
    try:
        from soma.capture.redaction import redact_text  # lazy: written concurrently
        text = redact_text(text)
    except Exception:
        pass
    return _fallback_redact(text)


# ---------------------------------------------------------------- extraction


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_text(path: Path, data: bytes) -> str | None:
    """Plain text for supported types; None when the type is unsupported here."""
    ext = path.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        try:
            import fitz  # pymupdf, optional
        except Exception:
            return None
        with fitz.open(stream=data, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    if ext == ".docx":
        try:
            import docx  # python-docx, optional
        except Exception:
            return None
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    return None


def _summarise(text: str, n: int) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed[:n]


def _is_hidden_or_system(p: Path) -> bool:
    if p.name.startswith(".") or p.name.lower() in SKIP_DIR_NAMES:
        return True
    try:
        attrs = getattr(os.lstat(p), "st_file_attributes", 0)
    except OSError:
        return True
    hidden = getattr(stat, "FILE_ATTRIBUTE_HIDDEN", 2)
    system = getattr(stat, "FILE_ATTRIBUTE_SYSTEM", 4)
    return bool(attrs & (hidden | system))


def item_id_for(path: str | Path) -> str:
    """Stable memory id for a file path (same file -> same id across runs)."""
    return "mem_" + hashlib.sha1(norm_path(path).encode("utf-8")).hexdigest()[:20]


@dataclass
class IngestReport:
    """Outcome of one ``ingest_folder`` call (paths only, no content)."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return len(self.added) + len(self.updated)


class Ingestor:
    """Builds permissioned memories from allowed folders and explicit user input."""

    def __init__(self, store: MemoryStore, index: VectorIndex, consent: ConsentManager,
                 text_embedder: TextEmbedder, image_embedder: ImageEmbedder | None = None,
                 cfg: MemoryConfig | None = None, summary_chars: int = 400,
                 save_thumbnails: bool = False,
                 redactor: Callable[[str], str] = redact) -> None:
        self.store = store
        self.index = index
        self.consent = consent
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.cfg = cfg or MemoryConfig()
        self.summary_chars = summary_chars
        self.save_thumbnails = save_thumbnails
        self.redact = redactor
        self.allowed_ext = {e.lower() for e in self.cfg.allowed_extensions}

    # -- folders
    def _refuse_broad(self, folder: Path) -> None:
        if folder == Path.home().resolve() or folder.parent == folder:
            raise ConsentError("Refusing to index an entire home directory or drive root.")

    def ingest_folder(self, folder: str | Path, trust: float | None = None) -> IngestReport:
        """Index allowed files directly inside ``folder`` (recursively).

        Raises :class:`ConsentError` when the folder is not explicitly allowed.
        """
        root = Path(folder).expanduser().resolve()
        self._refuse_broad(root)
        if not self.consent.path_allowed(root):
            raise ConsentError(f"Folder is not allowed for indexing: {root}")
        if not root.is_dir():
            raise NotADirectoryError(str(root))
        existing = self.store.get_source(root)
        if existing and existing["blocked"]:
            raise ConsentError(f"Source is blocked: {root}")
        self.store.set_source(root, allowed=True,
                              trust=trust if trust is not None else (existing or {}).get("trust", 0.5))

        report = IngestReport()
        for path in self._walk(root, report):
            try:
                self._ingest_file(path, root, report)
            except Exception as exc:  # one bad file must not stop the walk
                report.errors.append(f"{path}: {type(exc).__name__}")
        if report.changed:
            self.index.persist()
        return report

    def _walk(self, root: Path, report: IngestReport) -> Iterable[Path]:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            d = Path(dirpath)
            dirnames[:] = sorted(n for n in dirnames
                                 if not (d / n).is_symlink() and not _is_hidden_or_system(d / n))
            for name in sorted(filenames):
                p = d / name
                if _is_hidden_or_system(p):
                    report.skipped.append(str(p))
                    continue
                if p.is_symlink():
                    target = p.resolve()
                    if not path_within(target, root):
                        report.skipped.append(str(p))
                        continue
                if p.suffix.lower() not in self.allowed_ext:
                    report.skipped.append(str(p))
                    continue
                yield p

    def _ingest_file(self, path: Path, root: Path, report: IngestReport) -> None:
        resolved = path.resolve()
        if not path_within(resolved, root) or not self.consent.path_allowed(resolved):
            report.skipped.append(str(path))
            return
        st = resolved.stat()
        if st.st_size > self.cfg.max_file_bytes:
            report.skipped.append(str(path))
            return
        data = resolved.read_bytes()
        digest = _sha256(data)
        item_id = item_id_for(resolved)
        prior = self.store.get_item(item_id)
        if prior and prior.provenance.get("sha256") == digest and item_id in self.index:
            report.unchanged.append(str(path))
            return

        ext = resolved.suffix.lower()
        provenance = {"path": str(resolved), "root": str(root), "mtime": st.st_mtime,
                      "size": st.st_size, "sha256": digest, "ingested_at": utc_now_iso()}
        if ext in IMAGE_EXTENSIONS:
            if self.image_embedder is None:
                report.skipped.append(str(path))
                return
            vec, summary, stype = self._image(resolved, data, item_id, st.st_size, provenance)
        else:
            text = extract_text(resolved, data)
            if text is None:
                report.skipped.append(str(path))
                return
            clean = self.redact(text[:EMBED_CHARS])
            summary = f"{resolved.name}: {_summarise(clean, self.summary_chars)}"
            vec = self.text_embedder.embed_text(f"{resolved.stem} {clean}")
            stype = SourceType.DOCUMENT
        del data

        item = MemoryItem(id=item_id, source_type=stype, user_approved=True,
                          text_summary=summary, source_uri=str(resolved),
                          tags=[ext.lstrip(".")], provenance=provenance,
                          created_at=prior.created_at if prior else "")
        self.store.upsert_item(item)
        self.index.add(item_id, vec)
        (report.updated if prior else report.added).append(str(path))

    def _image(self, path: Path, data: bytes, item_id: str, size: int,
               provenance: dict) -> tuple[np.ndarray, str, SourceType]:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            w, h = im.size
            vec = self.image_embedder.embed_image(np.asarray(im))
            if self.save_thumbnails:
                self.store.derived_dir.mkdir(parents=True, exist_ok=True)
                thumb = self.store.derived_dir / f"{item_id}.png"
                im.thumbnail((THUMB_SIZE, THUMB_SIZE))
                im.save(thumb, format="PNG")
                provenance["thumbnail"] = str(thumb)
        summary = f"Image {path.name} ({w}x{h}, {max(1, size // 1024)} KB)"
        return vec, summary, SourceType.IMAGE

    # -- explicit user memories
    def _add_text_memory(self, text: str, source_type: SourceType, tags: Iterable[str],
                         provenance: dict, expires_at: str | None) -> MemoryItem:
        clean = self.redact(text)
        item = MemoryItem(id=new_id("mem_"), source_type=source_type, user_approved=True,
                          text_summary=clean, tags=list(tags),
                          provenance={**provenance, "ingested_at": utc_now_iso()},
                          expires_at=expires_at)
        self.store.upsert_item(item)
        self.index.add(item.id, self.text_embedder.embed_text(clean))
        self.index.persist()
        return item

    def add_preference(self, text: str, tags: Iterable[str] = (), conflict_key: str | None = None,
                       expires_at: str | None = None) -> MemoryItem:
        """Store an explicit user preference. ``conflict_key`` marks preferences that
        supersede each other (the most recent wins at retrieval)."""
        tags = list(tags)
        if conflict_key:
            tags.append(f"conflict:{conflict_key}")
        return self._add_text_memory(text, SourceType.PREFERENCE, tags,
                                     {"origin": "user"}, expires_at)

    def add_ui_event(self, description: str, tags: Iterable[str] = (), app: str | None = None,
                     expires_at: str | None = None) -> MemoryItem:
        """Store a short description of a UI event (requires FEEDBACK_STORAGE consent)."""
        self.consent.require(Scope.FEEDBACK_STORAGE)
        prov = {"origin": "ui-event"}
        if app:
            prov["app"] = app
        return self._add_text_memory(description, SourceType.UI_EVENT, tags, prov, expires_at)

    def add_feedback_memory(self, text: str, suggestion_id: str, tags: Iterable[str] = (),
                            expires_at: str | None = None) -> MemoryItem:
        """Store a feedback-derived memory (requires FEEDBACK_STORAGE consent)."""
        self.consent.require(Scope.FEEDBACK_STORAGE)
        return self._add_text_memory(text, SourceType.FEEDBACK, tags,
                                     {"origin": "feedback", "suggestion_id": suggestion_id},
                                     expires_at)
