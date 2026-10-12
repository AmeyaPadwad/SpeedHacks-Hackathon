"""SOMA local multimodal memory (milestone M2).

Permissioned SQLite metadata + cosine vector index + deterministic PyTorch
embedders. Retrieval filters by consent, approval, expiry and source blocks
*before* any vector is scored. Users can view, export, correct and forget.
"""

from soma.memory.deletion import forget, forget_source, purge_expired, verify_deleted
from soma.memory.embedder import ImageEmbedder, TextEmbedder
from soma.memory.export import correct, export_memories, list_for_viewer
from soma.memory.ingestion import IngestReport, Ingestor, item_id_for, redact
from soma.memory.retrieval import RerankWeights, Retriever, conflict_key, to_tensors
from soma.memory.sqlite_store import SUMMARY_CAP, MemoryStore, path_within
from soma.memory.vector_index import VectorIndex

__all__ = [
    "MemoryStore", "SUMMARY_CAP", "path_within",
    "VectorIndex",
    "TextEmbedder", "ImageEmbedder",
    "Ingestor", "IngestReport", "item_id_for", "redact",
    "Retriever", "RerankWeights", "conflict_key", "to_tensors",
    "forget", "forget_source", "purge_expired", "verify_deleted",
    "export_memories", "list_for_viewer", "correct",
]
