"""Question answering over retrieved HW2 evidence."""

from hw2rag.qa.prompt import EvidenceContext, build_evidence_context
from hw2rag.qa.reader import ExtractiveReader, OllamaReader, ReaderResponse


__all__ = [
    "EvidenceContext",
    "ExtractiveReader",
    "OllamaReader",
    "ReaderResponse",
    "build_evidence_context",
]
