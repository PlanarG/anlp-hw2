"""Deterministic, metadata-preserving document chunking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from hashlib import sha256
import json
import re
from typing import Any

_WHITESPACE = re.compile(r"[ \t\r\f\v]+")


def normalize_text(value: str) -> str:
    """Normalize whitespace without destroying paragraph boundaries."""

    lines = [_WHITESPACE.sub(" ", line).strip() for line in value.splitlines()]
    output: list[str] = []
    previous_blank = False
    for line in lines:
        if line:
            output.append(line)
            previous_blank = False
        elif output and not previous_blank:
            output.append("")
            previous_blank = True
    while output and not output[-1]:
        output.pop()
    return "\n".join(output)


CHUNK_SCHEMA_VERSION = 1
DEFAULT_MAX_WORDS = 180
DEFAULT_OVERLAP_WORDS = 30
_WORD = re.compile(r"\S+")
_PARAGRAPH = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
_SENTENCE_END = re.compile(r"[.!?](?:[\"'”’)]*)?(?=\s|$)")


@dataclass(frozen=True, slots=True)
class TextSpan:
    start: int
    end: int
    word_count: int


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    version_id: str
    document_id: str
    document_version_id: str
    source_id: str
    category: str
    document_type: str
    title: str
    text: str
    retrieval_text: str
    chunk_index: int
    chunk_count: int
    start_char: int
    end_char: int
    word_count: int
    canonical_url: str | None
    temporal: dict[str, Any]
    attributes: dict[str, Any]
    provenance: dict[str, Any]
    chunking: dict[str, Any]
    language: str = "en"
    text_format: str = "markdown"
    schema_version: int = CHUNK_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def count_words(text: str) -> int:
    return sum(1 for _ in _WORD.finditer(text))


def _display_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(filter(None, (_display_value(item) for item in value)))
    if isinstance(value, dict):
        return ", ".join(f"{key}: {_display_value(item)}" for key, item in value.items() if item not in (None, "", []))
    return "" if value is None else str(value)


def retrieval_prefix(document: dict[str, Any]) -> str:
    title = str(document.get("title") or "")
    document_type = str(document.get("document_type") or "")
    temporal = dict(document.get("temporal") or {})
    attributes = dict(document.get("attributes") or {})
    fields: list[tuple[str, Any]] = []
    if document_type in {"event", "game"}:
        local_date = temporal.get("local_date")
        if local_date:
            try:
                parsed_date = date.fromisoformat(str(local_date))
                display_date = parsed_date.strftime("%B %d, %Y").replace(" 0", " ")
                local_date = f"{display_date} ({local_date})"
            except ValueError:
                pass
            fields.extend([
                ("Local calendar date", local_date),
                ("Local start", temporal.get("local_start")),
                ("Time zone", temporal.get("timezone")),
                ("UTC start", temporal.get("utc_start")),
            ])
        fields.extend(
            [
                ("Start", None if local_date else temporal.get("event_start")),
                ("End", temporal.get("event_end")),
                ("Home team", attributes.get("home_team")),
                ("Away team", attributes.get("away_team")),
                ("Venue", attributes.get("venue")),
                ("Team", attributes.get("team")),
                ("Sport", attributes.get("sport")),
                ("Opponent", attributes.get("opponent")),
            ]
        )
    elif document_type == "person":
        fields.extend(
            [
                ("Team", attributes.get("team")),
                ("Number", attributes.get("jersey_number") or attributes.get("sweater_number")),
                ("Position", attributes.get("position")),
            ]
        )
    elif document_type == "artwork":
        fields.extend(
            [
                ("Artist", attributes.get("artists")),
                ("Accession", attributes.get("accession_number")),
                ("Date", attributes.get("creation_date")),
            ]
        )
    elif document_type == "pdf_page":
        fields.extend(
            [
                ("Document", attributes.get("file_title")),
                ("Document heading", attributes.get("document_heading")),
                ("Page", attributes.get("page_number")),
                ("Section", attributes.get("section_heading")),
                ("Page opening context", attributes.get("page_opening_context")),
            ]
        )
    lines = [title] if title else []
    lines.extend(f"{label}: {_display_value(value)}" for label, value in fields if _display_value(value))
    return normalize_text("\n".join(lines))


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _word_spans(text: str, start: int, end: int, max_words: int) -> list[TextSpan]:
    words = list(_WORD.finditer(text, start, end))
    if not words:
        return []
    part_count = (len(words) + max_words - 1) // max_words
    base_size, larger_parts = divmod(len(words), part_count)
    spans: list[TextSpan] = []
    cursor = 0
    for part_index in range(part_count):
        size = base_size + (1 if part_index < larger_parts else 0)
        group = words[cursor : cursor + size]
        spans.append(TextSpan(group[0].start(), group[-1].end(), len(group)))
        cursor += size
    return spans


def _sentence_spans(text: str, start: int, end: int, max_words: int) -> list[TextSpan]:
    spans: list[TextSpan] = []
    cursor = start
    boundaries = [match.end() for match in _SENTENCE_END.finditer(text, start, end)]
    if not boundaries or boundaries[-1] < end:
        boundaries.append(end)
    for boundary in boundaries:
        sentence_start, sentence_end = _trim_span(text, cursor, boundary)
        cursor = boundary
        if sentence_start >= sentence_end:
            continue
        words = count_words(text[sentence_start:sentence_end])
        if words <= max_words:
            spans.append(TextSpan(sentence_start, sentence_end, words))
        else:
            spans.extend(_word_spans(text, sentence_start, sentence_end, max_words))
    return spans


def atomic_spans(text: str, max_words: int) -> list[TextSpan]:
    """Return paragraph-first spans, splitting oversized paragraphs at sentences."""

    spans: list[TextSpan] = []
    for match in _PARAGRAPH.finditer(text):
        start, end = _trim_span(text, match.start(), match.end())
        words = count_words(text[start:end])
        if not words:
            continue
        spans.extend(_sentence_spans(text, start, end, max_words))
    if not spans and text.strip():
        start, end = _trim_span(text, 0, len(text))
        spans.extend(_word_spans(text, start, end, max_words))
    return spans


def chunk_spans(text: str, max_words: int, overlap_words: int) -> list[TextSpan]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be non-negative and smaller than max_words")
    total_words = count_words(text)
    if total_words <= max_words:
        start, end = _trim_span(text, 0, len(text))
        return [TextSpan(start, end, total_words)] if start < end else []

    atoms = atomic_spans(text, max_words)
    chunks: list[TextSpan] = []
    cursor = 0
    while cursor < len(atoms):
        end_index = cursor
        words = 0
        while end_index < len(atoms):
            candidate = atoms[end_index].word_count
            if end_index > cursor and words + candidate > max_words:
                break
            words += candidate
            end_index += 1
            if words >= max_words:
                break
        start_char, end_char = _trim_span(text, atoms[cursor].start, atoms[end_index - 1].end)
        chunks.append(TextSpan(start_char, end_char, count_words(text[start_char:end_char])))
        if end_index >= len(atoms):
            break

        next_cursor = end_index
        overlap = 0
        while next_cursor > cursor + 1:
            candidate = atoms[next_cursor - 1].word_count
            if overlap + candidate > overlap_words:
                break
            overlap += candidate
            next_cursor -= 1
        cursor = next_cursor
    return chunks


def chunk_document(
    document: dict[str, Any],
    *,
    document_path: str,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    text = str(document.get("text") or "")
    title = str(document.get("title") or "")
    document_id = str(document.get("document_id") or "")
    document_version_id = str(document.get("version_id") or "")
    if not document_id or not document_version_id or not text.strip():
        raise ValueError("document_id, version_id, and non-empty text are required")

    spans = chunk_spans(text, max_words=max_words, overlap_words=overlap_words)
    prefix = retrieval_prefix(document)
    chunks: list[Chunk] = []
    for index, span in enumerate(spans):
        chunk_text = text[span.start : span.end]
        retrieval_text = normalize_text(f"{prefix}\n\n{chunk_text}") if prefix else chunk_text
        version_payload = {
            "document_version_id": document_version_id,
            "chunk_index": index,
            "text": chunk_text,
            "retrieval_text": retrieval_text,
            "max_words": max_words,
            "overlap_words": overlap_words,
        }
        encoded = json.dumps(
            version_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}:chunk:{index:04d}",
                version_id=f"sha256:{sha256(encoded).hexdigest()}",
                document_id=document_id,
                document_version_id=document_version_id,
                source_id=str(document.get("source_id") or ""),
                category=str(document.get("category") or ""),
                document_type=str(document.get("document_type") or ""),
                title=title,
                text=chunk_text,
                retrieval_text=retrieval_text,
                chunk_index=index,
                chunk_count=len(spans),
                start_char=span.start,
                end_char=span.end,
                word_count=span.word_count,
                canonical_url=document.get("canonical_url"),
                temporal=dict(document.get("temporal") or {}),
                attributes=dict(document.get("attributes") or {}),
                provenance={
                    "document_path": document_path,
                    "document_id": document_id,
                    "document_version_id": document_version_id,
                    "raw": dict(document.get("provenance") or {}),
                },
                chunking={
                    "strategy": "paragraph_sentence_words_v1",
                    "max_words": max_words,
                    "overlap_words": overlap_words,
                },
                language=str(document.get("language") or "en"),
                text_format=str(document.get("text_format") or "markdown"),
            )
        )
    return chunks


CHUNK_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "HW2 retrieval chunk",
    "type": "object",
    "required": [
        "schema_version",
        "chunk_id",
        "version_id",
        "document_id",
        "document_version_id",
        "source_id",
        "category",
        "document_type",
        "title",
        "text",
        "retrieval_text",
        "chunk_index",
        "chunk_count",
        "start_char",
        "end_char",
        "word_count",
        "temporal",
        "attributes",
        "provenance",
        "chunking",
    ],
    "properties": {
        "schema_version": {"const": CHUNK_SCHEMA_VERSION},
        "chunk_id": {"type": "string", "minLength": 1},
        "version_id": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
        "document_id": {"type": "string", "minLength": 1},
        "document_version_id": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
        "source_id": {"type": "string", "minLength": 1},
        "category": {"type": "string", "minLength": 1},
        "document_type": {"type": "string", "minLength": 1},
        "title": {"type": "string"},
        "text": {"type": "string", "minLength": 1},
        "retrieval_text": {"type": "string", "minLength": 1},
        "chunk_index": {"type": "integer", "minimum": 0},
        "chunk_count": {"type": "integer", "minimum": 1},
        "start_char": {"type": "integer", "minimum": 0},
        "end_char": {"type": "integer", "minimum": 1},
        "word_count": {"type": "integer", "minimum": 1},
        "canonical_url": {"type": ["string", "null"]},
        "temporal": {"type": "object"},
        "attributes": {"type": "object"},
        "provenance": {"type": "object"},
        "chunking": {"type": "object"},
        "language": {"type": "string", "minLength": 2},
        "text_format": {"enum": ["markdown", "plain"]},
    },
}
