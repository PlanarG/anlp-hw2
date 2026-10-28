"""Build retrieval chunks from normalized document shards."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import logging
from pathlib import Path
from typing import Any

from hw2rag.utils import atomic_write, utc_now, write_json
from hw2rag.indexing.chunk import (
    CHUNK_JSON_SCHEMA,
    CHUNK_SCHEMA_VERSION,
    DEFAULT_MAX_WORDS,
    DEFAULT_OVERLAP_WORDS,
    Chunk,
    chunk_document,
)


LOGGER = logging.getLogger(__name__)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_chunks(path: Path, chunks: list[Chunk]) -> dict[str, Any]:
    identifiers = [chunk.chunk_id for chunk in chunks]
    duplicates = sorted(identifier for identifier, count in Counter(identifiers).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate chunk IDs ({len(duplicates)}): {', '.join(duplicates[:5])}")
    chunks.sort(key=lambda item: item.chunk_id)
    content = "".join(json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for chunk in chunks)
    atomic_write(path, content)
    encoded = content.encode("utf-8")
    word_counts = [chunk.word_count for chunk in chunks]
    per_document = Counter(chunk.document_id for chunk in chunks)
    return {
        "chunk_count": len(chunks),
        "documents_split": sum(1 for count in per_document.values() if count > 1),
        "max_chunk_words": max(word_counts, default=0),
        "mean_chunk_words": round(sum(word_counts) / len(word_counts), 2) if word_counts else 0,
        "min_chunk_words": min(word_counts, default=0),
        "output_path": path.name,
        "output_sha256": sha256(encoded).hexdigest(),
        "output_size_bytes": len(encoded),
    }


def build_source(
    *,
    root: Path,
    input_dir: Path,
    output_dir: Path,
    source: dict[str, Any],
    max_words: int,
    overlap_words: int,
) -> dict[str, Any]:
    source_id = str(source["source_id"])
    input_path = input_dir / str(source["output_path"])
    expected_hash = str(source["output_sha256"])
    actual_hash = file_sha256(input_path)
    if actual_hash != expected_hash:
        raise ValueError(f"{input_path}: hash differs from document manifest")
    document_path = str(input_path.relative_to(root))
    documents = read_jsonl(input_path)
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(
            document,
            document_path=document_path,
            max_words=max_words,
            overlap_words=overlap_words,
        )
    ]
    stats = write_chunks(output_dir / f"{source_id}.jsonl", chunks)
    stats.update(
        {
            "category": str(source.get("category") or "unknown"),
            "document_count": len(documents),
            "input_document_path": document_path,
            "input_document_sha256": actual_hash,
            "source_id": source_id,
        }
    )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="project root")
    parser.add_argument(
        "--document-manifest",
        type=Path,
        default=Path("data/corpus/documents/_manifest.json"),
        help="document manifest, relative to root by default",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/corpus/chunks"),
        help="chunk output directory, relative to root by default",
    )
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--overlap-words", type=int, default=DEFAULT_OVERLAP_WORDS)
    parser.add_argument("--only", action="append", help="build only this source ID; may be repeated")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.max_words <= 0 or args.overlap_words < 0 or args.overlap_words >= args.max_words:
        LOGGER.error("require max_words > overlap_words >= 0")
        return 2

    root = args.root.resolve()
    manifest_path = args.document_manifest if args.document_manifest.is_absolute() else root / args.document_manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    input_dir = manifest_path.parent
    document_manifest = read_json(manifest_path)
    sources = {str(source["source_id"]): source for source in document_manifest.get("sources", [])}
    requested = set(args.only or sources)
    unknown = requested - set(sources)
    if unknown:
        LOGGER.error("no document shard for: %s", ", ".join(sorted(unknown)))
        return 2

    write_json(output_dir / "_schema.json", CHUNK_JSON_SCHEMA)
    errors: list[dict[str, Any]] = []
    source_results: list[dict[str, Any]] = []
    for source_id in sorted(requested):
        try:
            result = build_source(
                root=root,
                input_dir=input_dir,
                output_dir=output_dir,
                source=sources[source_id],
                max_words=args.max_words,
                overlap_words=args.overlap_words,
            )
            source_results.append(result)
            LOGGER.info("built %s: %s chunks", source_id, result["chunk_count"])
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("failed to chunk %s", source_id)
            errors.append({"built_at": utc_now(), "error": str(exc), "source_id": source_id})

    error_content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in errors)
    atomic_write(output_dir / "_errors.jsonl", error_content)
    manifest = {
        "build_completed_at": utc_now(),
        "chunk_count": sum(int(row["chunk_count"]) for row in source_results),
        "chunking": {
            "strategy": "paragraph_sentence_words_v1",
            "max_words": args.max_words,
            "overlap_words": args.overlap_words,
        },
        "document_count": sum(int(row["document_count"]) for row in source_results),
        "error_count": len(errors),
        "input_document_manifest": str(manifest_path.relative_to(root)),
        "input_document_manifest_sha256": file_sha256(manifest_path),
        "schema_version": CHUNK_SCHEMA_VERSION,
        "source_count": len(source_results),
        "sources": source_results,
    }
    write_json(output_dir / "_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
