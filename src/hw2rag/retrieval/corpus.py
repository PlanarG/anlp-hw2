"""Load and verify the retrieval chunk corpus."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


@dataclass(slots=True)
class ChunkCorpus:
    chunks: list[dict[str, Any]]
    manifest_path: Path
    manifest_sha256: str
    source_ids: tuple[str, ...]

    @property
    def row_ids(self) -> list[str]:
        return [str(chunk["chunk_id"]) for chunk in self.chunks]

    @property
    def texts(self) -> list[str]:
        return [str(chunk["retrieval_text"]) for chunk in self.chunks]

    @classmethod
    def load(
        cls,
        manifest_path: Path,
        *,
        source_ids: Iterable[str] | None = None,
        verify_hashes: bool = True,
    ) -> "ChunkCorpus":
        manifest_path = manifest_path.resolve()
        manifest = read_json(manifest_path)
        if int(manifest.get("error_count", 0)):
            raise ValueError(f"{manifest_path}: chunk build contains errors")
        available = {
            str(source["source_id"]): source
            for source in manifest.get("sources", [])
            if isinstance(source, dict) and source.get("source_id")
        }
        selected = set(source_ids or available)
        unknown = selected - set(available)
        if unknown:
            raise ValueError(f"unknown chunk sources: {', '.join(sorted(unknown))}")

        chunks: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for source_id in sorted(selected):
            source = available[source_id]
            path = manifest_path.parent / str(source["output_path"])
            if verify_hashes and file_sha256(path) != str(source["output_sha256"]):
                raise ValueError(f"{path}: hash differs from chunk manifest")
            source_count = 0
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(chunk, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                chunk_id = str(chunk.get("chunk_id") or "")
                if not chunk_id:
                    raise ValueError(f"{path}:{line_number}: missing chunk_id")
                if chunk_id in seen_ids:
                    raise ValueError(f"duplicate chunk_id: {chunk_id}")
                if chunk.get("source_id") != source_id:
                    raise ValueError(f"{path}:{line_number}: source_id mismatch")
                if not chunk.get("retrieval_text"):
                    raise ValueError(f"{path}:{line_number}: missing retrieval_text")
                seen_ids.add(chunk_id)
                chunks.append(chunk)
                source_count += 1
            if source_count != int(source["chunk_count"]):
                raise ValueError(
                    f"{path}: expected {source['chunk_count']} chunks, found {source_count}"
                )
        return cls(
            chunks=chunks,
            manifest_path=manifest_path,
            manifest_sha256=file_sha256(manifest_path),
            source_ids=tuple(sorted(selected)),
        )
