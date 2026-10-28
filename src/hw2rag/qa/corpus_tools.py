"""Bounded, read-only search/grep/read tools over verified knowledge documents."""

from __future__ import annotations

from hashlib import sha256
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any

from hw2rag.retrieval.search import Retriever


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }
    for name, description, properties, required in [
        (
            "search",
            "Search knowledge chunks with a rewritten query. Use distinctive entities, dates, and synonyms. Returns document IDs and offsets for reading the full document.",
            {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 6},
            },
            ["query"],
        ),
        (
            "grep_documents",
            "Case-insensitive literal grep over ALL full knowledge documents, including text outside initial chunks. Try a short exact name or distinctive keyword; spaces mean an exact phrase. Returns snippets, document IDs, and offsets.",
            {
                "pattern": {"type": "string"},
                "source_id": {"type": "string", "description": "Optional source ID to narrow the search."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 8},
            },
            ["pattern"],
        ),
        (
            "read_document",
            "Read a full knowledge document in pages by its exact document_id. Use an offset from search/grep, or 0 for the beginning. Can continue at next_offset.",
            {
                "document_id": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "max_chars": {"type": "integer", "minimum": 100, "maximum": 6000},
            },
            ["document_id"],
        ),
    ]
]


class CorpusTools:
    def __init__(self, root: Path, retriever: Retriever, *, mode: str = "hybrid", search_options: dict[str, Any] | None = None) -> None:
        self.retriever = retriever
        self.mode = mode
        self.search_options = dict(search_options or {})
        self.schemas = deepcopy(TOOL_SCHEMAS)
        self.schemas[1]["function"]["parameters"]["properties"]["source_id"]["enum"] = ["", *retriever.corpus.source_ids]
        self.documents: dict[str, dict[str, Any]] = {}
        self.text_paths: dict[Path, str] = {}
        self.root = root.resolve()
        # Resolve the documents belonging to this exact chunk snapshot, not a
        # mutable project-wide default. This keeps archived indexes usable.
        chunk_manifest_path = getattr(retriever.corpus, "manifest_path", None)
        document_manifest_path = self.root / "data/corpus/documents/_manifest.json"
        expected_document_manifest_hash = None
        if chunk_manifest_path is not None:
            chunk_manifest = json.loads(Path(chunk_manifest_path).read_text())
            if chunk_manifest.get("input_document_manifest"):
                document_manifest_path = (self.root / chunk_manifest["input_document_manifest"]).resolve()
                expected_document_manifest_hash = chunk_manifest.get("input_document_manifest_sha256")
        if not document_manifest_path.resolve().is_relative_to(self.root / "data/corpus"):
            raise ValueError("document manifest escapes the corpus directory")
        manifest_bytes = document_manifest_path.read_bytes()
        if expected_document_manifest_hash and sha256(manifest_bytes).hexdigest() != expected_document_manifest_hash:
            raise ValueError("document manifest differs from the indexed snapshot")
        directory = document_manifest_path.parent
        manifest = json.loads(manifest_bytes)
        selected = set(retriever.corpus.source_ids)
        expected = {chunk["document_id"]: chunk["document_version_id"] for chunk in retriever.corpus.chunks}
        for source in manifest["sources"]:
            if source["source_id"] not in selected:
                continue
            path = (directory / source["output_path"]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("document shard escapes knowledge directory")
            content = path.read_bytes()
            if sha256(content).hexdigest() != source["output_sha256"]:
                raise ValueError(f"document shard hash mismatch: {path}")
            for line in content.decode().splitlines():
                document = json.loads(line)
                doc_id = document["document_id"]
                if doc_id not in expected:
                    continue
                if document["version_id"] != expected[doc_id]:
                    raise ValueError(f"document version differs from chunk input: {doc_id}")
                self.documents[doc_id] = document
        if set(self.documents) != set(expected):
            raise ValueError("some indexed documents are missing from the verified document shards")
        corpus_hash = retriever.corpus.manifest_sha256[:20]
        self.text_dir = self.root / "artifacts/corpus_text" / corpus_hash
        for doc_id, document in self.documents.items():
            source_id = document["source_id"]
            if not source_id.replace("_", "").replace("-", "").isalnum():
                raise ValueError("unsafe source ID")
            path = self.text_dir / source_id / (sha256(doc_id.encode()).hexdigest() + ".txt")
            text = self._text(document)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                path.write_text(text, encoding="utf-8")
            self.text_paths[path.resolve()] = doc_id

    @staticmethod
    def _text(document: dict[str, Any]) -> str:
        return f"Title: {document['title']}\n{document['text']}"

    @staticmethod
    def _metadata(document: dict[str, Any]) -> dict[str, Any]:
        metadata = {key: document.get(key) for key in ("document_id", "source_id", "title", "canonical_url")}
        attributes = document.get("attributes") or {}
        flags = list(attributes.get("quality_flags") or []) + list(attributes.get("description_quality_flags") or [])
        if flags:
            metadata["source_quality_warnings"] = flags
        return metadata

    def search(self, query: str, top_k: int = 4) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ValueError("query must contain 1–1000 characters")
        results = self.retriever.search(query, mode=self.mode, top_k=min(max(int(top_k), 1), 6), **self.search_options)
        return {
            "results": [
                {
                    **self._metadata(result["chunk"]),
                    "chunk_id": result["chunk"]["chunk_id"],
                    "offset": max(0, int(result["chunk"].get("start_char", 0)) - 200),
                    "text": result["chunk"]["retrieval_text"][:2200],
                }
                for result in results
            ]
        }

    def grep_documents(self, pattern: str, source_id: str = "", limit: int = 5) -> dict[str, Any]:
        if not isinstance(pattern, str) or not pattern.strip() or len(pattern) > 200 or "\n" in pattern:
            raise ValueError("grep pattern must be a nonempty literal phrase of at most 200 characters")
        if source_id and source_id not in self.retriever.corpus.source_ids:
            raise ValueError("unknown source_id")
        limit = min(max(int(limit), 1), 8)
        directory = self.text_dir / source_id if source_id else self.text_dir
        process = subprocess.run(
            ["rg", "--json", "--ignore-case", "--fixed-strings", "--max-count", "1", "--", pattern, str(directory)],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if process.returncode not in (0, 1):
            raise RuntimeError("knowledge grep failed")
        matches = []
        for line in process.stdout.splitlines():
            event = json.loads(line)
            if event["type"] != "match":
                continue
            doc_id = self.text_paths.get(Path(event["data"]["path"]["text"]).resolve())
            if doc_id is None:
                continue
            document = self.documents[doc_id]
            text = self._text(document)
            position = max(0, text.casefold().find(pattern.casefold()))
            start = max(0, position - 200)
            matches.append({**self._metadata(document), "offset": start, "snippet": text[start : start + 1000]})
        matches.sort(key=lambda item: str(item["document_id"]))
        return {"match_count": len(matches), "results": matches[:limit], "truncated": len(matches) > limit}

    def read_document(self, document_id: str, offset: int = 0, max_chars: int = 5000) -> dict[str, Any]:
        if document_id not in self.documents:
            raise ValueError("unknown document_id; use search or grep to obtain an exact ID")
        offset = max(int(offset), 0)
        max_chars = min(max(int(max_chars), 100), 6000)
        document = self.documents[document_id]
        text = self._text(document)
        end = min(len(text), offset + max_chars)
        return {
            **self._metadata(document),
            "offset": offset,
            "text": text[offset:end],
            "total_chars": len(text),
            "next_offset": end if end < len(text) else None,
        }

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if name not in {"search", "grep_documents", "read_document"}:
                raise ValueError("unknown tool")
            return getattr(self, name)(**arguments)
        except (ValueError, TypeError, RuntimeError, subprocess.TimeoutExpired) as exc:
            return {"error": str(exc)}
