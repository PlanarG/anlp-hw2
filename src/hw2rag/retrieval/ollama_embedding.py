"""Local Qwen embeddings with explicit query instructions and model identity."""

from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np


QWEN_MODEL = "Qwen/Qwen3-Embedding-8B"
QWEN_OLLAMA_TAG = "qwen3-embedding:8b"
QWEN_QUERY_PROMPT = (
    "Instruct: Given a question about Pittsburgh or Carnegie Mellon University, "
    "retrieve relevant passages that answer the question\nQuery: "
)
DEFAULT_INDEX_DIR = "artifacts/indexes/qwen3_8b"


def request_json(base_url: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    request = Request(
        base_url.rstrip("/") + endpoint,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=600) as response:
            result = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama {endpoint} HTTP {exc.code}: {detail}") from exc
    if not isinstance(result, dict) or result.get("error"):
        raise RuntimeError(f"Ollama {endpoint}: {result}")
    return result


class OllamaEmbedding:
    def __init__(
        self,
        model: str = QWEN_OLLAMA_TAG,
        base_url: str = "http://127.0.0.1:11434",
        expected_digest: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        matches = [item for item in request_json(base_url, "/api/tags")["models"] if item["name"] == model]
        if not matches:
            raise ValueError(f"Ollama model {model!r} is not installed")
        self.digest = str(matches[0]["digest"])
        if expected_digest is not None and self.digest != expected_digest:
            raise ValueError("embedding model digest has changed; restore the model or rebuild the index")
        self.details = matches[0].get("details", {})

    def encode(self, texts: list[str], *, batch_size: int = 16, **_: Any) -> np.ndarray:
        if len(texts) > batch_size:
            return np.concatenate([
                self.encode(texts[start : start + batch_size], batch_size=batch_size)
                for start in range(0, len(texts), batch_size)
            ])
        response = request_json(
            self.base_url,
            "/api/embed",
            {
                "model": self.model,
                "input": texts,
                "truncate": False,
                "keep_alive": "30m",
                "options": {"num_ctx": 4096},
            },
        )
        vectors = np.asarray(response["embeddings"], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
            raise ValueError(f"invalid embedding response shape/values: {vectors.shape}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError("embedding response contains zero vectors")
        return vectors / norms
