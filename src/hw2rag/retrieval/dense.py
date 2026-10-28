"""Sentence-transformer embeddings stored in an exact FAISS cosine index."""

from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from typing import Any, Iterable
import logging

import numpy as np

from hw2rag.retrieval.ollama_embedding import OllamaEmbedding, QWEN_QUERY_PROMPT


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _installed_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependencies
        raise RuntimeError("FAISS is required; install the retrieval extra") from exc
    return faiss


def _sentence_transformer_class() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependencies
        raise RuntimeError("sentence-transformers is required; install the retrieval extra") from exc
    return SentenceTransformer


@dataclass(slots=True)
class DenseIndex:
    index: Any
    model_name: str
    model_revision: str | None = None
    device: str = "cpu"
    batch_size: int = 64
    local_files_only: bool = True
    backend: str = "sentence_transformers"
    base_url: str = "http://127.0.0.1:11434"
    backend_model: str | None = None
    model_digest: str | None = None
    query_prompt: str = ""
    reused_embeddings: int = 0
    _model: Any = field(default=None, repr=False)

    @property
    def document_count(self) -> int:
        return int(self.index.ntotal)

    @property
    def dimension(self) -> int:
        return int(self.index.d)

    def model(self) -> Any:
        if self._model is None:
            if self.backend == "ollama":
                self._model = OllamaEmbedding(
                    model=self.backend_model or self.model_name,
                    base_url=self.base_url,
                    expected_digest=self.model_digest,
                )
                return self._model
            sentence_transformer = _sentence_transformer_class()
            self._model = sentence_transformer(
                self.model_name,
                device=self.device,
                revision=self.model_revision,
                local_files_only=self.local_files_only,
            )
        return self._model

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        model_revision: str | None = DEFAULT_MODEL_REVISION,
        device: str = "cpu",
        batch_size: int = 64,
        show_progress: bool = True,
        backend: str = "sentence_transformers",
        base_url: str = "http://127.0.0.1:11434",
        backend_model: str | None = None,
        query_prompt: str | None = None,
        checkpoint_dir: Path | None = None,
        reuse_index: "DenseIndex | None" = None,
        reuse_texts: Iterable[str] | None = None,
    ) -> "DenseIndex":
        text_list = list(texts)
        if backend == "ollama":
            urls = [url.strip() for url in base_url.split(",") if url.strip()]
            model = OllamaEmbedding(model=backend_model or model_name, base_url=urls[0])
            encoders = [model] + [
                OllamaEmbedding(model=backend_model or model_name, base_url=url, expected_digest=model.digest)
                for url in urls[1:]
            ]
            base_url = urls[0]
            reuse_rows: dict[str, int] = {}
            if reuse_index is not None:
                if reuse_index.backend != "ollama" or reuse_index.model_digest != model.digest:
                    raise ValueError("reused vectors require the exact same embedding model digest and backend")
                previous_texts = list(reuse_texts or [])
                if len(previous_texts) != reuse_index.document_count:
                    raise ValueError("reused texts must align exactly with the previous vector rows")
                reuse_rows = {text: row for row, text in enumerate(previous_texts)}
            reused_count = sum(text in reuse_rows for text in text_list)
            if reused_count:
                logging.getLogger(__name__).info("reusing %s/%s unchanged document embeddings", reused_count, len(text_list))
            checkpoint_key = sha256(json.dumps([model.digest, text_list], ensure_ascii=False).encode()).hexdigest()
            checkpoint = checkpoint_dir / checkpoint_key if checkpoint_dir is not None else None
            if checkpoint is not None:
                checkpoint.mkdir(parents=True, exist_ok=True)
            def embed_batch(item: tuple[int, int]) -> np.ndarray:
                batch_index, start = item
                path = checkpoint / f"{start:07d}_{batch_size}.npy" if checkpoint is not None else None
                if path is not None and path.exists():
                    batch = np.load(path, allow_pickle=False)
                else:
                    texts_in_batch = text_list[start : start + batch_size]
                    missing_positions = [i for i, text in enumerate(texts_in_batch) if text not in reuse_rows]
                    encoder = encoders[batch_index % len(encoders)]
                    if reuse_index is None:
                        batch = encoder.encode(texts_in_batch)
                    else:
                        batch = np.empty((len(texts_in_batch), reuse_index.dimension), dtype=np.float32)
                        for position, text in enumerate(texts_in_batch):
                            if text in reuse_rows:
                                batch[position] = reuse_index.index.reconstruct(reuse_rows[text])
                        if missing_positions:
                            batch[missing_positions] = encoder.encode([texts_in_batch[i] for i in missing_positions])
                    if path is not None:
                        temporary = path.with_suffix(".tmp.npy")
                        np.save(temporary, batch)
                        temporary.replace(path)
                if batch.ndim != 2 or batch.shape[0] != len(text_list[start : start + batch_size]) or not np.isfinite(batch).all():
                    raise ValueError("invalid cached embedding batch")
                if show_progress and (start // batch_size % 10 == 0 or start + batch_size >= len(text_list)):
                    logging.getLogger(__name__).info("embedded %s/%s chunks", min(start + batch_size, len(text_list)), len(text_list))
                return batch
            jobs = list(enumerate(range(0, len(text_list), batch_size)))
            with ThreadPoolExecutor(max_workers=len(encoders)) as executor:
                try:
                    batches = list(executor.map(embed_batch, jobs))
                except BaseException:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
            embeddings = np.concatenate(batches)
        elif backend == "sentence_transformers":
            sentence_transformer = _sentence_transformer_class()
            model = sentence_transformer(model_name, device=device, revision=model_revision)
            embeddings = model.encode(
                text_list,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        else:
            raise ValueError(f"unsupported embedding backend: {backend}")
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"expected a 2D embedding matrix, got shape {vectors.shape}")
        faiss = _faiss()
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(
            index=index,
            model_name=model_name,
            model_revision=model_revision,
            device=device,
            batch_size=batch_size,
            local_files_only=True,
            backend=backend,
            base_url=base_url,
            backend_model=backend_model,
            model_digest=model.digest if backend == "ollama" else None,
            query_prompt=(QWEN_QUERY_PROMPT if "Qwen3-Embedding" in model_name else "") if query_prompt is None else query_prompt,
            reused_embeddings=reused_count if backend == "ollama" else 0,
            _model=model,
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        return self.search_many([query], top_k=top_k)[0]

    def search_many(
        self,
        queries: Iterable[str],
        top_k: int = 10,
    ) -> list[list[tuple[int, float]]]:
        query_list = list(queries)
        if not query_list:
            return []
        if top_k <= 0 or not self.document_count:
            return [[] for _ in query_list]
        embeddings = self.model().encode(
            [self.query_prompt + query for query in query_list],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_vectors = np.asarray(embeddings, dtype=np.float32)
        scores, indices = self.index.search(query_vectors, min(top_k, self.document_count))
        return [
            [
                (int(index), float(score))
                for index, score in zip(row_indices, row_scores, strict=True)
                if int(index) >= 0
            ]
            for row_indices, row_scores in zip(indices, scores, strict=True)
        ]

    def save(self, directory: Path) -> dict[str, object]:
        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / "dense.faiss"
        _faiss().write_index(self.index, str(index_path))
        model = self.model()
        return {
            "algorithm": "faiss_index_flat_ip",
            "backend": self.backend,
            "base_url": self.base_url if self.backend == "ollama" else None,
            "backend_model": self.backend_model,
            "model_digest": self.model_digest,
            "query_prompt": self.query_prompt,
            "quantization": model.details.get("quantization_level") if self.backend == "ollama" else None,
            "batch_size": self.batch_size,
            "dimension": self.dimension,
            "document_count": self.document_count,
            "embedding_model": self.model_name,
            "embedding_model_revision": self.model_revision,
            "max_sequence_length": 4096 if self.backend == "ollama" else int(model.max_seq_length),
            "normalize_embeddings": True,
            "reused_unchanged_embeddings": self.reused_embeddings,
            "packages": {
                "faiss-cpu": version("faiss-cpu"),
                # Not needed (or installed) for the Ollama backend; record its absence instead of failing.
                "sentence-transformers": _installed_version("sentence-transformers"),
            },
            "files": {
                index_path.name: sha256(index_path.read_bytes()).hexdigest(),
            },
        }

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        model_name: str,
        model_revision: str | None = None,
        device: str = "cpu",
        batch_size: int = 64,
        backend: str = "sentence_transformers",
        base_url: str = "http://127.0.0.1:11434",
        backend_model: str | None = None,
        model_digest: str | None = None,
        query_prompt: str = "",
    ) -> "DenseIndex":
        index = _faiss().read_index(str(directory / "dense.faiss"))
        return cls(
            index=index,
            model_name=model_name,
            model_revision=model_revision,
            device=device,
            batch_size=batch_size,
            local_files_only=True,
            backend=backend,
            base_url=base_url,
            backend_model=backend_model,
            model_digest=model_digest,
            query_prompt=query_prompt,
        )
