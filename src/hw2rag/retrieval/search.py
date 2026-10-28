"""Query sparse, dense, or hybrid retrieval indexes."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from hw2rag.retrieval.bm25 import BM25Index
from hw2rag.retrieval.corpus import ChunkCorpus, read_json
from hw2rag.retrieval.dense import DenseIndex
from hw2rag.retrieval.ollama_embedding import DEFAULT_INDEX_DIR
from hw2rag.retrieval.hybrid import (
    FusedCandidate,
    normalized_weighted_fusion,
    reciprocal_rank_fusion,
)


DEFAULT_FUSION = "weighted"
DEFAULT_SPARSE_WEIGHT = 2.0


@dataclass(slots=True)
class Retriever:
    corpus: ChunkCorpus
    sparse: BM25Index | None
    dense: DenseIndex | None
    _focused: Any = field(default=None, repr=False)
    _fields: Any = field(default=None, repr=False)
    _safe: Any = field(default=None, repr=False)

    @classmethod
    def load(
        cls,
        *,
        root: Path,
        index_dir: Path,
        device: str = "cpu",
    ) -> "Retriever":
        root = root.resolve()
        index_dir = index_dir.resolve()
        manifest = read_json(index_dir / "manifest.json")
        chunk_manifest_value = Path(str(manifest["chunk_manifest"]))
        chunk_manifest = (
            chunk_manifest_value
            if chunk_manifest_value.is_absolute()
            else root / chunk_manifest_value
        )
        corpus = ChunkCorpus.load(chunk_manifest, source_ids=manifest["sources"])
        if corpus.manifest_sha256 != manifest["chunk_manifest_sha256"]:
            raise ValueError("chunk manifest has changed; rebuild the retrieval indexes")

        row_ids_info = manifest["row_ids"]
        row_ids_path = index_dir / str(row_ids_info["path"])
        if sha256(row_ids_path.read_bytes()).hexdigest() != row_ids_info["sha256"]:
            raise ValueError("row_ids.json hash mismatch")
        row_ids = json.loads(row_ids_path.read_text(encoding="utf-8"))
        if row_ids != corpus.row_ids:
            raise ValueError("chunk row order differs from the indexed row order")

        sparse_info = manifest.get("sparse")
        sparse = None
        if isinstance(sparse_info, dict):
            cls._verify_files(index_dir, sparse_info)
            sparse = BM25Index.load(index_dir)
            if sparse.document_count != len(corpus.chunks):
                raise ValueError("BM25 index size differs from corpus")

        dense_info = manifest.get("dense")
        dense = None
        if isinstance(dense_info, dict):
            cls._verify_files(index_dir, dense_info)
            dense = DenseIndex.load(
                index_dir,
                model_name=str(dense_info["embedding_model"]),
                model_revision=(
                    str(dense_info["embedding_model_revision"])
                    if dense_info.get("embedding_model_revision")
                    else None
                ),
                device=device,
                batch_size=int(dense_info.get("batch_size", 64)),
                backend=str(dense_info.get("backend", "sentence_transformers")),
                base_url=str(dense_info.get("base_url") or "http://127.0.0.1:11434"),
                backend_model=dense_info.get("backend_model"),
                model_digest=dense_info.get("model_digest"),
                query_prompt=str(dense_info.get("query_prompt") or ""),
            )
            if dense.document_count != len(corpus.chunks):
                raise ValueError("FAISS index size differs from corpus")
        return cls(corpus=corpus, sparse=sparse, dense=dense)

    @staticmethod
    def _verify_files(index_dir: Path, component: dict[str, Any]) -> None:
        for filename, expected_hash in component.get("files", {}).items():
            path = index_dir / filename
            if sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"{path}: index file hash mismatch")

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        fusion: str = DEFAULT_FUSION,
        top_k: int = 10,
        candidate_k: int | None = None,
        sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
        dense_weight: float = 1.0,
        rrf_k: int = 60,
        max_per_document: int = 2,
        retrieval_profile: str = "baseline",
        neighbor_radius: int = 0,
    ) -> list[dict[str, Any]]:
        return self.search_many(
            [query],
            mode=mode,
            fusion=fusion,
            top_k=top_k,
            candidate_k=candidate_k,
            sparse_weight=sparse_weight,
            dense_weight=dense_weight,
            rrf_k=rrf_k,
            max_per_document=max_per_document,
            retrieval_profile=retrieval_profile,
            neighbor_radius=neighbor_radius,
        )[0]

    def search_many(
        self,
        queries: list[str],
        *,
        mode: str = "hybrid",
        fusion: str = DEFAULT_FUSION,
        top_k: int = 10,
        candidate_k: int | None = None,
        sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
        dense_weight: float = 1.0,
        rrf_k: int = 60,
        max_per_document: int = 2,
        retrieval_profile: str = "baseline",
        neighbor_radius: int = 0,
    ) -> list[list[dict[str, Any]]]:
        if not queries:
            return []
        if top_k <= 0:
            return [[] for _ in queries]
        if mode not in {"sparse", "dense", "hybrid"}:
            raise ValueError(f"unsupported retrieval mode: {mode}")
        if retrieval_profile not in {"baseline", "focused", "field_aware", "record_safe"} or neighbor_radius not in {0, 1, 2}:
            raise ValueError("invalid retrieval profile or neighbor radius")
        if (retrieval_profile != "baseline" or neighbor_radius) and self._focused is None:
            from hw2rag.retrieval.focused import FocusedCorpus
            self._focused = FocusedCorpus(self.corpus.chunks)
        if retrieval_profile == "field_aware" and self._fields is None:
            from hw2rag.retrieval.field_evidence import FieldEvidence
            self._fields = FieldEvidence.load(self.corpus)
        if retrieval_profile == "record_safe" and self._safe is None:
            from hw2rag.retrieval.record_safe import RecordSafeEvidence
            self._safe = RecordSafeEvidence.load(self.corpus)
        candidate_k = candidate_k or max(50, top_k * 5)
        sparse_rankings: list[list[tuple[int, float]]] | None = None
        dense_rankings: list[list[tuple[int, float]]] | None = None
        if mode in {"sparse", "hybrid"}:
            if self.sparse is None:
                raise ValueError("sparse index is not available")
            sparse_rankings = [self.sparse.search(query, top_k=candidate_k) for query in queries]
        if mode in {"dense", "hybrid"}:
            if self.dense is None:
                raise ValueError("dense index is not available")
            dense_rankings = self.dense.search_many(queries, top_k=candidate_k)

        outputs: list[list[dict[str, Any]]] = []
        for query_index in range(len(queries)):
            rankings: dict[str, list[tuple[int, float]]] = {}
            if sparse_rankings is not None:
                rankings["sparse"] = sparse_rankings[query_index]
            if dense_rankings is not None:
                rankings["dense"] = dense_rankings[query_index]
            candidates = self._fuse_rankings(
                rankings,
                mode=mode,
                fusion=fusion,
                sparse_weight=sparse_weight,
                dense_weight=dense_weight,
                rrf_k=rrf_k,
            )
            if retrieval_profile in {"focused", "field_aware", "record_safe"}:
                candidates = self._focused.rerank(queries[query_index], candidates)
            outputs.append(
                self._hydrate(
                    candidates,
                    top_k=top_k,
                    max_per_document=max_per_document,
                )
            )
            if neighbor_radius:
                outputs[-1] = self._focused.expand(outputs[-1], neighbor_radius)
            if retrieval_profile == "field_aware":
                outputs[-1] = self._fields.apply(queries[query_index], outputs[-1], top_k=top_k)
            if retrieval_profile == "record_safe":
                outputs[-1] = self._safe.apply(queries[query_index], outputs[-1], top_k=top_k)
        return outputs

    def _fuse_rankings(
        self,
        rankings: dict[str, list[tuple[int, float]]],
        *,
        mode: str,
        fusion: str,
        sparse_weight: float,
        dense_weight: float,
        rrf_k: int,
    ) -> list[FusedCandidate]:
        if mode == "sparse":
            candidates = self._single_component(rankings["sparse"], "sparse")
        elif mode == "dense":
            candidates = self._single_component(rankings["dense"], "dense")
        elif fusion == "rrf":
            candidates = reciprocal_rank_fusion(
                rankings,
                rrf_k=rrf_k,
                weights={"sparse": sparse_weight, "dense": dense_weight},
            )
        elif fusion == "weighted":
            candidates = normalized_weighted_fusion(
                rankings,
                weights={"sparse": sparse_weight, "dense": dense_weight},
            )
        else:
            raise ValueError(f"unsupported mode/fusion: {mode}/{fusion}")
        return candidates

    @staticmethod
    def _single_component(
        ranking: list[tuple[int, float]],
        name: str,
    ) -> list[FusedCandidate]:
        return [
            FusedCandidate(
                row_index=row_index,
                score=score,
                component_scores={name: score},
                component_ranks={name: rank},
            )
            for rank, (row_index, score) in enumerate(ranking, start=1)
        ]

    def _hydrate(
        self,
        candidates: list[FusedCandidate],
        *,
        top_k: int,
        max_per_document: int,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        per_document: Counter[str] = Counter()
        for candidate in candidates:
            chunk = self.corpus.chunks[candidate.row_index]
            document_id = str(chunk["document_id"])
            if max_per_document > 0 and per_document[document_id] >= max_per_document:
                continue
            per_document[document_id] += 1
            output.append(
                {
                    "rank": len(output) + 1,
                    "score": candidate.score,
                    "component_scores": candidate.component_scores,
                    "component_ranks": candidate.component_ranks,
                    "row_index": candidate.row_index,
                    "chunk": chunk,
                }
            )
            if len(output) >= top_k:
                break
        return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--index-dir", type=Path, default=Path(DEFAULT_INDEX_DIR))
    parser.add_argument("--mode", choices=("sparse", "dense", "hybrid"), default="hybrid")
    parser.add_argument("--fusion", choices=("rrf", "weighted"), default=DEFAULT_FUSION)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument("--sparse-weight", type=float, default=DEFAULT_SPARSE_WEIGHT)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--max-per-document", type=int, default=2)
    parser.add_argument("--retrieval-profile", choices=("baseline", "focused", "field_aware", "record_safe"), default="baseline")
    parser.add_argument("--neighbor-radius", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    index_dir = args.index_dir if args.index_dir.is_absolute() else root / args.index_dir
    retriever = Retriever.load(root=root, index_dir=index_dir, device=args.device)
    results = retriever.search(
        args.query,
        mode=args.mode,
        fusion=args.fusion,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        sparse_weight=args.sparse_weight,
        dense_weight=args.dense_weight,
        rrf_k=args.rrf_k,
        max_per_document=args.max_per_document,
        retrieval_profile=args.retrieval_profile,
        neighbor_radius=args.neighbor_radius,
    )
    if args.json:
        print(json.dumps({"query": args.query, "results": results}, ensure_ascii=False, indent=2))
        return 0
    for result in results:
        chunk = result["chunk"]
        component = ", ".join(
            f"{name}={score:.4f}"
            for name, score in sorted(result["component_scores"].items())
        )
        print(
            f"[{result['rank']}] {result['score']:.6f} ({component}) "
            f"{chunk['source_id']} | {chunk['title']}"
        )
        preview = str(chunk["text"]).replace("\n", " ")
        print(f"    {preview[:500]}")
        if chunk.get("canonical_url"):
            print(f"    {chunk['canonical_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
