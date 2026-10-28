"""Build persistent BM25 and FAISS indexes over retrieval chunks."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import logging
from pathlib import Path

from hw2rag.utils import atomic_write, utc_now, write_json
from hw2rag.retrieval.bm25 import BM25Index
from hw2rag.retrieval.corpus import ChunkCorpus
from hw2rag.retrieval.dense import DenseIndex
from hw2rag.retrieval.ollama_embedding import DEFAULT_INDEX_DIR, QWEN_MODEL, QWEN_OLLAMA_TAG


LOGGER = logging.getLogger(__name__)
INDEX_SCHEMA_VERSION = 1


def portable_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="project root")
    parser.add_argument(
        "--chunk-manifest",
        type=Path,
        default=Path("data/corpus/chunks/_manifest.json"),
        help="chunk manifest, relative to root by default",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_INDEX_DIR),
        help="index output directory, relative to root by default",
    )
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--embedding-backend", choices=("ollama", "sentence_transformers"), default="ollama")
    parser.add_argument("--embedding-url", default="http://127.0.0.1:11434", help="one local Ollama URL, or comma-separated build workers serving the same model")
    parser.add_argument("--ollama-model", default=QWEN_OLLAMA_TAG)
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Hugging Face model revision/commit; pass an empty string for the latest revision",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--only-source", action="append")
    parser.add_argument("--skip-sparse", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--reuse-index", type=Path, help="reuse exactly unchanged text vectors from a verified index of the same model")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.skip_sparse and args.skip_dense:
        LOGGER.error("cannot skip both sparse and dense indexes")
        return 2
    if args.batch_size <= 0 or args.k1 <= 0 or not 0 <= args.b <= 1:
        LOGGER.error("require batch_size > 0, k1 > 0, and 0 <= b <= 1")
        return 2

    root = args.root.resolve()
    chunk_manifest = args.chunk_manifest if args.chunk_manifest.is_absolute() else root / args.chunk_manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    previous_dir = None
    if args.reuse_index:
        previous_dir = args.reuse_index if args.reuse_index.is_absolute() else root / args.reuse_index
        if previous_dir.resolve() == output_dir.resolve():
            raise ValueError("--reuse-index must differ from --output-dir; preserve the original index")
    corpus = ChunkCorpus.load(chunk_manifest, source_ids=args.only_source)
    LOGGER.info("loaded %s chunks from %s sources", len(corpus.chunks), len(corpus.source_ids))
    previous = None
    if previous_dir is not None and not args.skip_dense:
        if args.embedding_backend != "ollama":
            raise ValueError("--reuse-index currently supports only the pinned Ollama embedding backend")
        from hw2rag.retrieval.search import Retriever

        previous_manifest = json.loads((previous_dir / "manifest.json").read_text())
        previous_info = previous_manifest.get("dense") or {}
        if not previous_info.get("normalize_embeddings") or previous_info.get("max_sequence_length") != 4096:
            raise ValueError("previous vector normalization/context must match the current embedding protocol")
        previous = Retriever.load(root=root, index_dir=previous_dir)
        if previous.dense is None:
            raise ValueError("previous index does not contain dense vectors")
    output_dir.mkdir(parents=True, exist_ok=True)

    row_ids_path = output_dir / "row_ids.json"
    row_ids_content = json.dumps(corpus.row_ids, ensure_ascii=False, separators=(",", ":")) + "\n"
    atomic_write(row_ids_path, row_ids_content)
    manifest: dict[str, object] = {
        "build_completed_at": None,
        "chunk_count": len(corpus.chunks),
        "chunk_manifest": portable_path(corpus.manifest_path, root),
        "chunk_manifest_sha256": corpus.manifest_sha256,
        "dense": None,
        "row_ids": {
            "path": row_ids_path.name,
            "sha256": sha256(row_ids_content.encode("utf-8")).hexdigest(),
        },
        "schema_version": INDEX_SCHEMA_VERSION,
        "sources": list(corpus.source_ids),
        "sparse": None,
    }

    if not args.skip_sparse:
        LOGGER.info("building BM25 index")
        sparse = BM25Index.build(corpus.texts, k1=args.k1, b=args.b)
        manifest["sparse"] = sparse.save(output_dir)

    if not args.skip_dense:
        LOGGER.info("building dense index with %s", args.model)
        dense = DenseIndex.build(
            corpus.texts,
            model_name=args.model,
            model_revision=args.model_revision or None,
            device=args.device,
            batch_size=args.batch_size,
            show_progress=not args.no_progress,
            backend=args.embedding_backend,
            base_url=args.embedding_url,
            backend_model=args.ollama_model if args.embedding_backend == "ollama" else None,
            checkpoint_dir=output_dir / "embedding_checkpoints",
            reuse_index=previous.dense if previous else None,
            reuse_texts=previous.corpus.texts if previous else None,
        )
        manifest["dense"] = dense.save(output_dir)

    manifest["build_completed_at"] = utc_now()
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
