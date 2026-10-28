"""Evaluate retrieval with answer-string recall on the public questions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Callable

from hw2rag.utils import utc_now, write_json
from hw2rag.retrieval.hybrid import (
    FusedCandidate,
    normalized_weighted_fusion,
    reciprocal_rank_fusion,
)
from hw2rag.retrieval.search import Retriever
from hw2rag.retrieval.ollama_embedding import DEFAULT_INDEX_DIR


EVALUATION_SCHEMA_VERSION = 1
MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
DATE_PATTERN = re.compile(
    rf"\b({'|'.join(MONTHS)})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{{4}}))?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    name: str
    make_candidates: Callable[
        [dict[str, list[tuple[int, float]]]],
        list[FusedCandidate],
    ]


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    without_marks = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join(re.findall(r"[a-z0-9]+", without_marks))


def answer_in_text(answer: str, text: str, *, normalized_text: str | None = None) -> bool:
    normalized_answer = normalize_text(answer)
    haystack = normalized_text if normalized_text is not None else normalize_text(text)
    if normalized_answer and f" {normalized_answer} " in f" {haystack} ":
        return True

    # Structured records commonly use ISO timestamps while answers spell dates out.
    for match in DATE_PATTERN.finditer(answer):
        month = MONTHS[match.group(1).casefold()]
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else None
        raw = text.casefold()
        if year is not None and re.search(rf"\b{year}-{month:02d}-{day:02d}(?!\d)", raw):
            return True
        if year is None and re.search(rf"\b\d{{4}}-{month:02d}-{day:02d}(?!\d)", raw):
            return True
    return False


def matching_answer(
    answers: list[str],
    text: str,
    *,
    normalized_text: str | None = None,
) -> str | None:
    for answer in answers:
        if answer_in_text(answer, text, normalized_text=normalized_text):
            return answer
    return None


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


def evaluation_configs() -> list[EvaluationConfig]:
    return [
        EvaluationConfig("sparse", lambda rankings: _single_component(rankings["sparse"], "sparse")),
        EvaluationConfig("dense", lambda rankings: _single_component(rankings["dense"], "dense")),
        EvaluationConfig(
            "hybrid_rrf",
            lambda rankings: reciprocal_rank_fusion(rankings, rrf_k=60),
        ),
        EvaluationConfig(
            "hybrid_rrf_sparse_2",
            lambda rankings: reciprocal_rank_fusion(
                rankings,
                rrf_k=60,
                weights={"sparse": 2.0, "dense": 1.0},
            ),
        ),
        EvaluationConfig(
            "hybrid_weighted",
            lambda rankings: normalized_weighted_fusion(rankings),
        ),
        EvaluationConfig(
            "hybrid_weighted_sparse_2",
            lambda rankings: normalized_weighted_fusion(
                rankings,
                weights={"sparse": 2.0, "dense": 1.0},
            ),
        ),
    ]


def load_questions(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array")
    questions: list[dict[str, Any]] = []
    for position, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: item {position} is not an object")
        question = str(item.get("question", "")).strip()
        answers = [str(answer).strip() for answer in item.get("answers", []) if str(answer).strip()]
        if not question or not answers:
            raise ValueError(f"{path}: item {position} needs a question and answers")
        questions.append({"id": str(item.get("id", position)), "question": question, "answers": answers})
    return questions[:limit] if limit is not None else questions


def evaluate(
    retriever: Retriever,
    questions: list[dict[str, Any]],
    *,
    top_k: int = 10,
    candidate_k: int = 50,
    max_per_document: int = 2,
) -> tuple[dict[str, dict[str, float | int]], list[dict[str, Any]]]:
    if retriever.sparse is None or retriever.dense is None:
        raise ValueError("evaluation requires both sparse and dense indexes")
    if top_k <= 0 or candidate_k < top_k:
        raise ValueError("require top_k > 0 and candidate_k >= top_k")

    configs = evaluation_configs()
    normalized_chunks = [normalize_text(str(chunk["retrieval_text"])) for chunk in retriever.corpus.chunks]
    summaries: dict[str, dict[str, float | int]] = {
        config.name: {"hits_at_1": 0, "hits_at_5": 0, "hits_at_10": 0, "reciprocal_rank_sum": 0.0}
        for config in configs
    }
    details: list[dict[str, Any]] = []
    covered_count = 0
    dense_rankings = retriever.dense.search_many(
        (item["question"] for item in questions),
        top_k=candidate_k,
    )

    for item, dense_ranking in zip(questions, dense_rankings, strict=True):
        answers = item["answers"]
        coverage_row: int | None = None
        for row_index, chunk in enumerate(retriever.corpus.chunks):
            if matching_answer(
                answers,
                str(chunk["retrieval_text"]),
                normalized_text=normalized_chunks[row_index],
            ):
                coverage_row = row_index
                break
        if coverage_row is not None:
            covered_count += 1

        rankings = {
            "sparse": retriever.sparse.search(item["question"], top_k=candidate_k),
            "dense": dense_ranking,
        }
        matches: dict[str, dict[str, Any] | None] = {}
        for config in configs:
            results = retriever._hydrate(
                config.make_candidates(rankings),
                top_k=top_k,
                max_per_document=max_per_document,
            )
            found: dict[str, Any] | None = None
            for result in results:
                chunk = result["chunk"]
                answer = matching_answer(answers, str(chunk["retrieval_text"]))
                if answer is not None:
                    found = {
                        "answer": answer,
                        "chunk_id": chunk["chunk_id"],
                        "rank": result["rank"],
                        "source_id": chunk["source_id"],
                        "title": chunk["title"],
                    }
                    break
            matches[config.name] = found
            if found is not None:
                rank = int(found["rank"])
                summary = summaries[config.name]
                summary["reciprocal_rank_sum"] = float(summary["reciprocal_rank_sum"]) + 1.0 / rank
                if rank <= 1:
                    summary["hits_at_1"] = int(summary["hits_at_1"]) + 1
                if rank <= 5:
                    summary["hits_at_5"] = int(summary["hits_at_5"]) + 1
                if rank <= 10:
                    summary["hits_at_10"] = int(summary["hits_at_10"]) + 1
        details.append(
            {
                **item,
                "lexically_covered": coverage_row is not None,
                "coverage_chunk_id": (
                    retriever.corpus.chunks[coverage_row]["chunk_id"]
                    if coverage_row is not None
                    else None
                ),
                "matches": matches,
            }
        )

    question_count = len(questions)
    for summary in summaries.values():
        for cutoff in (1, 5, 10):
            hits = int(summary[f"hits_at_{cutoff}"])
            summary[f"hit_rate_at_{cutoff}"] = hits / question_count if question_count else 0.0
            summary[f"covered_recall_at_{cutoff}"] = hits / covered_count if covered_count else 0.0
        reciprocal_rank_sum = float(summary.pop("reciprocal_rank_sum"))
        summary["covered_mrr"] = reciprocal_rank_sum / covered_count if covered_count else 0.0
        summary["question_count"] = question_count
        summary["lexically_covered_count"] = covered_count
    return summaries, details


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--questions", type=Path, default=Path("data/eval/public_evaluation_data.json"))
    parser.add_argument("--index-dir", type=Path, default=Path(DEFAULT_INDEX_DIR))
    parser.add_argument("--output", type=Path, default=Path("outputs/retrieval/public_eval.json"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--max-per-document", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cpu")
    return parser


def _from_root(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    questions_path = _from_root(root, args.questions)
    index_dir = _from_root(root, args.index_dir)
    output_path = _from_root(root, args.output)
    questions = load_questions(questions_path, limit=args.limit)
    retriever = Retriever.load(root=root, index_dir=index_dir, device=args.device)
    summaries, details = evaluate(
        retriever,
        questions,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        max_per_document=args.max_per_document,
    )
    report = {
        "created_at": utc_now(),
        "details": details,
        "index_manifest_sha256": sha256((index_dir / "manifest.json").read_bytes()).hexdigest(),
        "metric_note": (
            "Answer-string hit rate is a retrieval proxy, not final QA accuracy. "
            "Covered recall uses only questions whose answer string occurs in the chunk corpus."
        ),
        "parameters": {
            "candidate_k": args.candidate_k,
            "max_per_document": args.max_per_document,
            "top_k": args.top_k,
        },
        "questions_path": str(questions_path.relative_to(root)),
        "questions_sha256": sha256(questions_path.read_bytes()).hexdigest(),
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "summaries": summaries,
    }
    write_json(output_path, report)
    print(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
