"""Score generated answers against public references with standard QA metrics."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata
from typing import Any

from hw2rag.utils import utc_now, write_json


EVALUATION_SCHEMA_VERSION = 1
ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(
        "" if unicodedata.category(character).startswith("P") else character
        for character in value
    )
    value = ARTICLES.sub(" ", value)
    return " ".join(value.split())


def _token_overlap(prediction: str, reference: str) -> tuple[float, float, float]:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens and not reference_tokens:
        return 1.0, 1.0, 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0, 0.0, 0.0
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if not overlap:
        return 0.0, 0.0, 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _lcs_length(left: list[str], right: list[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for right_token in right:
        current = [0]
        for index, left_token in enumerate(left, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0
    length = _lcs_length(prediction_tokens, reference_tokens)
    precision = length / len(prediction_tokens)
    recall = length / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if length else 0.0


def score_answer(prediction: str, references: list[str]) -> dict[str, float]:
    if not references:
        raise ValueError("at least one reference answer is required")
    exact_match = max(
        float(normalize_answer(prediction) == normalize_answer(reference))
        for reference in references
    )
    overlaps = [_token_overlap(prediction, reference) for reference in references]
    return {
        "exact_match": exact_match,
        "token_precision": max(value[0] for value in overlaps),
        "token_recall": max(value[1] for value in overlaps),
        "token_f1": max(value[2] for value in overlaps),
        "rouge_l_f1": max(rouge_l_f1(prediction, reference) for reference in references),
    }


def evaluate_predictions(
    predictions: dict[str, str],
    references: list[dict[str, Any]],
) -> tuple[dict[str, float | int | str], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for item in references:
        question_id = str(item["id"])
        prediction = predictions.get(question_id, "")
        answers = [str(answer) for answer in item["answers"]]
        metrics = score_answer(prediction, answers)
        totals.update(metrics)
        details.append(
            {
                "answers": answers,
                "id": question_id,
                "metrics": metrics,
                "prediction": prediction,
                "question": item["question"],
            }
        )
    count = len(references)
    summary: dict[str, float | int | str] = {
        metric: 100.0 * totals[metric] / count if count else 0.0
        for metric in ("exact_match", "token_precision", "token_recall", "token_f1", "rouge_l_f1")
    }
    summary["question_count"] = count
    summary["scale"] = "0_to_100"
    return summary, details


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--references", type=Path, default=Path("data/eval/public_evaluation_data.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--only-predicted", action="store_true", help="explicitly evaluate only reference IDs present in this diagnostic prediction file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    predictions_path = _resolve(root, args.predictions)
    references_path = _resolve(root, args.references)
    output_path = (
        _resolve(root, args.output)
        if args.output is not None
        else predictions_path.with_suffix(".metrics.json")
    )
    prediction_data = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(prediction_data, dict):
        raise ValueError("predictions must be a JSON object")
    predictions = {
        str(question_id): str(answer)
        for question_id, answer in prediction_data.items()
        if question_id != "andrewid"
    }
    references = json.loads(references_path.read_text(encoding="utf-8"))
    if not isinstance(references, list):
        raise ValueError("references must be a JSON array")
    if args.only_predicted:
        references = [item for item in references if str(item["id"]) in predictions]
        if not references:
            raise ValueError("no reference IDs match predictions")
    reference_ids = {str(item["id"]) for item in references}
    missing = sorted(reference_ids - set(predictions))
    extra = sorted(set(predictions) - reference_ids)
    if missing and not args.allow_missing:
        raise ValueError(f"missing predictions for {len(missing)} questions: {missing[:5]}")
    summary, details = evaluate_predictions(predictions, references)
    report = {
        "created_at": utc_now(),
        "details": details,
        "extra_prediction_ids": extra,
        "missing_prediction_ids": missing,
        "predictions_sha256": sha256(predictions_path.read_bytes()).hexdigest(),
        "references_sha256": sha256(references_path.read_bytes()).hexdigest(),
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "reference_subset": "prediction_ids" if args.only_predicted else "all_references",
        "summary": summary,
    }
    write_json(output_path, report)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
