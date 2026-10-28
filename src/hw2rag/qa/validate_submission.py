"""Validate an answer file against the assignment's exact submission format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hw2rag.qa.generate import load_questions
from hw2rag.qa.reader import is_valid_answer


def validate_submission(
    data: Any,
    expected_ids: list[str],
    *,
    submission_format: str,
) -> None:
    if not isinstance(data, dict):
        raise ValueError("submission must be a JSON object")
    has_andrewid = "andrewid" in data
    if submission_format == "leaderboard":
        if not has_andrewid or not isinstance(data["andrewid"], str) or not data["andrewid"].strip():
            raise ValueError("leaderboard submission requires a non-empty string 'andrewid'")
    elif has_andrewid:
        raise ValueError("unseen-test submission must not contain 'andrewid'")

    answer_ids = [str(key) for key in data if key != "andrewid"]
    missing = sorted(set(expected_ids) - set(answer_ids))
    extra = sorted(set(answer_ids) - set(expected_ids))
    if missing or extra:
        raise ValueError(f"question ID mismatch: missing={missing[:5]}, extra={extra[:5]}")
    invalid: list[str] = []
    for question_id in expected_ids:
        answer = data.get(question_id)
        if not isinstance(answer, str) or not is_valid_answer(answer):
            invalid.append(question_id)
    if invalid:
        raise ValueError(f"invalid answer strings for {len(invalid)} questions: {invalid[:5]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--format", choices=("unseen", "leaderboard"), default="unseen")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    submission_path = args.submission if args.submission.is_absolute() else root / args.submission
    questions_path = args.questions if args.questions.is_absolute() else root / args.questions
    data = json.loads(submission_path.read_text(encoding="utf-8"))
    questions = load_questions(questions_path)
    validate_submission(
        data,
        [item["id"] for item in questions],
        submission_format=args.format,
    )
    print(f"valid {args.format} submission: {len(questions)} answers ({submission_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
