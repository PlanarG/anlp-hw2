#!/usr/bin/env python3
"""Apply the calibrated answer-formatting profile to an existing run.

Reads a run trace (*.trace.json written by hw2rag.qa.generate), re-applies
normalize_field_answer + calibrate_answer to each raw reader answer, and writes a
submission file. This is exactly what `--answer-normalization calibrated` does at
generation time, without re-running retrieval or the reader.

    python scripts/calibrate_outputs.py --trace outputs/run/predictions.trace.json \
        --output system_outputs/system_output_2.json [--andrewid ID] [--report changes.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw2rag.qa.normalize import calibrate_answer, normalize_field_answer  # noqa: E402
from hw2rag.qa.validate_submission import validate_submission  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--andrewid", help="include for leaderboard submissions; omit for unseen-test outputs")
    parser.add_argument("--report", type=Path, help="optional JSON list of the answers that changed")
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text())
    ids = list(trace["selected_question_ids"])
    answers, changes = {}, []
    for qid in ids:
        item = trace["items"][qid]
        raw = item.get("reader_answer", item["answer"])
        fields, steps = normalize_field_answer(item["question"], raw)
        calibrated, extra = calibrate_answer(item["question"], fields)
        answers[qid] = calibrated
        if calibrated != item["answer"]:
            changes.append({"id": qid, "question": item["question"], "before": item["answer"],
                            "after": calibrated, "rules": extra})
    submission = {"andrewid": args.andrewid, **answers} if args.andrewid else answers
    validate_submission(submission, ids, submission_format="leaderboard" if args.andrewid else "unseen")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(submission, ensure_ascii=False, indent=2) + "\n")
    if args.report:
        args.report.write_text(json.dumps(changes, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(changes)}/{len(ids)} answers changed -> {args.output}")
    for c in changes:
        print(f"  #{c['id']} [{', '.join(c['rules'])}]\n      {c['before']!r}\n   -> {c['after']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
