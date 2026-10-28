"""Generate concise answers with retrieval augmentation or a closed-book reader."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from hw2rag.utils import utc_now, write_json
from hw2rag.qa.prompt import PROMPT_VERSION, CONCISE_PROMPT_VERSION, GROUNDED_PROMPT_VERSION, build_evidence_context, system_prompt
from hw2rag.qa.reader import (
    DEFAULT_OLLAMA_MODEL,
    ExtractiveReader,
    OllamaReader,
    Reader,
    is_valid_answer,
)
from hw2rag.retrieval.search import DEFAULT_FUSION, DEFAULT_SPARSE_WEIGHT, Retriever
from hw2rag.retrieval.ollama_embedding import DEFAULT_INDEX_DIR, request_json


TRACE_SCHEMA_VERSION = 1


def load_questions(path: Path, *, limit: int | None = None) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        data = data["questions"]
    items: list[dict[str, str]] = []
    if isinstance(data, list):
        for position, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"{path}: question {position} is not an object")
            question_id = str(item.get("id", position)).strip()
            question = str(item.get("question", "")).strip()
            if not question_id or not question:
                raise ValueError(f"{path}: question {position} needs id and question")
            items.append({"id": question_id, "question": question})
    elif isinstance(data, dict):
        for question_id, value in data.items():
            if question_id == "andrewid":
                continue
            if isinstance(value, dict):
                question = str(value.get("question", "")).strip()
            else:
                question = str(value).strip()
            if not question:
                raise ValueError(f"{path}: question {question_id!r} is empty")
            items.append({"id": str(question_id), "question": question})
    else:
        raise ValueError(f"{path}: expected a question array or object")
    seen: set[str] = set()
    for item in items:
        if item["id"] in seen:
            raise ValueError(f"{path}: duplicate question id {item['id']!r}")
        seen.add(item["id"])
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        items = items[:limit]
    return items


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _load_existing_predictions(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: existing predictions must be an object")
    return {
        str(question_id): str(answer)
        for question_id, answer in data.items()
        if question_id != "andrewid"
    }


def _submission(predictions: dict[str, str], andrewid: str | None) -> dict[str, str]:
    if andrewid:
        return {"andrewid": andrewid, **predictions}
    return dict(predictions)


def _reader_from_args(args: argparse.Namespace, *, root: Path, retriever: Retriever | None) -> Reader:
    if args.reader == "extractive":
        return ExtractiveReader()
    reader = OllamaReader(
        model=args.model,
        base_url=args.ollama_url,
        timeout=args.timeout,
        temperature=args.temperature,
        seed=args.seed,
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
        keep_alive=args.keep_alive,
        answer_profile=args.answer_profile,
        strict_json=args.strict_json,
    )
    if args.reader == "agent":
        if retriever is None:
            raise ValueError("agent reader requires a retrieval mode")
        from hw2rag.qa.agent import AgentReader
        from hw2rag.qa.corpus_tools import CorpusTools

        return AgentReader(
            reader, CorpusTools(root, retriever, mode=args.mode, search_options={
                "fusion": args.fusion, "candidate_k": args.candidate_k,
                "sparse_weight": args.sparse_weight, "dense_weight": args.dense_weight,
                "rrf_k": args.rrf_k, "max_per_document": args.max_per_document,
                "retrieval_profile": args.retrieval_profile, "neighbor_radius": args.neighbor_radius,
            }),
            max_rounds=args.agent_max_rounds, max_calls=args.agent_max_calls,
        )
    return reader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--questions", type=Path, default=Path("hw2-public/leaderboard_queries.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.json"))
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--index-dir", type=Path, default=Path(DEFAULT_INDEX_DIR))
    parser.add_argument("--reader", choices=("ollama", "agent", "extractive"), default="ollama")
    parser.add_argument("--agent-max-rounds", type=int, default=4)
    parser.add_argument("--agent-max-calls", type=int, default=6)
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11435")
    parser.add_argument("--mode", choices=("closed-book", "sparse", "dense", "hybrid"), default="hybrid")
    parser.add_argument("--fusion", choices=("rrf", "weighted"), default=DEFAULT_FUSION)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--sparse-weight", type=float, default=DEFAULT_SPARSE_WEIGHT)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--max-per-document", type=int, default=2)
    parser.add_argument("--retrieval-profile", choices=("baseline", "focused", "field_aware", "record_safe"), default="baseline")
    parser.add_argument("--neighbor-radius", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--answer-profile", choices=("baseline", "concise", "grounded"), default="baseline")
    parser.add_argument("--answer-normalization", choices=("none", "fields", "calibrated"), default="none",
                        help="calibrated = fields + the extra formatting rules in qa/normalize.calibrate_answer")
    parser.add_argument("--strict-json", action="store_true", help="validate inner answer JSON; retry malformed/length-limited attempts with at least 256 tokens")
    parser.add_argument("--max-context-chars", type=int, default=14_000)
    parser.add_argument("--max-chunk-chars", type=int, default=3_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-predict", type=int, default=96)
    parser.add_argument("--num-ctx", type=int, default=None, help="defaults to 16384 for agent, 8192 for other readers")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--keep-alive", default="15m")
    parser.add_argument("--andrewid", help="include only for public leaderboard submissions")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-id", action="append", help="select IDs for reproducible diagnostic runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_ctx is None:
        args.num_ctx = 16384 if args.reader == "agent" else 8192
    if args.answer_profile != "baseline" and args.reader != "ollama":
        raise ValueError("answer profiles currently require the ordinary Ollama reader")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.top_k <= 0 or args.candidate_k < args.top_k:
        raise ValueError("require top_k > 0 and candidate_k >= top_k")
    root = args.root.resolve()
    questions_path = _resolve(root, args.questions)
    output_path = _resolve(root, args.output)
    index_dir = _resolve(root, args.index_dir)
    trace_path = (
        _resolve(root, args.trace)
        if args.trace is not None
        else output_path.with_suffix(".trace.json")
    )
    questions = load_questions(questions_path, limit=args.limit)
    if args.only_id:
        selected_ids = set(args.only_id)
        questions = [question for question in questions if question["id"] in selected_ids]
        if {question["id"] for question in questions} != selected_ids:
            raise ValueError("some selected question IDs were not found")
    reader_digest: str | None = None
    if args.reader != "extractive":
        models = request_json(args.ollama_url, "/api/tags").get("models", [])
        matches = [model for model in models if model["name"] == args.model]
        if not matches:
            raise ValueError(f"reader model {args.model!r} is not installed at {args.ollama_url}")
        reader_digest = str(matches[0]["digest"])

    if output_path.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --resume or --overwrite")
    predictions = _load_existing_predictions(output_path) if args.resume and output_path.exists() else {}
    predictions = {
        question_id: answer
        for question_id, answer in predictions.items()
        if is_valid_answer(answer)
    }
    expected_ids = {item["id"] for item in questions}
    predictions = {
        question_id: answer
        for question_id, answer in predictions.items()
        if question_id in expected_ids
    }
    existing_trace: dict[str, Any] = {}
    if args.resume and trace_path.exists():
        loaded_trace = json.loads(trace_path.read_text(encoding="utf-8"))
        if isinstance(loaded_trace, dict) and isinstance(loaded_trace.get("items"), dict):
            existing_trace = loaded_trace["items"]
    else:
        loaded_trace = None

    pending = [item for item in questions if item["id"] not in predictions]
    retrieval_results: list[list[dict[str, Any]]]
    retriever: Retriever | None = None
    index_manifest_hash: str | None = None
    if args.mode == "closed-book":
        retrieval_results = [[] for _ in pending]
    else:
        retriever = Retriever.load(root=root, index_dir=index_dir, device=args.device)
        index_manifest_hash = sha256((index_dir / "manifest.json").read_bytes()).hexdigest()
        retrieval_results = retriever.search_many(
            [item["question"] for item in pending],
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
    reader = _reader_from_args(args, root=root, retriever=retriever)
    trace_items = dict(existing_trace)
    trace: dict[str, Any] = {
        "created_at": utc_now(),
        "index_manifest_sha256": index_manifest_hash,
        "items": trace_items,
        "model": ExtractiveReader.model if args.reader == "extractive" else args.model,
        "reader_model_digest": reader_digest,
        "parameters": {
            "candidate_k": args.candidate_k,
            "dense_weight": args.dense_weight,
            "fusion": args.fusion,
            "max_chunk_chars": args.max_chunk_chars,
            "max_context_chars": args.max_context_chars,
            "max_per_document": args.max_per_document,
            "mode": args.mode,
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
            "reader": args.reader,
            "rrf_k": args.rrf_k,
            "seed": args.seed,
            "sparse_weight": args.sparse_weight,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "agent_max_rounds": args.agent_max_rounds if args.reader == "agent" else None,
            "agent_max_calls": args.agent_max_calls if args.reader == "agent" else None,
        },
        "prompt_version": {"baseline": PROMPT_VERSION, "concise": CONCISE_PROMPT_VERSION, "grounded": GROUNDED_PROMPT_VERSION}[args.answer_profile],
        "questions_path": str(questions_path.relative_to(root)),
        "questions_sha256": sha256(questions_path.read_bytes()).hexdigest(),
        "selected_question_ids": [item["id"] for item in questions],
        "schema_version": TRACE_SCHEMA_VERSION,
    }
    if args.retrieval_profile != "baseline":
        trace["parameters"]["retrieval_profile"] = args.retrieval_profile
    if args.retrieval_profile in {"field_aware", "record_safe"}:
        from hw2rag.retrieval import field_evidence
        trace["parameters"]["retrieval_profile_version"] = field_evidence.VERSION
        implementation_files = ("retrieval/field_evidence.py", "retrieval/focused.py", "retrieval/search.py")
        if args.retrieval_profile == "record_safe":
            from hw2rag.retrieval import record_safe
            trace["parameters"]["retrieval_profile_version"] = record_safe.VERSION
            implementation_files += ("retrieval/record_safe.py",)
        trace["parameters"]["retrieval_implementation_sha256"] = sha256(b"".join(
            (Path(__file__).resolve().parents[1] / file).read_bytes() for file in implementation_files)).hexdigest()
    if args.neighbor_radius:
        trace["parameters"]["neighbor_radius"] = args.neighbor_radius
    if args.answer_profile != "baseline":
        trace["parameters"]["answer_profile"] = args.answer_profile
    if args.answer_normalization != "none":
        trace["parameters"]["answer_normalization"] = args.answer_normalization
    if args.strict_json:
        trace["parameters"]["strict_json"] = True
    prompt_content = system_prompt(closed_book=args.mode == "closed-book", answer_profile=args.answer_profile)
    if args.reader == "agent":
        from hw2rag.qa.agent import AGENT_INSTRUCTION, AGENT_PROMPT_VERSION
        from hw2rag.qa.corpus_tools import TOOL_SCHEMAS

        prompt_content += AGENT_INSTRUCTION + json.dumps(TOOL_SCHEMAS, sort_keys=True)
        trace["agent_prompt_version"] = AGENT_PROMPT_VERSION
    trace["prompt_sha256"] = sha256(prompt_content.encode()).hexdigest()
    identity_fields = ("index_manifest_sha256", "model", "reader_model_digest", "parameters", "questions_sha256", "selected_question_ids", "prompt_sha256")
    trace["run_identity"] = sha256(json.dumps({key: trace[key] for key in identity_fields}, sort_keys=True).encode()).hexdigest()
    if args.resume and predictions:
        if not loaded_trace or loaded_trace.get("run_identity") != trace["run_identity"]:
            raise ValueError("cannot resume: model, prompt, index, questions, or parameters differ; use a new output path")
        trace["created_at"] = loaded_trace["created_at"]

    for item, results in zip(pending, retrieval_results, strict=True):
        context = None
        if args.mode != "closed-book":
            context = build_evidence_context(
                results,
                max_context_chars=args.max_context_chars,
                max_chunk_chars=args.max_chunk_chars,
            )
        response = reader.answer(item["question"], context)
        answer = response.answer
        normalization_steps = []
        if args.answer_normalization in {"fields", "calibrated"}:
            from hw2rag.qa.normalize import normalize_field_answer
            answer, normalization_steps = normalize_field_answer(item["question"], answer)
        if args.answer_normalization == "calibrated":
            from hw2rag.qa.normalize import calibrate_answer
            answer, extra_steps = calibrate_answer(item["question"], answer)
            normalization_steps += extra_steps
        predictions[item["id"]] = answer
        trace_items[item["id"]] = {
            "answer": answer,
            "context_truncated": context.truncated if context is not None else False,
            "duration_seconds": response.duration_seconds,
            "evidence": context.sources if context is not None else [],
            "model": response.model,
            "question": item["question"],
            "raw_response": response.raw_response,
            "reader_metadata": response.metadata,
        }
        if args.answer_normalization != "none":
            trace_items[item["id"]]["reader_answer"] = response.answer
            trace_items[item["id"]]["normalization_steps"] = normalization_steps
        ordered_predictions = {
            question["id"]: predictions[question["id"]]
            for question in questions
            if question["id"] in predictions
        }
        write_json(output_path, _submission(ordered_predictions, args.andrewid))
        trace["completed_at"] = utc_now()
        trace["completed_count"] = len(ordered_predictions)
        trace["question_count"] = len(questions)
        write_json(trace_path, trace)
        print(f"[{len(predictions)}/{len(questions)}] {item['id']}: {answer}", flush=True)

    if not pending:
        ordered_predictions = {item["id"]: predictions[item["id"]] for item in questions}
        write_json(output_path, _submission(ordered_predictions, args.andrewid))
        trace["completed_at"] = utc_now()
        trace["completed_count"] = len(ordered_predictions)
        trace["question_count"] = len(questions)
        write_json(trace_path, trace)
    print(f"predictions: {output_path}")
    print(f"trace: {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
