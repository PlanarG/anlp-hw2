#!/usr/bin/env python3
"""Run the RAG pipeline end to end on a question file.

1. Start a local Ollama worker for the embedding model on GPU and build the index
   (BM25 + dense) from data/corpus/chunks if it does not exist yet.
2. Start a second worker for the reader model, then retrieve evidence and answer
   every question with the parameters in configs/pipeline.json.
3. Validate the output file and write a validation.json next to it.

Both models must be fully GPU-resident; the script stops instead of falling back to CPU.
It only manages the Ollama processes it starts itself (ports 11434 and 11436).

    python scripts/run_pipeline.py --check
    python scripts/run_pipeline.py --models-dir ~/.ollama/models \
        --questions hw2-public/test_set_day_2.json --output outputs/test/predictions.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CONFIG = ROOT / "configs/pipeline.json"
EMBED_PORT, READER_PORT = 11434, 11436


def load(path: Path):
    return json.loads(path.read_text())


def api(port: int, endpoint: str, data=None, timeout=30):
    request = Request(f"http://127.0.0.1:{port}{endpoint}",
                      data=json.dumps(data).encode() if data is not None else None,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if value.get("error"):
        raise RuntimeError(value["error"])
    return value


def require_free_gpu(selection: str, mib: int) -> None:
    selected = set(selection.split(","))
    if not selected or any(not value.isdigit() for value in selected):
        raise ValueError("GPU selection must be physical integer IDs, e.g. 0 or 0,1")
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    available = {a.strip(): int(b.strip()) for a, b in (line.split(",") for line in output.splitlines())}
    if not selected <= available.keys():
        raise ValueError(f"Unknown GPU IDs: {selected - available.keys()}")
    free = sum(available[value] for value in selected)
    if free < mib:
        raise RuntimeError(f"GPU {selection}: only {free / 1024:.1f} GiB free; need {mib / 1024:.1f} GiB. No CPU fallback.")


def resolve_model_store(override: Path | None) -> Path:
    selected = override or Path(os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama/models")))
    selected = selected.expanduser().resolve()
    if not (selected / "manifests").is_dir() or not (selected / "blobs").is_dir():
        raise FileNotFoundError(f"No Ollama model store (manifests/ and blobs/) at {selected}; pass --models-dir")
    return selected


def start_worker(role: str, gpu: str, port: int, models_dir: Path):
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} is already in use; refusing to reuse or stop that service")
    command = [sys.executable, str(ROOT / "scripts/serve_models.py"), role, f"--gpu={gpu}", "--port", str(port),
               "--log-file", str(ROOT / f"artifacts/runtime_logs/{role}_{port}.log"), "--models-dir", str(models_dir)]
    proc = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"{role} worker exited; see artifacts/runtime_logs")
            try:
                api(port, "/api/tags", timeout=2)
                return proc
            except (OSError, ValueError):
                time.sleep(.5)
        raise TimeoutError(f"{role} worker did not start in time")
    except BaseException:
        stop_worker(proc)
        raise


def stop_worker(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait()
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def fully_on_gpu(port: int, name: str) -> None:
    matches = [m for m in api(port, "/api/ps")["models"] if m["name"] == name]
    if len(matches) != 1 or matches[0].get("size_vram", 0) < matches[0]["size"]:
        raise RuntimeError(f"{name} is not fully GPU-resident; stopping rather than running on CPU")


def check_digest(port: int, name: str, digest: str) -> None:
    installed = [m for m in api(port, "/api/tags")["models"] if m["name"] == name]
    if len(installed) != 1 or installed[0]["digest"] != digest:
        raise ValueError(f"{name}: installed digest differs from configs/pipeline.json")


def check(cfg: dict) -> None:
    from hw2rag.retrieval.record_safe import RecordSafeEvidence
    from hw2rag.retrieval.search import Retriever
    retriever = Retriever.load(root=ROOT, index_dir=ROOT / cfg["index_dir"])
    fields = RecordSafeEvidence.load(retriever.corpus)
    print(f"OK: {len(fields.documents)} documents, {len(retriever.corpus.chunks)} chunks, index {cfg['index_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", help="question file (default: the public leaderboard questions)")
    parser.add_argument("--output", type=Path, default=Path("outputs/leaderboard/predictions.json"))
    parser.add_argument("--andrewid", help="needed for leaderboard submissions; omit for unseen-test outputs")
    parser.add_argument("--answer-normalization", choices=("fields", "calibrated"), default="fields",
                        help="fields = system 1; calibrated = system 2 (extra formatting rules)")
    parser.add_argument("--models-dir", type=Path, help="Ollama model store (default: $OLLAMA_MODELS or ~/.ollama/models)")
    parser.add_argument("--embedding-gpu", default="0")
    parser.add_argument("--reader-gpu", default="0")
    parser.add_argument("--check", action="store_true", help="only verify corpus and index; no models are loaded")
    parser.add_argument("--build-only", action="store_true", help="build the index (if missing) and stop")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        raise RuntimeError("Python 3.12+ is required")
    os.chdir(ROOT)
    cfg = load(CONFIG)
    if args.check:
        check(cfg)
        return
    questions_path = args.questions or cfg["questions_path"]
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    models_dir = resolve_model_store(args.models_dir)
    workers = []
    try:
        require_free_gpu(args.embedding_gpu, 6656)
        workers.append(start_worker("embedding", args.embedding_gpu, EMBED_PORT, models_dir))
        check_digest(EMBED_PORT, cfg["embedding_model"], cfg["embedding_digest"])
        api(EMBED_PORT, "/api/embed", {"model": cfg["embedding_model"], "input": ["warm-up"], "truncate": False,
                                       "keep_alive": "30m", "options": {"num_ctx": 4096}}, timeout=600)
        fully_on_gpu(EMBED_PORT, cfg["embedding_model"])
        if not (ROOT / cfg["index_dir"] / "manifest.json").exists():
            from hw2rag.indexing.build_index import main as build_index
            code = build_index(["--root", str(ROOT), "--chunk-manifest", cfg["chunk_manifest"],
                                "--output-dir", cfg["index_dir"], "--embedding-url", f"http://127.0.0.1:{EMBED_PORT}",
                                "--ollama-model", cfg["embedding_model"], "--no-progress"])
            if code:
                raise SystemExit(code)
            fully_on_gpu(EMBED_PORT, cfg["embedding_model"])
        print(f"index ready: {cfg['index_dir']}", flush=True)
        if args.build_only:
            return

        require_free_gpu(args.reader_gpu, 22528)
        workers.append(start_worker("reader", args.reader_gpu, READER_PORT, models_dir))
        check_digest(READER_PORT, cfg["model"], cfg["reader_model_digest"])
        api(READER_PORT, "/api/generate", {"model": cfg["model"], "keep_alive": "30m",
                                           "options": {"num_ctx": 16384}}, timeout=300)
        fully_on_gpu(READER_PORT, cfg["model"])

        from hw2rag.qa.generate import main as generate
        p = cfg["parameters"]
        argv = ["--questions", questions_path, "--index-dir", cfg["index_dir"], "--model", cfg["model"],
                "--ollama-url", f"http://127.0.0.1:{READER_PORT}", "--retrieval-profile", p["retrieval_profile"],
                "--neighbor-radius", str(p["neighbor_radius"]), "--strict-json",
                "--answer-normalization", args.answer_normalization, "--num-ctx", str(p["num_ctx"]),
                "--num-predict", str(p["num_predict"]), "--timeout", "600", "--output", str(output), "--resume"]
        if args.andrewid:
            argv += ["--andrewid", args.andrewid]
        generate(argv)
        fully_on_gpu(EMBED_PORT, cfg["embedding_model"])
        fully_on_gpu(READER_PORT, cfg["model"])

        from hw2rag.qa.generate import load_questions
        from hw2rag.qa.validate_submission import validate_submission
        predictions = load(output)
        questions = load_questions(ROOT / questions_path)
        validate_submission(predictions, [q["id"] for q in questions],
                            submission_format="leaderboard" if args.andrewid else "unseen")
        trace = load(output.with_suffix(".trace.json"))
        keys = ["model", "reader_model_digest", "prompt_sha256"] + (
            ["questions_sha256"] if questions_path == cfg["questions_path"] else [])
        report = {
            "predictions": str(output.relative_to(ROOT)), "questions": questions_path, "answers": len(questions),
            "unknown_count": sum(predictions[q["id"]] == "UNKNOWN" for q in questions),
            "matches_config": {k: trace[k] == cfg[k] for k in keys},
            "parameter_differences": {k: [p.get(k), v] for k, v in trace["parameters"].items() if p.get(k) != v},
            "index_manifest_sha256": trace["index_manifest_sha256"],
        }
        (output.parent / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        for worker in reversed(workers):
            stop_worker(worker)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped; finished answers are kept. Rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
