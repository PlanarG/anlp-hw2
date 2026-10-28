#!/usr/bin/env python3
"""Start one project-local Ollama worker for embeddings or document reading."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("embedding", "reader"))
    parser.add_argument("--gpu", help="physical CUDA GPU index; omit to let Ollama choose")
    parser.add_argument("--port", type=int)
    parser.add_argument("--models-dir", type=Path, help="use an existing external Ollama model store")
    parser.add_argument("--log-file", type=Path, help="append runtime logs to a file to avoid blocked/unread terminal output")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    local = root / "artifacts/runtime/ollama-0.40.11/bin/ollama"
    executable = str(local) if local.exists() else shutil.which("ollama")
    if executable is None:
        raise RuntimeError("install Ollama first; see README")
    env = dict(os.environ)
    env.update({
        "OLLAMA_MODELS": str(args.models_dir.expanduser().resolve() if args.models_dir is not None else root / "artifacts/ollama/models"),
        "OLLAMA_HOST": f"127.0.0.1:{args.port or (11434 if args.role == 'embedding' else 11435)}",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_NO_CLOUD": "1",
        "OLLAMA_CONTEXT_LENGTH": "4096" if args.role == "embedding" else "16384",
    })
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        # Vulkan enumerates GPUs independently of CUDA_VISIBLE_DEVICES. An
        # inherited OLLAMA_VULKAN=true can otherwise ignore --gpu and collide
        # with another worker, forcing slow CPU offload.
        env["OLLAMA_VULKAN"] = "false"
    if args.log_file is not None:
        log_path = args.log_file.resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Appending model-worker logs to {log_path}", flush=True)
        descriptor = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
        if descriptor > 2:
            os.close(descriptor)
    os.execve(executable, [executable, "serve"], env)


if __name__ == "__main__":
    main()
