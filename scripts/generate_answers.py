#!/usr/bin/env python3
"""Generate system outputs from a source checkout."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hw2rag.qa.generate import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
