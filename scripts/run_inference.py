#!/usr/bin/env python3
"""Run a guarded MedGemma inference experiment.

This runner refuses to start if the input split is missing or the model weights
are unavailable. It is intentionally not a deployment interface. Raw output and
configuration provenance must be saved by the caller's experiment directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_utils import load_hf_token  # noqa: E402


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"input split does not exist: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"input split is empty: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="google/medgemma-4b-it")
    parser.add_argument("--revision")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_hf_token()
    rows = load_rows(args.input)
    config = {
        "model_id": args.model_id,
        "revision": args.revision,
        "experiment": args.experiment,
        "adapter": args.adapter,
        "max_new_tokens": args.max_new_tokens,
        "records": len(rows),
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
    }
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "config": config}, indent=2))
        return 0

    try:
        from transformers import AutoProcessor  # noqa: F401
    except ImportError as exc:
        raise SystemExit("transformers is required for inference") from exc
    if not os.environ.get("HF_TOKEN") and args.model_id.startswith("google/"):
        raise SystemExit("HF_TOKEN is required for gated MedGemma inference")
    raise SystemExit(
        "Model execution is intentionally not implicit: adapt the official MedGemma "
        "multimodal collator/model example for this repository before running a real experiment. "
        "The dry-run verifies data/configuration without fabricating predictions."
    )


if __name__ == "__main__":
    raise SystemExit(main())
