#!/usr/bin/env python3
"""Preflight the text-attribution experiment without fabricating SHAP values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    split = Path(config.get("instruction_dir", "data/secondary/instruction_tuning")) / f"{args.split}.jsonl"
    result = {
        "split": str(split),
        "split_exists": split.exists(),
        "target_requirement": "fixed teacher-forced output span and documented masking protocol",
        "status": "READY" if split.exists() else "BLOCKED_NO_TEXT_SPLIT",
    }
    print(json.dumps(result, indent=2))
    if args.run:
        raise SystemExit("Text explanation execution is blocked until eligible source-derived text, model weights and SHAP/IG dependencies exist.")
    return 0 if result["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
