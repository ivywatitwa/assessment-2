#!/usr/bin/env python3
"""Preflight the image-attribution experiment without fabricating heatmaps."""
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
        "target_requirement": "complete canonical class-sequence score and verified vision-token grid",
        "status": "READY" if split.exists() else "BLOCKED_NO_IMAGE_SPLIT",
    }
    print(json.dumps(result, indent=2))
    if args.run:
        raise SystemExit("Image explanation execution is blocked until eligible images, model weights and target-layer probes exist.")
    return 0 if result["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
