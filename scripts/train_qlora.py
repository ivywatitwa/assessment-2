#!/usr/bin/env python3
"""Guarded QLoRA preflight for the official MedGemma training workflow.

The official Google multimodal collator and model-loading code must be copied
and verified against the pinned model revision before a real run. This script
does not silently substitute a text-only trainer or generate a checkpoint.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from env_utils import load_hf_token  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="refuse clearly until the official collator is adapted")
    args = parser.parse_args()
    load_hf_token()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    checks = {
        "study_policy": config.get("study_policy") == "secondary_data_only",
        "model_id_is_multimodal_candidate": config.get("model_id") == "google/medgemma-4b-it",
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "transformers_installed": importlib.util.find_spec("transformers") is not None,
        "peft_installed": importlib.util.find_spec("peft") is not None,
        "trl_installed": importlib.util.find_spec("trl") is not None,
        "bitsandbytes_installed": importlib.util.find_spec("bitsandbytes") is not None,
        "public_instruction_dataset_exists": (ROOT / config.get("instruction_dir", "data/secondary/instruction_tuning")).exists(),
        "synthetic_auxiliary_dataset_exists": (ROOT / config.get("synthetic_t2_dir", "data/secondary/instruction_tuning_synthetic")).exists(),
        "public_image_split_exists": (ROOT / "data" / "secondary" / "splits" / "images").exists(),
    }
    required = [key for key in checks if key not in {"public_instruction_dataset_exists"}]
    result = {"config": str(args.config), "checks": checks, "status": "READY" if all(checks[key] for key in required) else "BLOCKED"}
    print(json.dumps(result, indent=2))
    if args.run:
        raise SystemExit(
            "Real QLoRA training is blocked until every preflight check passes and the official "
            "MedGemma multimodal collator has been adapted and overfit-tested. No checkpoint was created."
        )
    return 0 if result["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
