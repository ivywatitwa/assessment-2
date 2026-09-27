#!/usr/bin/env python3
"""Run bounded synthetic-only T2 generation with the recovered adapter.

The adapter was trained on T1, not T2. Outputs from this command are therefore
auxiliary generation diagnostics and must not be presented as clinical results.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xai_model_utils import generate_target, load_medgemma  # noqa: E402


def load_records(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("task") == "T2_treatment" and row.get("image") is None]


def field(text: str, name: str) -> str | None:
    match = re.search(rf"^{re.escape(name)}:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--allow-mps", action="store_true")
    args = parser.parse_args()

    model, processor, device, model_dtype = load_medgemma(args.adapter, args.allow_mps)
    records = load_records(args.input)[: args.max_examples]
    output_rows: list[dict[str, Any]] = []

    for index, row in enumerate(records, 1):
        user_message = {"role": "user", "content": row["messages"][0]["content"]}
        raw_output = generate_target(
            model,
            processor,
            [user_message],
            device,
            model_dtype,
            args.max_new_tokens,
        )
        gold = row["messages"][1]["content"]
        expected_diagnosis = field(gold, "Diagnosis")
        expected_drug = field(gold, "Drug")
        predicted_diagnosis = field(raw_output, "Diagnosis")
        predicted_drug = field(raw_output, "Drug")
        output_rows.append({
            "item_id": row["meta"]["case_id"],
            "provenance": row["meta"].get("provenance"),
            "clinical_use": row["meta"].get("clinical_use"),
            "evaluation_role": row["meta"].get("evaluation_role"),
            "gold_diagnosis": expected_diagnosis,
            "gold_drug": expected_drug,
            "predicted_diagnosis": predicted_diagnosis,
            "predicted_drug": predicted_drug,
            "gold_referral": "insufficient evidence" in gold.lower() or "refer to" in gold.lower(),
            "predicted_referral": "insufficient evidence" in raw_output.lower() or "refer to" in raw_output.lower(),
            "raw_output": raw_output,
        })
        print(f"generated {index}/{len(records)}: {row['meta']['case_id']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
