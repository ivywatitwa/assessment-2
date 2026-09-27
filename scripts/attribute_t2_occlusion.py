#!/usr/bin/env python3
"""Create field-level token occlusion attributions for synthetic T2 prompts."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xai_model_utils import load_medgemma, score_target  # noqa: E402


FIELDS = (
    "Species",
    "Breed",
    "Age",
    "Weight",
    "Sex",
    "Location",
    "Presenting signs",
    "Vitals",
    "Microscopy",
    "Culture & sensitivity",
)


def load_records(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("task") == "T2_treatment" and row.get("image") is None]


def mask_field(text: str, field_name: str) -> str:
    pattern = re.compile(rf"^{re.escape(field_name)}:\s*.*$", flags=re.MULTILINE)
    return pattern.sub(f"{field_name}: [MASKED]", text, count=1)


def target_from_gold(text: str) -> str:
    match = re.search(r"^Diagnosis:\s*.+$", text, flags=re.MULTILINE)
    return match.group(0).strip() if match else "Diagnosis: insufficient evidence"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=16)
    parser.add_argument("--method", choices=("occlusion", "shap"), default="occlusion")
    parser.add_argument("--shap-samples", type=int, default=64)
    parser.add_argument("--allow-mps", action="store_true")
    args = parser.parse_args()

    model, processor, device, model_dtype = load_medgemma(args.adapter, args.allow_mps)
    records = load_records(args.input)[: args.max_examples]
    output_rows: list[dict[str, Any]] = []

    for index, row in enumerate(records, 1):
        user_text = row["messages"][0]["content"]
        target = target_from_gold(row["messages"][1]["content"])
        base_messages = [{"role": "user", "content": user_text}]
        baseline = score_target(model, processor, base_messages, target, device, model_dtype)
        if args.method == "occlusion":
            fields: dict[str, float] = {}
            for field_name in FIELDS:
                masked = mask_field(user_text, field_name)
                masked_score = score_target(
                    model,
                    processor,
                    [{"role": "user", "content": masked}],
                    target,
                    device,
                    model_dtype,
                )
                fields[field_name] = baseline - masked_score
        else:
            try:
                import numpy as np
                import shap
            except ImportError as exc:
                raise SystemExit("Install shap and numpy to use --method shap") from exc

            def score_masks(mask_matrix):
                scores = []
                for mask in mask_matrix:
                    masked_text = user_text
                    for field_index, field_name in enumerate(FIELDS):
                        if not bool(mask[field_index]):
                            masked_text = mask_field(masked_text, field_name)
                    scores.append(
                        score_target(
                            model,
                            processor,
                            [{"role": "user", "content": masked_text}],
                            target,
                            device,
                            model_dtype,
                        )
                    )
                return np.asarray(scores, dtype=float)

            explainer = shap.KernelExplainer(score_masks, np.zeros((1, len(FIELDS))))
            shap_values = explainer.shap_values(
                np.ones((1, len(FIELDS))),
                nsamples=args.shap_samples,
                silent=True,
            )
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            fields = {
                field_name: float(value)
                for field_name, value in zip(FIELDS, shap_values[0])
            }
        output_rows.append({
            "item_id": row["meta"]["case_id"],
            "group_id": row["meta"].get("case_id"),
            "provenance": row["meta"].get("provenance"),
            "clinical_use": row["meta"].get("clinical_use"),
            "method": f"field_{args.method}",
            "target": target,
            "baseline_score": baseline,
            "fields": fields,
        })
        print(f"attributed {index}/{len(records)}: {row['meta']['case_id']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
