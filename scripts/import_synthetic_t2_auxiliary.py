#!/usr/bin/env python3
"""Import the existing synthetic T2 cases as a labelled auxiliary corpus.

This script does not convert synthetic data into secondary evidence. It preserves
the generator's provenance flags and writes a separate output tree so synthetic
training/development records cannot be confused with public source-derived items.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def render_prompt(case: dict[str, Any]) -> str:
    vitals = case.get("vitals") or {}
    parts = [
        f"Species: {case.get('species', 'unknown')}",
        f"Breed: {case.get('breed', 'unknown')}",
        f"Age: {case.get('age_months', 'unknown')} months",
        f"Weight: {case.get('weight_kg', 'unknown')} kg",
        f"Sex: {case.get('sex', 'unknown')}",
        f"Location: {case.get('county', 'unknown')} County, Kenya",
        "Presenting signs: " + ", ".join(case.get("presenting_signs", []) or ["none recorded"]),
        "Vitals: " + ", ".join(f"{key}={value}" for key, value in vitals.items()) if vitals else "Vitals: not recorded",
    ]
    microscopy = case.get("microscopy_finding")
    parts.append(f"Microscopy: {json.dumps(microscopy, ensure_ascii=False) if microscopy else 'not performed'}")
    culture = case.get("culture_sensitivity")
    parts.append(f"Culture & sensitivity: {json.dumps(culture, ensure_ascii=False) if culture else 'not available'}")
    return "\n".join(parts)


def render_answer(case: dict[str, Any]) -> str:
    treatment = case.get("treatment") or {}
    diagnosis = case.get("diagnosis", "Undetermined")
    lines = [f"Diagnosis: {diagnosis}"]
    if not treatment.get("primary_drug"):
        lines.extend([
            "",
            "There is insufficient evidence to commit to a specific pharmacological treatment from the information provided.",
            "Recommendation: " + str(treatment.get("referral", "Refer for further veterinary assessment and diagnostics.")),
        ])
    else:
        lines.extend([
            "",
            "Treatment:",
            f"Drug: {treatment.get('primary_drug')}",
            f"Dose: {treatment.get('dose_mg_per_kg')} mg/kg",
            f"Route: {treatment.get('route')}",
            f"Frequency: {treatment.get('frequency')}",
            f"Duration: {treatment.get('duration_days')} day(s)",
        ])
        if treatment.get("supportive_care"):
            lines.append("Supportive care: " + "; ".join(treatment["supportive_care"]))
    if case.get("clinician_rationale"):
        lines.extend(["", "Rationale: " + str(case["clinician_rationale"])])
    return "\n".join(lines)


def convert(case: dict[str, Any], split: str, rng: random.Random) -> dict[str, Any]:
    if not (case.get("provenance") == "synthetic" and case.get("clinical_use") is False
            and case.get("requires_expert_validation") is True):
        raise ValueError(f"{case.get('case_id')}: synthetic provenance contract is not satisfied")
    instructions = [
        "Given the following synthetic research case, provide an assessment and treatment plan.",
        "Review this synthetic veterinary case and state the evidence-limited management recommendation.",
    ]
    return {
        "task": "T2_treatment",
        "image": None,
        "messages": [
            {"role": "user", "content": f"{rng.choice(instructions)}\n\n{render_prompt(case)}"},
            {"role": "assistant", "content": render_answer(case)},
        ],
        "meta": {
            "case_id": case["case_id"],
            "split": split,
            "diagnosis": case.get("diagnosis"),
            "species": case.get("species"),
            "provenance": "synthetic",
            "clinical_use": False,
            "requires_expert_validation": True,
            "evaluation_role": "synthetic_auxiliary_only",
            "source_id": "synthetic_clinical_cases_repository",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "clinical_text")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "secondary" / "instruction_tuning_synthetic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        source = args.input_dir / f"cases_{split}.jsonl"
        rows = [convert(case, split, rng) for case in read_jsonl(source)]
        write_jsonl(args.out_dir / f"{split}.jsonl", rows)
        counts[split] = len(rows)
    probe_path = args.input_dir / "hallucination_probe.jsonl"
    probe = [convert(case, "probe", rng) for case in read_jsonl(probe_path)]
    write_jsonl(args.out_dir / "hallucination_probe.jsonl", probe)
    report = {
        "source": str(args.input_dir),
        "output": str(args.out_dir),
        "seed": args.seed,
        "main_counts": counts,
        "probe_count": len(probe),
        "evaluation_role": "synthetic_auxiliary_only",
        "clinical_use": False,
        "requires_expert_validation": True,
        "secondary_evidence": False,
        "status": "PASS",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "stats.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
