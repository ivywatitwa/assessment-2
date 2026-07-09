#!/usr/bin/env python3
"""
build_instruction_set.py — merge T1 (microscopy) + T2 (treatment) into ONE
instruction-tuning set, per the research proposal, Phase 3:

    "... instruction-response pairs covering both tasks, microscopy image plus
     disease target, and clinical text plus treatment target, mixed in a single
     training set via instruction tuning."

Reads:
    data/clinical_text/cases_{train,val,test}.jsonl        (T2 source)
    data/clinical_text/hallucination_probe.jsonl           (T2 probe, eval only)
    data/microscopy/splits/{train,val,test}.jsonl          (T1 source, from Spark ETL)

Writes:
    data/instruction_tuning/{train,val,test}.jsonl
    data/instruction_tuning/hallucination_probe.jsonl
    data/instruction_tuning/stats.json

Output conforms to PROJECT_SPEC.md §4.2.

NOTE ON SCOPE: this script does NOT invent data. If a source split is missing it
says so and skips it rather than silently emitting a half-set. A quietly
half-built training set is the kind of bug that costs you a month of confused
fine-tuning runs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")

# --------------------------------------------------------------------------
# Prompt templates.
#
# Deliberately several variants per task: a single fixed instruction string
# teaches the model to key off that exact string, which then collapses at
# inference when the phrasing differs. Variation is a regulariser, not decoration.
# --------------------------------------------------------------------------

T1_INSTRUCTIONS = [
    "Examine this veterinary microscopy image and identify the pathogen present.",
    "This is a slide from a {specimen}. Which pathogen, if any, is visible?",
    "Identify the organism in this microscopy field. If the sample is negative, say so.",
    "As a veterinary parasitologist, classify the pathogen shown in this image.",
]

T2_INSTRUCTIONS = [
    "Given the following clinical presentation, provide a diagnosis and treatment plan.",
    "Review these lab technician notes and recommend appropriate treatment.",
    "A {species} presents with the findings below. What is your assessment and recommended management?",
    "Based on the clinical notes and any laboratory results provided, give a diagnosis, "
    "differentials, and a treatment recommendation with dosing.",
]

REFUSAL_MARKER = "insufficient evidence"


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: malformed JSON — {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Rendering: case record -> chat messages
# --------------------------------------------------------------------------

def render_clinical_prompt(case: dict[str, Any]) -> str:
    """Render a case into the free-text clinical note a technician would actually write."""
    v = case.get("vitals") or {}
    parts = [
        f"Species: {case['species']}",
        f"Breed: {case.get('breed', 'unknown')}",
        f"Age: {case.get('age_months', '?')} months",
        f"Weight: {case.get('weight_kg', '?')} kg",
        f"Sex: {case.get('sex', 'unknown')}",
        f"Location: {case.get('county', 'unknown')} County, Kenya",
        "",
        "Presenting signs: " + ", ".join(case.get("presenting_signs", []) or ["none recorded"]),
    ]
    if v:
        parts.append(
            "Vitals: "
            + ", ".join(f"{k}={val}" for k, val in v.items())
        )

    micro = case.get("microscopy_finding")
    if micro:
        line = f"Microscopy: {micro.get('class', 'not performed')}"
        if micro.get("parasitaemia_pct") is not None:
            line += f" (parasitaemia {micro['parasitaemia_pct']}%)"
        parts.append(line)
    else:
        parts.append("Microscopy: not performed")

    cs = case.get("culture_sensitivity")
    if cs:
        parts.append(f"Culture & sensitivity: {json.dumps(cs, ensure_ascii=False)}")
    else:
        # The explicit absence is signal, not noise. The proposal's task is defined
        # as using C&S "when available" — the model must learn the unavailable case.
        parts.append("Culture & sensitivity: not available")

    return "\n".join(parts)


def render_treatment_answer(case: dict[str, Any]) -> str:
    """Render the target treatment plan."""
    tx = case.get("treatment") or {}

    # Referral / insufficient-evidence cases (hallucination probe) have no drug.
    if not tx.get("primary_drug"):
        return (
            f"Assessment: {case.get('diagnosis', 'Undetermined')}\n\n"
            f"There is {REFUSAL_MARKER} to commit to a specific pharmacological "
            f"treatment from the information provided.\n\n"
            f"Differentials to exclude: {', '.join(case.get('differentials', [])) or 'not established'}\n\n"
            f"Recommendation: {tx.get('referral', 'Refer to a licensed veterinary surgeon for physical examination and further diagnostics.')}\n\n"
            f"Rationale: {case.get('clinician_rationale', '')}"
        )

    lines = [
        f"Diagnosis: {case['diagnosis']}",
        "",
        f"Differentials considered: {', '.join(case.get('differentials', [])) or 'none'}",
        "",
        "Treatment:",
        f"  - Drug: {tx['primary_drug']}",
        f"  - Dose: {tx['dose_mg_per_kg']} mg/kg",
        f"  - Route: {tx['route']}",
        f"  - Frequency: {tx['frequency']}",
        f"  - Duration: {tx['duration_days']} day(s)",
    ]

    dose_mg = tx.get("dose_mg_per_kg")
    weight = case.get("weight_kg")
    if isinstance(dose_mg, (int, float)) and isinstance(weight, (int, float)):
        lines.append(f"  - Total dose for this {weight} kg animal: {round(dose_mg * weight, 1)} mg")

    if tx.get("supportive_care"):
        lines += ["", "Supportive care:"] + [f"  - {s}" for s in tx["supportive_care"]]

    if case.get("withdrawal_period_days") is not None:
        lines += ["", f"Withdrawal period: {case['withdrawal_period_days']} days "
                      f"(milk/meat must not enter the food chain before this elapses)."]

    if case.get("follow_up_days") is not None:
        lines.append(f"Follow-up: review in {case['follow_up_days']} days.")

    if case.get("clinician_rationale"):
        lines += ["", f"Rationale: {case['clinician_rationale']}"]

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Record builders
# --------------------------------------------------------------------------

def build_t2(case: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    instr = rng.choice(T2_INSTRUCTIONS).format(species=case.get("species", "animal"))
    return {
        "task": "T2_treatment",
        "image": None,
        "messages": [
            {"role": "user", "content": f"{instr}\n\n{render_clinical_prompt(case)}"},
            {"role": "assistant", "content": render_treatment_answer(case)},
        ],
        "meta": {
            "case_id": case["case_id"],
            "diagnosis": case.get("diagnosis"),
            "provenance": case.get("provenance", "synthetic"),
            "is_referral_case": not (case.get("treatment") or {}).get("primary_drug"),
        },
    }


def build_t1(img: dict[str, Any], rng: random.Random, taxonomy: dict[int, dict]) -> dict[str, Any]:
    cls_id = img["class_id"]
    cls = taxonomy.get(cls_id, {})
    specimen = cls.get("specimen", "slide")
    instr = rng.choice(T1_INSTRUCTIONS).format(specimen=specimen)

    class_name = img.get("class_name") or cls.get("class", "unknown")
    if class_name == "uninfected":
        answer = "No pathogen is visible in this field. The sample appears uninfected."
    else:
        disease = cls.get("disease")
        answer = f"Pathogen identified: {class_name}."
        if disease and disease != "—":
            answer += f" This organism is the causative agent of {disease}."

    return {
        "task": "T1_microscopy",
        "image": img["image_path"],
        "messages": [
            {"role": "user", "content": instr},
            {"role": "assistant", "content": answer},
        ],
        "meta": {
            "class_id": cls_id,
            "class_name": class_name,
            "source_dataset": img.get("source_dataset"),
            "provenance": "public_dataset",
            "is_pretraining_proxy": bool(cls.get("is_pretraining_proxy")),
        },
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def load_taxonomy(path: Path) -> dict[int, dict]:
    if not path.exists():
        print(f"  ! taxonomy not found at {path} — T1 answers will be less specific", file=sys.stderr)
        return {}
    raw = json.loads(path.read_text())
    classes = raw["classes"] if isinstance(raw, dict) and "classes" in raw else raw
    return {int(c["id"]): c for c in classes}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-proxy", action="store_true",
                    help="Drop the Plasmodium pre-training proxy class from val/test "
                         "(spec §2: it is excluded from the veterinary evaluation set).")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "instruction_tuning")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    taxonomy = load_taxonomy(REPO / "data" / "microscopy" / "class_taxonomy.json")

    stats: dict[str, Any] = {"seed": args.seed, "splits": {}, "missing_sources": []}

    for split in SPLITS:
        t2_src = REPO / "data" / "clinical_text" / f"cases_{split}.jsonl"
        t1_src = REPO / "data" / "microscopy" / "splits" / f"{split}.jsonl"

        cases = read_jsonl(t2_src)
        images = read_jsonl(t1_src)

        if not cases:
            stats["missing_sources"].append(str(t2_src.relative_to(REPO)))
        if not images:
            stats["missing_sources"].append(str(t1_src.relative_to(REPO)))

        rows = [build_t2(c, rng) for c in cases]

        for img in images:
            cls = taxonomy.get(img["class_id"], {})
            if args.exclude_proxy and split in ("val", "test") and cls.get("is_pretraining_proxy"):
                continue
            rows.append(build_t1(img, rng, taxonomy))

        # Interleave tasks. Contiguous blocks of one task correlate task identity
        # with training step, which interacts badly with LR warmup and can make the
        # model transiently "forget" whichever task it saw first.
        rng.shuffle(rows)

        write_jsonl(args.out / f"{split}.jsonl", rows)

        counts = Counter(r["task"] for r in rows)
        stats["splits"][split] = {
            "total": len(rows),
            "T1_microscopy": counts.get("T1_microscopy", 0),
            "T2_treatment": counts.get("T2_treatment", 0),
        }
        print(f"  {split:<6} {len(rows):>6} rows  "
              f"(T1={counts.get('T1_microscopy', 0)}, T2={counts.get('T2_treatment', 0)})")

    # Hallucination probe: eval-only, never mixed into train.
    probe_src = REPO / "data" / "clinical_text" / "hallucination_probe.jsonl"
    probe = read_jsonl(probe_src)
    if probe:
        probe_rows = [build_t2(c, rng) for c in probe]
        write_jsonl(args.out / "hallucination_probe.jsonl", probe_rows)
        stats["hallucination_probe"] = len(probe_rows)
        print(f"  probe  {len(probe_rows):>6} rows  (eval only — NOT in train)")
    else:
        stats["missing_sources"].append(str(probe_src.relative_to(REPO)))

    (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    if stats["missing_sources"]:
        print("\n  ! MISSING SOURCES (set is incomplete — do not train on this yet):", file=sys.stderr)
        for m in stats["missing_sources"]:
            print(f"      - {m}", file=sys.stderr)
        return 1

    # Leakage check: no case_id may appear in more than one split.
    seen: dict[str, str] = {}
    for split in SPLITS:
        for row in read_jsonl(args.out / f"{split}.jsonl"):
            cid = row["meta"].get("case_id")
            if cid is None:
                continue
            if cid in seen and seen[cid] != split:
                print(f"\n  ! LEAKAGE: {cid} in both {seen[cid]} and {split}", file=sys.stderr)
                return 1
            seen[cid] = split

    print("\n  OK — instruction set built, no case_id leakage detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
