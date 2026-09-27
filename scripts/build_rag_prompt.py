#!/usr/bin/env python3
"""Build a grounded treatment prompt for one stored clinical case.

This is intentionally model-agnostic: it verifies the RAG layer before adding
MedGemma inference. It requires the Chroma index to have been built first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_instruction_set import read_jsonl, render_clinical_prompt  # noqa: E402
from query_rag import query  # noqa: E402


def find_case(path: Path, case_id: str) -> dict:
    for row in read_jsonl(path):
        if row.get("case_id") == case_id:
            return row
    raise SystemExit(f"Case not found: {case_id} in {path}")


def make_prompt(case: dict, hits: list[dict]) -> str:
    references = "\n\n".join(
        f"[{i}] {hit['text']}\nSource: {hit['citation']}"
        for i, hit in enumerate(hits, 1)
    )
    return (
        "You are assisting a licensed veterinarian. This is a research-only draft, "
        "not veterinary advice. Use only the case evidence and references below. "
        "If evidence is insufficient, say 'insufficient evidence' and recommend referral. "
        "Do not invent a drug, dose, route, duration, or withdrawal period. Cite [N] "
        "after each recommendation.\n\n"
        f"CASE:\n{render_clinical_prompt(case)}\n\n"
        f"REFERENCES:\n{references}\n\n"
        "Return: assessment, differentials, treatment or referral, safety/withdrawal notes, "
        "and follow-up."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    parser.add_argument("--split", choices=("train", "val", "test", "probe"), default="test")
    parser.add_argument("-k", "--top-k", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    filename = "hallucination_probe.jsonl" if args.split == "probe" else f"cases_{args.split}.jsonl"
    case = find_case(ROOT / "data" / "clinical_text" / filename, args.case_id)
    hits = query(
        render_clinical_prompt(case),
        args.top_k,
        {"$and": [{"species_" + case["species"]: True}, {"section": "treatment"}]},
    )
    output = {"case_id": args.case_id, "references": hits, "prompt": make_prompt(case, hits)}
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(output["prompt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
