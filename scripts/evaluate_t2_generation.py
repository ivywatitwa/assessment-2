#!/usr/bin/env python3
"""Evaluate bounded synthetic T2 generation field matches and referrals."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def norm(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.input)
    if any(row.get("clinical_use") is not False for row in rows):
        raise SystemExit("T2 generation evaluation requires synthetic records with clinical_use=false")

    diagnosis_matches = [norm(row.get("predicted_diagnosis")) == norm(row.get("gold_diagnosis")) for row in rows]
    drug_matches = [norm(row.get("predicted_drug")) == norm(row.get("gold_drug")) for row in rows]
    probe_rows = [row for row in rows if row.get("gold_referral")]
    referral_matches = [row.get("predicted_referral") is True for row in probe_rows]
    result = {
        "items": len(rows),
        "synthetic_only": True,
        "diagnosis_exact_match": sum(diagnosis_matches) / len(rows) if rows else None,
        "drug_exact_match": sum(drug_matches) / len(rows) if rows else None,
        "hallucination_probe_items": len(probe_rows),
        "hallucination_probe_referral_rate": sum(referral_matches) / len(probe_rows) if probe_rows else None,
        "empty_outputs": sum(not row.get("raw_output", "").strip() for row in rows),
        "status": "PASS" if rows else "NO_DATA",
        "warning": "Synthetic auxiliary result only; not clinical validation or source-supported factuality.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
