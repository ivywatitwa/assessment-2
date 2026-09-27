#!/usr/bin/env python3
"""Score structured T2 claims against source-derived evidence claims.

Each prediction must contain `item_id` and `claims`; each claim must contain
`claim_id`, `text`, `supported` (boolean or null), and optionally `contradicted`.
This interface intentionally separates unsupported from contradicted output and
does not award a perfect score to an empty answer.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "item_id" not in row or "claims" not in row:
            raise ValueError(f"{path}:{line_no}: expected item_id and claims")
        rows.append(row)
    return rows


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        claims = row["claims"]
        if not isinstance(claims, list):
            raise ValueError(f"{row['item_id']}: claims must be a list")
        if not claims:
            counts["empty_outputs"] += 1
        for claim in claims:
            supported = claim.get("supported")
            contradicted = bool(claim.get("contradicted", False))
            if contradicted:
                counts["contradicted"] += 1
            elif supported is True:
                counts["supported"] += 1
            elif supported is False:
                counts["unsupported"] += 1
            else:
                counts["unscorable"] += 1
    eligible = counts["supported"] + counts["unsupported"] + counts["contradicted"]
    return {
        "items": len(rows),
        "claims": eligible + counts["unscorable"],
        "supported_claims": counts["supported"],
        "unsupported_claims": counts["unsupported"],
        "contradicted_claims": counts["contradicted"],
        "unscorable_claims": counts["unscorable"],
        "empty_outputs": counts["empty_outputs"],
        "source_supported_claim_precision": counts["supported"] / eligible if eligible else None,
        "unsupported_or_contradicted_rate": (counts["unsupported"] + counts["contradicted"]) / eligible if eligible else None,
        "status": "PASS" if rows else "NO_DATA",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(load(args.input))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
