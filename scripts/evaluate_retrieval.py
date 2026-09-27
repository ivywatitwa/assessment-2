#!/usr/bin/env python3
"""Evaluate retrieval against source-derived relevant document IDs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from query_rag import build_where, query  # noqa: E402


def load(path: Path) -> list[dict[str, Any]]:
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


def evaluate(rows: list[dict[str, Any]], k: int, query_fn: Callable = query) -> dict[str, Any]:
    if k < 1 or not rows:
        raise ValueError("Evaluation requires k >= 1 and at least one query")
    details: list[dict[str, Any]] = []
    recall_values: list[float] = []
    reciprocal_values: list[float] = []
    for row in rows:
        relevant = set(row.get("relevant_doc_ids", []))
        if not relevant:
            raise ValueError(f"{row.get('item_id')} has no relevant_doc_ids")
        where = build_where(row.get("disease"), row.get("species"), row.get("section"))
        hits = query_fn(str(row["query"]), k, where)
        ids = [h.get("doc_id") for h in hits]
        found = relevant.intersection(ids)
        recall = len(found) / len(relevant)
        reciprocal = next((1.0 / (index + 1) for index, doc_id in enumerate(ids) if doc_id in relevant), 0.0)
        recall_values.append(recall)
        reciprocal_values.append(reciprocal)
        details.append({
            "item_id": row.get("item_id"),
            "section": row.get("section"),
            "relevant_doc_ids": sorted(relevant),
            "returned_doc_ids": ids,
            "recall_at_k": recall,
            "reciprocal_rank": reciprocal,
        })
    return {
        "items": len(rows),
        "k": k,
        "recall_at_k": sum(recall_values) / len(recall_values) if rows else None,
        "mrr": sum(reciprocal_values) / len(reciprocal_values) if rows else None,
        "details": details,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "secondary" / "reports" / "retrieval_eval.json")
    args = parser.parse_args()
    result = evaluate(load(args.input), args.k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
