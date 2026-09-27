#!/usr/bin/env python3
"""Convert source-derived retrieval items into tiny literature-grounded T2 items.

The answer is assembled only from the cited KB chunks. This creates a development
smoke set; it is intentionally not presented as a sufficient clinical benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    queries = read_jsonl(args.queries)
    chunks = {row["doc_id"]: row for row in read_jsonl(args.chunks)}
    output: list[dict[str, Any]] = []
    for query in queries:
        selected = [chunks[doc_id] for doc_id in query["relevant_doc_ids"] if doc_id in chunks]
        if len(selected) != len(query["relevant_doc_ids"]):
            missing = set(query["relevant_doc_ids"]) - {row["doc_id"] for row in selected}
            raise SystemExit(f"{query['item_id']}: missing KB chunks: {sorted(missing)}")
        answer = "\n\n".join(row["text"] for row in selected)
        output.append({
            "item_id": query["item_id"],
            "task": "T2_treatment",
            "item_type": "literature_derived_item",
            "group_id": query["group_id"],
            "source_id": query["source_id"],
            "question": query["query"],
            "reference_answer": answer,
            "source_locator": query["source_locator"],
            "species": query.get("species"),
            "disease": query.get("disease"),
            "section": query.get("section"),
            "answerability": query.get("answerability"),
            "provenance": "literature_derived_item",
            "clinical_use": False,
            "evidence_status": "copied_from_curated_kb_chunks",
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in output) + ("\n" if output else ""), encoding="utf-8")
    print(json.dumps({"items": len(output), "status": "PASS"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
