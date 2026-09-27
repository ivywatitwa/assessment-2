#!/usr/bin/env python3
"""Merge public task records with the explicitly labelled synthetic T2 corpus.

Public and synthetic records remain identifiable in metadata. The merger is for
mixed-task model training; evaluators must still report public and synthetic
results separately.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(record: dict[str, Any], expected_split: str) -> None:
    meta = record.get("meta") or {}
    if meta.get("split") != expected_split:
        raise ValueError(f"record {meta.get('case_id', meta.get('item_id'))} has incorrect split metadata")
    if meta.get("provenance") in {"local_unverified", "synthetic_generated"}:
        raise ValueError("unapproved provenance in mixed instruction set")
    if meta.get("provenance") == "synthetic":
        if meta.get("clinical_use") is not False or meta.get("requires_expert_validation") is not True:
            raise ValueError("synthetic record lost its safety flags")
        if meta.get("evaluation_role") != "synthetic_auxiliary_only":
            raise ValueError("synthetic record is missing evaluation_role=synthetic_auxiliary_only")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--synthetic-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"splits": {}, "status": "PASS"}
    for split in ("train", "val", "test"):
        public = read(args.public_dir / f"{split}.jsonl")
        synthetic = read(args.synthetic_dir / f"{split}.jsonl")
        records = public + synthetic
        for record in records:
            validate(record, split)
        (args.out_dir / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""),
            encoding="utf-8",
        )
        stats["splits"][split] = {
            "total": len(records),
            "public": len(public),
            "synthetic_auxiliary": len(synthetic),
            "tasks": dict(Counter(record.get("task") for record in records)),
        }
    stats["synthetic_is_secondary_evidence"] = False
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
