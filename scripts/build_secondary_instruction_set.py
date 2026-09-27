#!/usr/bin/env python3
"""Render eligible revised secondary manifests into mixed-task JSONL records."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def reject(row: dict[str, Any]) -> str | None:
    if row.get("provenance") in {"synthetic", "synthetic_generated", "local_unverified"}:
        return "prohibited provenance"
    if str(row.get("image_path", "")).startswith("PLAN-ONLY://"):
        return "plan-only path"
    if row.get("intended_role") == "excluded" or row.get("eligibility_status", "").startswith("excluded_"):
        return "excluded source"
    return None


def build_image(row: dict[str, Any]) -> dict[str, Any]:
    label = row.get("benchmark_label") or row.get("class_name") or row.get("label")
    if not label:
        raise ValueError(f"image row {row.get('record_id')} has no label")
    return {
        "task": "T1_microscopy",
        "image": row.get("processed_image_path") or row.get("image_path"),
        "messages": [
            {"role": "user", "content": "Classify the organism status in this microscopy image."},
            {"role": "assistant", "content": str(label)},
        ],
        "meta": {
            "record_id": row.get("record_id"),
            "group_id": row.get("group_id"),
            "split": row.get("split"),
            "source_id": row.get("source_id"),
            "host_species": row.get("host_species"),
            "organism": row.get("organism"),
            "provenance": row.get("provenance", "public_dataset"),
            "annotation_source": row.get("annotation_source"),
        },
    }


def build_text(row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("reference_answer")
    if not answer:
        raise ValueError(f"text row {row.get('item_id')} has no reference_answer")
    question = row.get("question") or row.get("input_facts")
    if not question:
        raise ValueError(f"text row {row.get('item_id')} has no question/input")
    return {
        "task": "T2_treatment",
        "image": None,
        "messages": [
            {"role": "user", "content": str(question)},
            {"role": "assistant", "content": str(answer)},
        ],
        "meta": {
            "item_id": row.get("item_id"),
            "group_id": row.get("group_id"),
            "split": row.get("split"),
            "source_id": row.get("source_id"),
            "source_locator": row.get("source_locator"),
            "species": row.get("species"),
            "disease": row.get("disease"),
            "answerability": row.get("answerability"),
            "provenance": row.get("provenance", "literature_derived_item"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest", type=Path)
    parser.add_argument("--text-manifest", type=Path)
    parser.add_argument("--image-split-dir", type=Path)
    parser.add_argument("--text-split-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    image_rows = read_jsonl(args.image_manifest) if args.image_manifest else []
    text_rows = read_jsonl(args.text_manifest) if args.text_manifest else []
    if args.image_split_dir:
        image_rows = []
        for split in ("train", "val", "test"):
            for row in read_jsonl(args.image_split_dir / f"{split}.jsonl"):
                row["split"] = split
                image_rows.append(row)
    if args.text_split_dir:
        text_rows = []
        for split in ("train", "val", "test"):
            for row in read_jsonl(args.text_split_dir / f"{split}.jsonl"):
                row["split"] = split
                text_rows.append(row)
    records: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for row in image_rows:
        if reason := reject(row):
            exclusions[reason] += 1
            continue
        records.append(build_image(row))
    for row in text_rows:
        if reason := reject(row):
            exclusions[reason] += 1
            continue
        records.append(build_text(row))

    report = {
        "images_input": len(image_rows),
        "text_input": len(text_rows),
        "records_output": len(records),
        "task_counts": dict(Counter(r["task"] for r in records)),
        "excluded": dict(exclusions),
        "dry_run": args.dry_run,
        "status": "PASS",
    }
    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            (args.out_dir / f"{split}.jsonl").write_text("", encoding="utf-8")
        # Records must already carry a frozen split. Failing loudly prevents an
        # accidental random split at instruction-building time.
        missing_split = [r for r in records if not r["meta"].get("split")]
        if missing_split:
            report["status"] = "FAIL"
            report["error"] = "input records need a frozen split before instruction building"
            (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2))
            return 2
        handles = {
            split: (args.out_dir / f"{split}.jsonl").open("w", encoding="utf-8")
            for split in ("train", "val", "test")
        }
        try:
            for record in records:
                split = record["meta"]["split"]
                if split not in handles:
                    raise ValueError(f"unsupported split: {split}")
                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
        finally:
            for handle in handles.values():
                handle.close()
        report["written_splits"] = {
            split: sum(1 for record in records if record["meta"]["split"] == split)
            for split in ("train", "val", "test")
        }
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
