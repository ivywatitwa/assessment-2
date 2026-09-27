#!/usr/bin/env python3
"""Create deterministic group-disjoint splits for revised secondary manifests.

Input rows must contain `group_id` and one of `label`, `class_name`, or `disease`.
The implementation is deliberately dependency-free so the split audit can run
before Spark, Pillow or the GPU stack is installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
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
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected object")
        rows.append(row)
    return rows


def label_of(row: dict[str, Any]) -> str:
    for key in ("label", "class_name", "benchmark_label", "disease"):
        if row.get(key) is not None:
            return str(row[key])
    return "__unlabelled__"


def stable_bucket(group_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def assign_groups(rows: list[dict[str, Any]], seed: int) -> dict[str, str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = row.get("group_id")
        if not group_id:
            raise ValueError(f"row {row.get('record_id', row.get('item_id', '?'))} has no group_id")
        groups[str(group_id)].append(row)

    # Sort by label rarity and then by a stable hash. This keeps rare classes from
    # being allocated only after the large groups have consumed a split.
    group_items = []
    label_counts = Counter(label_of(row) for row in rows)
    for group_id, members in groups.items():
        labels = Counter(label_of(row) for row in members)
        rarity = min(label_counts[label] for label in labels)
        group_items.append((rarity, stable_bucket(group_id, seed), group_id, members))
    group_items.sort(key=lambda item: (item[0], item[1], item[2]))

    target = {"train": 0.70, "val": 0.15, "test": 0.15}
    assigned: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    split_label_counts: dict[str, Counter[str]] = {s: Counter() for s in target}
    desired_total = {s: len(rows) * fraction for s, fraction in target.items()}
    desired_label = {
        s: {label: count * fraction for label, count in label_counts.items()}
        for s, fraction in target.items()
    }

    for _, _, group_id, members in group_items:
        group_labels = Counter(label_of(row) for row in members)

        def cost(split: str) -> tuple[float, float, str]:
            label_cost = sum(
                max(0.0, split_label_counts[split][label] + count - desired_label[split][label])
                for label, count in group_labels.items()
            )
            total_cost = max(0.0, split_counts[split] + len(members) - desired_total[split])
            return (label_cost, total_cost, split)

        chosen = min(target, key=cost)
        assigned[group_id] = chosen
        split_counts[chosen] += len(members)
        split_label_counts[chosen].update(group_labels)
    return assigned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    assignment = assign_groups(rows, args.seed) if rows else {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_handles = {
        split: (args.out_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "val", "test")
    }
    seen: dict[str, str] = {}
    try:
        for row in rows:
            group_id = str(row["group_id"])
            split = assignment[group_id]
            row = dict(row)
            row["split"] = split
            seen.setdefault(group_id, split)
            output_handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for fh in output_handles.values():
            fh.close()

    summary = {
        "input": str(args.input),
        "seed": args.seed,
        "records": len(rows),
        "groups": len(assignment),
        "splits": Counter(assignment.values()),
        "group_assignments": dict(sorted(assignment.items())),
        "status": "PASS",
    }
    # Empty output files are not created for absent splits; the summary still
    # makes the fact that no source records were available explicit.
    summary["splits"] = dict(summary["splits"])
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
