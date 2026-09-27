#!/usr/bin/env python3
"""Export the public T1 splits in MLX-VLM's images/messages JSONL schema."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = ("negative_control", "trypanosome_present")
PROMPT = (
    "Inspect the entire bright-field blood-smear microscopy image for "
    "Trypanosoma parasites. Look for an elongated, curved protozoan with an "
    "undulating membrane or flagellum among the blood cells; ignore stain "
    "artifacts, debris, and filenames. Reply with exactly one label: "
    "`trypanosome_present` if at least one Trypanosoma is visible, or "
    "`negative_control` if none is visible. Do not add an explanation."
)


def read_t1(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [
        row for row in rows
        if row.get("task") == "T1_microscopy" and row.get("image")
    ]
    if not records:
        raise ValueError(f"No T1 image records in {path}")
    for row in records:
        label = row["messages"][1]["content"]
        if label not in LABELS:
            raise ValueError(f"Unexpected T1 label {label!r} in {path}")
    return records


def balance(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    by_label = {label: [] for label in LABELS}
    for row in records:
        by_label[row["messages"][1]["content"]].append(row)
    if any(not rows for rows in by_label.values()):
        raise ValueError("Both T1 labels must occur in the training split")

    rng = random.Random(seed)
    for rows in by_label.values():
        rng.shuffle(rows)
    per_class = max(len(rows) for rows in by_label.values())
    return [
        by_label[label][i % len(by_label[label])]
        for i in range(per_class)
        for label in LABELS
    ]


def mlx_record(root: Path, row: dict[str, Any], sample_index: int) -> dict[str, Any]:
    image = row["image"]
    image_path = root / image
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    label = row["messages"][1]["content"]
    meta = row.get("meta", {})
    return {
        # mlx-vlm LORA.MD expects these two columns.
        "images": [image],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}],
            },
        ],
        # Keep source traceability as additional JSON columns.
        "task": "T1_microscopy",
        "sample_index": sample_index,
        "source_record_id": meta.get("record_id"),
        "group_id": meta.get("group_id"),
        "source_split": meta.get("split"),
        "label": label,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/secondary/mlx_t1_minicpmv46"),
        help="output directory, relative to --root unless absolute",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    source_dir = root / "data/secondary/instruction_tuning_mixed"
    source = {
        split: read_t1(source_dir / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    output.mkdir(parents=True, exist_ok=True)

    balanced_train = balance(source["train"], args.seed)
    exported = {
        "train": balanced_train,
        "val": source["val"],
        "test": source["test"],
    }
    for split, records in exported.items():
        rows = [mlx_record(root, row, i) for i, row in enumerate(records)]
        write_jsonl(output / f"{split}.jsonl", rows)
        print(
            f"{split}: {len(rows)} rows; "
            f"{dict(Counter(row['label'] for row in rows))}; "
            f"saved {output / f'{split}.jsonl'}"
        )

    (output / "dataset_info.json").write_text(
        json.dumps(
            {
                "schema": "mlx-vlm-lora-images-messages-jsonl",
                "model_id": "openbmb/MiniCPM-V-4.6",
                "prompt": PROMPT,
                "seed": args.seed,
                "train_class_balanced": True,
                "image_paths": "repository-relative; run MLX jobs with cwd at --root",
                "source": "data/secondary/instruction_tuning_mixed/{train,val,test}.jsonl",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
