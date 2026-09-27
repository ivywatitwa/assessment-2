#!/usr/bin/env python3
"""Create auditable image-tile occlusion attributions for the T1 task.

This is a model-agnostic fallback for the generative MedGemma output. It scores
the margin between the two fixed class strings and measures the margin change
when image tiles are masked. It is not labelled as Grad-CAM because no
unverified vision-token heatmap is produced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageStat

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xai_model_utils import load_medgemma, score_target  # noqa: E402


POSITIVE = "trypanosome_present"
NEGATIVE = "negative_control"


def load_records(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("task") == "T1_microscopy" and row.get("image")]


def load_boxes(path: Path | None) -> dict[str, list[list[float]]]:
    if path is None or not path.exists():
        return {}
    return {
        row["record_id"]: row.get("boxes_yolo_normalised", [])
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def masked_tile(image: Image.Image, row: int, col: int, grid_size: int) -> Image.Image:
    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    mean = tuple(int(value) for value in ImageStat.Stat(image).mean[:3])
    width, height = image.size
    left = col * width // grid_size
    top = row * height // grid_size
    right = (col + 1) * width // grid_size
    bottom = (row + 1) * height // grid_size
    draw.rectangle((left, top, right, bottom), fill=mean)
    return masked


def tile_overlaps_box(row: int, col: int, grid_size: int, box: list[float]) -> bool:
    cx, cy, width, height = box
    box_left, box_right = cx - width / 2, cx + width / 2
    box_top, box_bottom = cy - height / 2, cy + height / 2
    tile_left, tile_right = col / grid_size, (col + 1) / grid_size
    tile_top, tile_bottom = row / grid_size, (row + 1) / grid_size
    return not (
        tile_right <= box_left
        or tile_left >= box_right
        or tile_bottom <= box_top
        or tile_top >= box_bottom
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--allow-mps", action="store_true")
    args = parser.parse_args()
    if args.grid_size < 2:
        raise SystemExit("--grid-size must be at least 2")

    model, processor, device, model_dtype = load_medgemma(args.adapter, args.allow_mps)
    all_records = load_records(args.input)
    if args.max_examples and args.max_examples < len(all_records):
        negative_records = [row for row in all_records if row["messages"][1]["content"] == NEGATIVE]
        positive_records = [row for row in all_records if row["messages"][1]["content"] == POSITIVE]
        half = args.max_examples // 2
        records = negative_records[:half] + positive_records[: args.max_examples - half]
    else:
        records = all_records
    boxes_by_id = load_boxes(args.manifest)
    output_rows: list[dict[str, Any]] = []

    for index, row in enumerate(records, 1):
        image = Image.open(args.root / row["image"]).convert("RGB")
        prompt = row["messages"][0]["content"]

        def messages_for(candidate: Image.Image):
            return [{
                "role": "user",
                "content": [
                    {"type": "image", "image": candidate},
                    {"type": "text", "text": prompt},
                ],
            }]

        baseline_messages = messages_for(image)
        positive_score = score_target(model, processor, baseline_messages, POSITIVE, device, model_dtype)
        negative_score = score_target(model, processor, baseline_messages, NEGATIVE, device, model_dtype)
        baseline_margin = positive_score - negative_score
        tiles: list[dict[str, float | int]] = []

        for tile_row in range(args.grid_size):
            for tile_col in range(args.grid_size):
                occluded = masked_tile(image, tile_row, tile_col, args.grid_size)
                messages = messages_for(occluded)
                occluded_positive = score_target(model, processor, messages, POSITIVE, device, model_dtype)
                occluded_negative = score_target(model, processor, messages, NEGATIVE, device, model_dtype)
                occluded_margin = occluded_positive - occluded_negative
                tiles.append({
                    "row": tile_row,
                    "col": tile_col,
                    "delta": baseline_margin - occluded_margin,
                })

        boxes = boxes_by_id.get(row["meta"]["record_id"], [])
        positive_deltas = [max(float(tile["delta"]), 0.0) for tile in tiles]
        inside = [
            delta
            for tile, delta in zip(tiles, positive_deltas)
            if any(tile_overlaps_box(int(tile["row"]), int(tile["col"]), args.grid_size, box) for box in boxes)
        ]
        total_positive = sum(positive_deltas)
        output_rows.append({
            "record_id": row["meta"]["record_id"],
            "group_id": row["meta"]["group_id"],
            "gold_label": row["messages"][1]["content"],
            "method": "image_tile_occlusion",
            "grid_size": args.grid_size,
            "image_size": list(image.size),
            "baseline_positive_score": positive_score,
            "baseline_negative_score": negative_score,
            "baseline_margin": baseline_margin,
            "predicted_label": POSITIVE if baseline_margin >= 0 else NEGATIVE,
            "boxes_yolo_normalised": boxes,
            "tiles": tiles,
            "positive_attribution_mass_inside_boxes": sum(inside) / total_positive if total_positive else None,
        })
        print(f"attributed {index}/{len(records)}: {row['meta']['record_id']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
