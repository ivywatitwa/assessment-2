#!/usr/bin/env python3
"""Evaluate saved occlusion-based XAI artifacts without inventing results."""
from __future__ import annotations

import argparse
import json
import statistics
import zipfile
from pathlib import Path
from typing import Any


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tile_overlaps_box(tile: dict[str, Any], grid_size: int, box: list[float]) -> bool:
    cx, cy, width, height = box
    box_left, box_right = cx - width / 2, cx + width / 2
    box_top, box_bottom = cy - height / 2, cy + height / 2
    tile_left, tile_right = tile["col"] / grid_size, (tile["col"] + 1) / grid_size
    tile_top, tile_bottom = tile["row"] / grid_size, (tile["row"] + 1) / grid_size
    return not (
        tile_right <= box_left
        or tile_left >= box_right
        or tile_bottom <= box_top
        or tile_top >= box_bottom
    )


def image_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    with_boxes = [row for row in rows if row.get("boxes_yolo_normalised")]
    masses = [row["positive_attribution_mass_inside_boxes"] for row in with_boxes
              if row.get("positive_attribution_mass_inside_boxes") is not None]
    pointing_hits = 0
    pointing_total = 0
    for row in with_boxes:
        tiles = row.get("tiles", [])
        if not tiles:
            continue
        top = max(tiles, key=lambda tile: tile.get("delta", float("-inf")))
        if top["delta"] <= 0:
            continue  # No tile positively supports the scored class.
        pointing_total += 1
        if any(tile_overlaps_box(top, row["grid_size"], box)
               for box in row["boxes_yolo_normalised"]):
            pointing_hits += 1
    return {
        "records": len(rows),
        "mean_positive_attribution_mass_inside_boxes": statistics.mean(masses) if masses else None,
        "pointing_game_accuracy": pointing_hits / pointing_total if pointing_total else None,
        "records_with_boxes": len(with_boxes),
        "records_with_defined_box_mass": len(masses),
        "records_with_positive_pointing_target": pointing_total,
        "method": "image_tile_occlusion",
    }


def load_archive(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with zipfile.ZipFile(path) as archive:
        def rows(name: str) -> list[dict[str, Any]]:
            return [json.loads(line) for line in archive.read(name).splitlines() if line.strip()]
        return (
            rows("xai_t2_results/t1_image_attributions.jsonl"),
            rows("xai_t2_results/t2_text_attributions.jsonl"),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--xai-dir", type=Path, default=Path("results/xai"))
    parser.add_argument("--image-input", type=Path)
    parser.add_argument("--text-input", type=Path)
    parser.add_argument("--archive", type=Path, help="Re-evaluate the saved xai-t2-results.zip without rerunning a model")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    image_rows, text_rows = load_archive(args.archive) if args.archive else (
        load_jsonl(args.image_input), load_jsonl(args.text_input))
    if image_rows or text_rows:
        text_methods = sorted({row.get("method", "unknown") for row in text_rows})
        result = {
            "config": str(args.config),
            "image": image_summary(image_rows),
            "text": {
                "records": len(text_rows),
                "mean_max_field_attribution": statistics.mean(
                    max((float(value) for value in row.get("fields", {}).values()), default=0.0)
                    for row in text_rows
                ) if text_rows else None,
                "method": text_methods[0] if len(text_methods) == 1 else text_methods,
            },
            "status": "PASS",
            "warning": "Small-sample attribution diagnostics; undefined image box scores are excluded, and no explanation faithfulness is established.",
        }
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0

    image_arrays = list(args.xai_dir.glob("*image*")) if args.xai_dir.exists() else []
    text_arrays = list(args.xai_dir.glob("*text*")) if args.xai_dir.exists() else []
    result = {
        "xai_dir": str(args.xai_dir),
        "image_artifacts": len(image_arrays),
        "text_artifacts": len(text_arrays),
        "requires": ["raw_attribution", "target_score", "perturbation_scores", "group_id"],
        "status": "READY" if image_arrays and text_arrays else "BLOCKED_NO_ATTRIBUTION_ARTIFACTS",
    }
    print(json.dumps(result, indent=2))
    if args.run:
        raise SystemExit("XAI evaluation is blocked until attribution and perturbation artifacts exist.")
    return 0 if result["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
