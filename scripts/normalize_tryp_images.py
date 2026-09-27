#!/usr/bin/env python3
"""Normalise Tryp images with macOS `sips` while preserving source metadata.

The source images are first resized to fit within 896x896 and then padded with
black to a square. No labels are changed. The generated manifest records the
transformation and processed path for the instruction builder.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_sips(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # sips creates the first output, preserving aspect ratio with -Z.
    subprocess.run(
        ["sips", "-Z", "896", str(source), "--out", str(destination)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Pad the resized image to the model square. Padding is recorded in the row.
    subprocess.run(
        ["sips", "-p", "896", "896", "--padColor", "000000", str(destination), "--out", str(destination)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["sips", "-s", "format", "png", str(destination), "--out", str(destination)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="process only the first N rows for a smoke test")
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        path = args.split_dir / f"{split}.jsonl"
        if path.exists():
            rows.extend(read_jsonl(path))
    if args.limit:
        rows = rows[:args.limit]
    prepared: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in rows:
        source = Path(row["image_path"])
        destination = args.out_dir / row["split"] / row["benchmark_label"] / f"{row['record_id']}.png"
        try:
            run_sips(source, destination)
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"{row.get('record_id')}: {exc}")
            continue
        prepared_row = dict(row)
        prepared_row.update({
            "processed_image_path": str(destination),
            "normalisation": {
                "target_px": [896, 896],
                "colour": "RGB_or_source_converted_by_sips",
                "resize": "aspect_preserving_fit",
                "padding": "black",
                "processor": "macOS_sips",
            },
        })
        prepared.append(prepared_row)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in prepared) + ("\n" if prepared else ""), encoding="utf-8")
    args.errors.write_text("\n".join(errors) + ("\n" if errors else ""), encoding="utf-8")
    result = {"input": len(rows), "prepared": len(prepared), "errors": len(errors), "status": "PASS" if not errors else "FAIL"}
    print(json.dumps(result, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
