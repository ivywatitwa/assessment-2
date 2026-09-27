#!/usr/bin/env python3
"""Parse a downloaded Tryp archive into an auditable image manifest.

The parser consumes the publisher's existing YOLO labels and filenames. It never
creates labels by visual inspection and it never assigns livestock or Kenyan
metadata that the source did not publish. The archive itself is intentionally not
downloaded by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

VIDEO_RE = re.compile(r"^(?P<kind>positive|negative)_video_(?P<number>\d+)", re.I)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_video_id(path: Path) -> str | None:
    match = VIDEO_RE.match(path.stem)
    if not match:
        return None
    return f"{match.group('kind').lower()}_video_{int(match.group('number')):03d}"


def parse_yolo_label(path: Path) -> list[list[float]]:
    boxes: list[list[float]] = []
    if not path.exists():
        return boxes
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_no}: expected class x_center y_center width height")
        try:
            _, x_center, y_center, width, height = (float(value) for value in fields)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: non-numeric YOLO field") from exc
        if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
            raise ValueError(f"{path}:{line_no}: normalised coordinates outside [0, 1]")
        boxes.append([x_center, y_center, width, height])
    return boxes


def image_rows(root: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    exclusions: list[str] = []
    seen_hashes: dict[str, Path] = {}
    for image in sorted(root.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        video_id = parse_video_id(image)
        if video_id is None:
            errors.append(f"{image}: source video id not found in filename")
            continue
        label_path = image.parent.parent / "labels" / f"{image.stem}.txt"
        try:
            boxes = parse_yolo_label(label_path)
            image_hash = sha256(image)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            continue
        is_negative_partition = "negative_images" in image.parts
        if image_hash in seen_hashes:
            exclusions.append(f"{image}: duplicate bytes of {seen_hashes[image_hash]}; excluded")
            continue
        seen_hashes[image_hash] = image
        if not boxes and not is_negative_partition:
            exclusions.append(f"{image}: positive-partition image has no parasite boxes; excluded")
            continue
        benchmark_label = "trypanosome_present" if boxes else "negative_control"
        rows.append({
            "record_id": f"tryp-{image_hash[:16]}",
            "source_id": "tryp_figshare_v1",
            "source_version": "v1",
            "image_path": str(image),
            "sha256": image_hash,
            "native_label": "trypanosome_positive" if boxes else "negative_control",
            "benchmark_label": benchmark_label,
            "organism": "Trypanosoma brucei brucei" if boxes else None,
            "host_species": "mouse",
            "specimen": "unstained thick blood smear",
            "video_id": video_id,
            "group_id": video_id,
            "boxes_yolo_normalised": boxes,
            "annotation_source": "publisher_yolo_labels",
            "provenance": "public_dataset",
            "eligibility_status": "eligible_public_secondary_data",
        })
    return rows, errors, exclusions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="extracted Tryp positive/negative image root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    args = parser.parse_args()
    rows, errors, exclusions = image_rows(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    diagnostics = [f"EXCLUDED: {message}" for message in exclusions] + [f"ERROR: {message}" for message in errors]
    args.errors.write_text("\n".join(diagnostics) + ("\n" if diagnostics else ""), encoding="utf-8")
    summary = {
        "records": len(rows),
        "errors": len(errors),
        "exclusions": len(exclusions),
        "status": "PASS_WITH_EXCLUSIONS" if not errors and exclusions else ("PASS" if not errors else "FAIL"),
    }
    print(json.dumps(summary, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
