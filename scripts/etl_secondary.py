#!/usr/bin/env python3
"""Run the provenance-first secondary-data metadata ETL.

This dependency-free stage validates manifests and writes a clean metadata
manifest. Pixel normalisation and distributed Spark execution are separate,
deliberate stages because neither should silently manufacture missing images.
Use `--spark-count` when PySpark is installed to demonstrate a Spark read/count
over the clean JSONL metadata.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_status(register: dict[str, Any]) -> dict[str, str]:
    return {str(row["source_id"]): str(row.get("eligibility_status", "")) for row in register.get("sources", [])}


def reject(row: dict[str, Any], statuses: dict[str, str], metadata_only: bool) -> str | None:
    source_id = row.get("source_id")
    if not source_id:
        return "missing source_id"
    status = statuses.get(str(source_id))
    if status is None:
        return "source_id not in source register"
    if status.startswith("excluded_") or row.get("provenance") in {"synthetic", "synthetic_generated", "local_unverified"}:
        return "prohibited provenance or excluded source"
    if not row.get("group_id"):
        return "missing group_id"
    if not metadata_only and row.get("image_path") and not Path(row["image_path"]).exists():
        return "image_path does not exist"
    if row.get("task") == "T2_treatment" and not (row.get("source_locator") or row.get("source_locators")):
        return "text item missing source locator"
    return None


def main() -> int:
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--image-manifest", type=Path)
    parser.add_argument("--text-manifest", type=Path)
    parser.add_argument("--source-register", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--spark-count", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config) if args.config else {}
    image_manifest = args.image_manifest or (ROOT / config.get("image_manifest", "data/secondary/manifests/images_raw.jsonl"))
    text_manifest = args.text_manifest or (ROOT / config.get("text_manifest", "data/secondary/text/items.jsonl"))
    register_path = args.source_register or (ROOT / config.get("source_register", "data/secondary/sources/source_register.json"))
    out_dir = args.out_dir or (ROOT / config.get("processed_dir", "data/secondary/processed"))
    register = read_json(register_path)
    statuses = source_status(register)
    rows = read_jsonl(image_manifest) + read_jsonl(text_manifest)
    clean: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for row in rows:
        if reason := reject(row, statuses, args.metadata_only):
            rejected[reason] += 1
        else:
            clean.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_path = out_dir / "clean_metadata.jsonl"
    clean_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in clean) + ("\n" if clean else ""), encoding="utf-8")
    report: dict[str, Any] = {
        "study_policy": config.get("study_policy", "secondary_data_only"),
        "image_input": str(image_manifest),
        "text_input": str(text_manifest),
        "records_input": len(rows),
        "records_clean": len(clean),
        "rejected": dict(rejected),
        "metadata_only": args.metadata_only,
        "spark_count": None,
        "status": "PASS" if not rejected else "REVIEW_REQUIRED",
    }
    if args.spark_count:
        try:
            from pyspark.sql import SparkSession
        except ImportError as exc:
            raise SystemExit("--spark-count requires pyspark; run the dependency-pinned ETL environment") from exc
        spark = SparkSession.builder.appName("secondary-metadata-etl").getOrCreate()
        try:
            report["spark_count"] = spark.read.json(str(clean_path)).count()
        finally:
            spark.stop()
    report["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    (out_dir / "etl_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
