#!/usr/bin/env python3
"""Run the registered-source metadata eligibility ETL with real local PySpark.

Keeps Spark outputs separate from the dependency-free metadata ETL. Image bytes
are not loaded; with --metadata-only, image paths are not checked on disk.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest", type=Path, default=ROOT / "data/secondary/manifests/images_raw.jsonl")
    parser.add_argument("--text-manifest", type=Path, default=ROOT / "data/secondary/text/items.jsonl")
    parser.add_argument("--source-register", type=Path, default=ROOT / "data/secondary/sources/source_register.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/secondary/spark_processed")
    parser.add_argument("--master", default="local[2]")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    for source in (args.image_manifest, args.text_manifest, args.source_register):
        if not source.is_file():
            parser.error(f"Missing source: {source}")

    try:
        import pyspark
        from pyspark.sql import SparkSession, functions as F
    except ImportError as exc:
        parser.error("Install pyspark to execute this stage (pip install pyspark==3.5.8)")

    statuses = json.loads(args.source_register.read_text(encoding="utf-8"))["sources"]
    sources = [(str(row["source_id"]), str(row.get("eligibility_status", ""))) for row in statuses]
    start = time.perf_counter()
    spark = SparkSession.builder.master(args.master).appName("vet-secondary-metadata-etl").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        records = spark.read.json([str(args.image_manifest), str(args.text_manifest)])
        registered = spark.createDataFrame(sources, ["registered_source_id", "source_status"])
        joined = records.join(F.broadcast(registered), records.source_id == registered.registered_source_id, "left")
        absent = lambda column: F.col(column).isNull() | (F.trim(F.col(column)) == "")
        reject = (
            F.when(absent("source_id"), "missing source_id")
            .when(F.col("source_status").isNull(), "source_id not in source register")
            .when(
                F.col("source_status").startswith("excluded_")
                | F.col("provenance").isin("synthetic", "synthetic_generated", "local_unverified"),
                "prohibited provenance or excluded source",
            )
            .when(absent("group_id"), "missing group_id")
        )
        if not args.metadata_only:
            # Python UDF checks filesystem paths, preserving the dependency-free
            # ETL's optional existence rule for locally available image files.
            from pyspark.sql.types import BooleanType

            exists = F.udf(lambda value: not value or (ROOT / value).exists(), BooleanType())
            reject = reject.when(~exists(F.col("image_path")), "image_path does not exist")
        locators_missing = (
            F.col("source_locators").isNull()
            if "source_locators" in records.columns else F.lit(True)
        )
        reject = reject.when(
            (F.col("task") == "T2_treatment")
            & absent("source_locator")
            & locators_missing,
            "text item missing source locator",
        )
        checked = joined.withColumn("rejection_reason", reject.otherwise(F.lit(None).cast("string"))).cache()
        counts = checked.groupBy("rejection_reason").count().collect()
        rejected = {row["rejection_reason"]: row["count"] for row in counts if row["rejection_reason"] is not None}
        clean_count = next((row["count"] for row in counts if row["rejection_reason"] is None), 0)
        total = sum(row["count"] for row in counts)

        out_dir = args.out_dir.resolve()
        if out_dir in (ROOT.resolve(), args.image_manifest.parent.resolve(), args.text_manifest.parent.resolve()):
            raise ValueError("Output directory must not overlap the input directories")
        out_dir.mkdir(parents=True, exist_ok=True)
        clean_dir = out_dir / "clean_metadata_json"
        checked.filter(F.col("rejection_reason").isNull()).drop(
            "rejection_reason", "registered_source_id", "source_status"
        ).coalesce(1).write.mode("overwrite").json(str(clean_dir))
        written = spark.read.json(str(clean_dir)).count() if clean_count else 0
        if written != clean_count:
            raise RuntimeError(f"Spark output count mismatch: expected {clean_count}, got {written}")
        report = {
            "engine": "pyspark", "pyspark_version": pyspark.__version__, "master": args.master,
            "image_input": str(args.image_manifest), "text_input": str(args.text_manifest),
            "records_input": total, "records_clean": clean_count, "records_written": written,
            "rejected": rejected, "metadata_only": args.metadata_only,
            "output": str(clean_dir), "elapsed_seconds": round(time.perf_counter() - start, 3),
            "status": "PASS" if not rejected else "REVIEW_REQUIRED",
        }
        (out_dir / "spark_etl_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "PASS" else 2
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
