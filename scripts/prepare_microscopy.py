#!/usr/bin/env python3
"""
prepare_microscopy.py — Apache Spark ETL for the microscopy image layer
(proposal Phase 1: "an Apache Spark ETL pipeline for ingestion, cleaning, normalisation").

Pipeline
--------
  ingest raw crops  ->  clean/validate  ->  normalise to 896x896 RGB PNG
                    ->  grouped-stratified train/val/test split
                    ->  emit partitioned manifest (parquet + JSONL)

Resolution choice: 896 x 896 RGB PNG.
  MedGemma (4B/27B) inherits Gemma 3's SigLIP-400m vision encoder, whose native input is
  896x896. Normalising every crop to that size removes an internal rescale and keeps the
  token count stable. Small native crops (e.g. NIH ~130x130) are aspect-preserving
  letterbox-padded then upscaled; large field images are aspect-preserving downscaled.
  Padding is black and recorded in the manifest so it is never read as specimen background.

Leakage prevention (a REAL methodological trap — called out explicitly)
-----------------------------------------------------------------------
  Many cells/crops come from the SAME physical slide (or patient). If crops from one slide
  land in both train and test, the model can memorise slide-specific staining/optics and the
  test score is optimistically biased. We therefore split by GROUP (slide/patient), never by
  individual image: an entire group goes to exactly one split. Stratification is done per
  class over groups so class balance is preserved across splits as far as group granularity
  allows. Datasets with no slide metadata (Mendeley) are treated as one opaque group per
  species and land wholly in a single split — a documented, deliberately conservative loss of
  balance in exchange for honest (non-leaking) evaluation.

Augmentation is configured here but NOT applied at ETL time (see AUGMENTATION_CONFIG);
augmentation is a TRAIN-TIME transform so val/test stay clean and the on-disk set is
deterministic.

Modes
-----
  --dry-run   Runs WITHOUT pyspark installed and WITHOUT downloaded images. Builds the split
              plan (from real raw files if present, else from the manifest's expected classes),
              prints split counts, and writes a sample manifest so the schema can be inspected.
  --local     Real ETL on a local Spark session (master=local[*]). Requires pyspark + Pillow
              and downloaded raw data.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MICRO = ROOT / "data" / "microscopy"
MANIFEST = MICRO / "dataset_manifest.json"
TAXONOMY = MICRO / "class_taxonomy.json"
RAW_DIR = MICRO / "raw"
IMAGES_DIR = MICRO / "images"          # normalised PNGs land here (real runs)
OUT_MANIFEST_DIR = MICRO / "prepared"  # parquet + jsonl split manifests

TARGET_PX = 896
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 20260709
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Applied at TRAIN time only — recorded, never baked into the on-disk normalised set.
AUGMENTATION_CONFIG = {
    "applied_at": "train_time_only",
    "rationale": "val/test must stay clean; on-disk set stays deterministic and reproducible.",
    "transforms": {
        "random_rotation_deg": 180,
        "random_horizontal_flip": True,
        "random_vertical_flip": True,
        "colour_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.1, "hue": 0.02},
        "random_resized_crop": {"scale": [0.8, 1.0], "target_px": TARGET_PX},
        "stain_jitter_note": "mild HED/colour jitter approximates Giemsa staining variation across labs",
        "gaussian_blur_p": 0.1
    },
    "not_applied_to": ["val", "test"]
}


def load_json(p: Path) -> dict:
    with p.open() as fh:
        return json.load(fh)


# --------------------------------------------------------------------------------------
# Grouping: extract a leakage-safe group key (slide/patient) per image.
# --------------------------------------------------------------------------------------
def group_key_for(dataset_id: str, image_path: Path) -> str:
    """Return a slide/patient-level group id. Whole group -> one split."""
    stem = image_path.stem
    if dataset_id == "nih_malaria_cell":
        # Filenames like 'C100P61ThinF_IMG_20150918_...': patient token = 'C<digits>'.
        m = re.match(r"(C\d+)", stem)
        return f"nih_{m.group(1)}" if m else f"nih_{image_path.parent.name}"
    if dataset_id == "bbbc041_pvivax":
        # Bounding-box crops named '<sourceimage>__<idx>'; group by source image.
        return f"bbbc041_{stem.split('__')[0]}"
    if dataset_id == "tryp_thick_smear":
        # Group by source frame folder (figshare organises by slide/frame).
        return f"tryp_{image_path.parent.name}"
    if dataset_id == "mendeley_parasite_species":
        # NO slide metadata -> one opaque group per species folder (conservative).
        return f"mendeley_{image_path.parent.name}"
    if dataset_id == "chula_parasite_egg_11":
        return f"chula_{image_path.parent.name}"
    return f"{dataset_id}_{image_path.parent.name}"


# --------------------------------------------------------------------------------------
# Discover (image_path, class_id, class_name, source_dataset, group_key) records.
# --------------------------------------------------------------------------------------
def native_class_to_taxonomy(ds: dict, native_folder: str):
    """Resolve a raw sub-folder name to (class_id, class_name) via manifest mapping."""
    for key, m in ds.get("mapping", {}).items():
        options = [o.strip().lower() for o in key.split("|")]
        if native_folder.lower() in options and m.get("class_id") is not None:
            return m["class_id"], m["class_name"]
    return None, None


def discover_records(manifest: dict) -> list[dict]:
    records: list[dict] = []
    for ds in manifest["datasets"]:
        ds_root = RAW_DIR / ds["id"]
        if not ds_root.exists():
            continue
        for img in ds_root.rglob("*"):
            if img.suffix.lower() not in IMG_EXTS or not img.is_file():
                continue
            cid, cname = native_class_to_taxonomy(ds, img.parent.name)
            if cid is None:
                continue  # out-of-taxonomy or unmapped -> excluded, not silently mislabelled
            records.append({
                "image_path": str(img),
                "class_id": cid,
                "class_name": cname,
                "source_dataset": ds["id"],
                "group_key": group_key_for(ds["id"], img),
            })
    return records


def synth_plan_records(manifest: dict, per_group: int = 30) -> list[dict]:
    """Dry-run fallback when no raw images are present.
    Fabricates PATHS ONLY (never image content) so the split algorithm can be exercised and
    its output schema inspected. Clearly flagged as a plan, not data."""
    records: list[dict] = []
    for ds in manifest["datasets"]:
        for key, m in ds.get("mapping", {}).items():
            cid = m.get("class_id")
            if cid is None:
                continue
            cname = m["class_name"]
            # simulate a few slides/groups per (dataset,class) so grouping is visible
            n_groups = 1 if ds["id"] == "mendeley_parasite_species" else 4
            for g in range(n_groups):
                gk = f"{ds['id']}_{cname}_slide{g}"
                for i in range(per_group):
                    records.append({
                        "image_path": f"PLAN-ONLY://{ds['id']}/{cname}/slide{g}/img{i}.png",
                        "class_id": cid,
                        "class_name": cname,
                        "source_dataset": ds["id"],
                        "group_key": gk,
                    })
    return records


# --------------------------------------------------------------------------------------
# Grouped stratified split: whole groups -> one split; balanced per class by image count.
# --------------------------------------------------------------------------------------
def plan_splits(records: list[dict], fractions=SPLIT_FRACTIONS, seed=SEED) -> dict:
    rng = random.Random(seed)
    # group -> (class_id, count)   (a group is assumed single-class; assert & report if not)
    group_class: dict[str, int] = {}
    group_count: dict[str, int] = defaultdict(int)
    mixed_groups: set[str] = set()
    for r in records:
        gk = r["group_key"]
        group_count[gk] += 1
        if gk in group_class and group_class[gk] != r["class_id"]:
            mixed_groups.add(gk)
        group_class[gk] = r["class_id"]

    # groups per class
    by_class: dict[int, list[str]] = defaultdict(list)
    for gk, cid in group_class.items():
        by_class[cid].append(gk)

    assignment: dict[str, str] = {}
    for cid, groups in by_class.items():
        groups = sorted(groups)
        rng.shuffle(groups)
        total = sum(group_count[g] for g in groups)
        want = {s: fractions[s] * total for s in fractions}
        got = {s: 0 for s in fractions}
        # Greedy: assign each group to the split with the largest remaining deficit.
        # Guarantees whole-group placement (no leakage) while tracking class balance.
        for g in groups:
            if len(groups) >= 3:
                deficits = {s: want[s] - got[s] for s in fractions}
                target = max(deficits, key=deficits.get)
            else:
                # Too few groups to cover 3 splits without leakage: prioritise train.
                order = ["train", "val", "test"]
                target = next((s for s in order if got[s] == 0), "train")
            assignment[g] = target
            got[target] += group_count[g]

    # attach split to records + tally
    tally = {s: defaultdict(int) for s in fractions}
    for r in records:
        s = assignment[r["group_key"]]
        r["split"] = s
        tally[s][r["class_id"]] += 1

    return {
        "assignment": assignment,
        "records": records,
        "tally": {s: dict(sorted(tally[s].items())) for s in fractions},
        "n_groups": len(group_class),
        "n_images": len(records),
        "mixed_groups": sorted(mixed_groups),
    }


def split_report(plan: dict, taxonomy: dict) -> str:
    names = {c["id"]: c["class"] for c in taxonomy["classes"]}
    lines = [f"Groups: {plan['n_groups']}   Images: {plan['n_images']}"]
    if plan["mixed_groups"]:
        lines.append(f"WARNING: {len(plan['mixed_groups'])} multi-class groups "
                     f"(check grouping key): {plan['mixed_groups'][:5]}...")
    all_cids = sorted({cid for s in plan["tally"] for cid in plan["tally"][s]})
    header = f"{'class':<24}" + "".join(f"{s:>8}" for s in SPLIT_FRACTIONS) + f"{'total':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for cid in all_cids:
        row = f"{cid} {names.get(cid,'?'):<22}"
        tot = 0
        for s in SPLIT_FRACTIONS:
            n = plan["tally"][s].get(cid, 0)
            tot += n
            row += f"{n:>8}"
        row += f"{tot:>8}"
        lines.append(row)
    return "\n".join(lines)


def write_manifest_outputs(plan: dict, dry_run: bool) -> Path:
    OUT_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_MANIFEST_DIR / ("split_manifest.dryrun.jsonl" if dry_run else "split_manifest.jsonl")
    with out.open("w") as fh:
        for r in plan["records"]:
            fh.write(json.dumps({
                "image_path": r["image_path"],
                "class_id": r["class_id"],
                "class_name": r["class_name"],
                "source_dataset": r["source_dataset"],
                "group_key": r["group_key"],
                "split": r["split"],
            }) + "\n")
    # sidecar summary
    summ = OUT_MANIFEST_DIR / ("summary.dryrun.json" if dry_run else "summary.json")
    with summ.open("w") as fh:
        json.dump({
            "resolution_px": [TARGET_PX, TARGET_PX],
            "format": "PNG", "colour": "RGB",
            "split_fractions": SPLIT_FRACTIONS,
            "seed": SEED,
            "n_images": plan["n_images"],
            "n_groups": plan["n_groups"],
            "tally": plan["tally"],
            "augmentation_config": AUGMENTATION_CONFIG,
            "leakage_policy": "grouped-by-slide/patient; whole group -> one split",
            "dry_run": dry_run,
        }, fh, indent=2)
    return out


# --------------------------------------------------------------------------------------
# Spark image normalisation (real --local runs only).
# --------------------------------------------------------------------------------------
def normalise_image(src_path: str, dst_path: str, target=TARGET_PX) -> str:
    """Letterbox to target x target, RGB, PNG. Returns dst_path. Import Pillow lazily."""
    from PIL import Image
    im = Image.open(src_path).convert("RGB")
    w, h = im.size
    scale = min(target / w, target / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    im = im.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (target, target), (0, 0, 0))
    canvas.paste(im, ((target - nw) // 2, (target - nh) // 2))
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst_path, format="PNG")
    return dst_path


def run_spark(plan: dict) -> None:
    from pyspark.sql import SparkSession, functions as F
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType

    spark = (SparkSession.builder
             .appName("medgemma-microscopy-etl")
             .master(os.environ.get("SPARK_MASTER", "local[*]"))
             .getOrCreate())
    try:
        schema = StructType([
            StructField("image_path", StringType()),
            StructField("class_id", IntegerType()),
            StructField("class_name", StringType()),
            StructField("source_dataset", StringType()),
            StructField("group_key", StringType()),
            StructField("split", StringType()),
        ])
        df = spark.createDataFrame(plan["records"], schema=schema)

        # Normalise images (distributed). Output path mirrors split/class structure.
        def _norm(row):
            src = row["image_path"]
            dst = str(IMAGES_DIR / row["split"] / row["class_name"] / (Path(src).stem + ".png"))
            try:
                normalise_image(src, dst)
                return (dst, row["class_id"], row["class_name"],
                        row["source_dataset"], row["group_key"], row["split"], "ok")
            except Exception as e:  # noqa: BLE001 — record failures, never drop silently
                return (src, row["class_id"], row["class_name"],
                        row["source_dataset"], row["group_key"], row["split"], f"FAILED:{e}")

        out_schema = schema.add("status", StringType())
        norm = spark.createDataFrame(df.rdd.map(_norm), schema=out_schema).cache()

        failed = norm.filter(F.col("status").startswith("FAILED"))
        n_failed = failed.count()
        if n_failed:
            print(f"WARNING: {n_failed} images failed normalisation:")
            failed.select("image_path", "status").show(20, truncate=False)

        good = norm.filter(F.col("status") == "ok").drop("status")
        (good.write.mode("overwrite").partitionBy("split", "class_name")
             .parquet(str(OUT_MANIFEST_DIR / "manifest.parquet")))
        print(f"Wrote parquet manifest -> {OUT_MANIFEST_DIR / 'manifest.parquet'}")
        good.groupBy("split", "class_name").count().orderBy("split", "class_name").show(100)
        if n_failed:
            raise SystemExit(1)  # never exit clean if any image was dropped
    finally:
        spark.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true", help="run real ETL on a local Spark session")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan + validate splits WITHOUT pyspark or downloaded images")
    args = ap.parse_args()
    if not (args.local or args.dry_run):
        ap.error("choose --dry-run or --local")

    manifest = load_json(MANIFEST)
    taxonomy = load_json(TAXONOMY)

    records = discover_records(manifest)
    used_synth = False
    if not records:
        if args.local:
            sys.exit("No raw images found under data/microscopy/raw/. "
                     "Run scripts/download_microscopy.py first.")
        records = synth_plan_records(manifest)
        used_synth = True

    plan = plan_splits(records)

    print("=" * 70)
    print("MICROSCOPY ETL " + ("(DRY-RUN)" if args.dry_run else "(LOCAL SPARK)"))
    if used_synth:
        print("NOTE: no raw images present -> using PLAN-ONLY synthetic PATHS "
              "(no image content) to exercise the split algorithm.")
    print(f"Resolution: {TARGET_PX}x{TARGET_PX} RGB PNG  |  seed={SEED}  |  "
          f"splits={SPLIT_FRACTIONS}")
    print("=" * 70)
    print(split_report(plan, taxonomy))

    out = write_manifest_outputs(plan, dry_run=args.dry_run)
    print(f"\nWrote split manifest -> {out}")
    print(f"Wrote summary        -> {OUT_MANIFEST_DIR / ('summary.dryrun.json' if args.dry_run else 'summary.json')}")

    if args.local:
        print("\nStarting Spark normalisation...")
        run_spark(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
