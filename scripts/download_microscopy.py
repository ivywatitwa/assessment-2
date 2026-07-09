#!/usr/bin/env python3
"""
download_microscopy.py — Acquire the verified PUBLIC microscopy datasets listed in
data/microscopy/dataset_manifest.json into data/microscopy/raw/<dataset>/.

Design goals (per project brief):
  * Stream large files (no whole-file-in-RAM), show progress.
  * Idempotent / resumable: HTTP Range resume of partial downloads; skip completed files.
  * Verify SHA-256 where a checksum is publishable; otherwise record the observed digest
    to raw/<dataset>/OBSERVED_SHA256 so re-runs can detect silent corruption/upstream drift.
  * Gated sources (Kaggle, IEEE DataPort) are DETECTED and the user is given precise
    instructions — never a cryptic failure.
  * NEVER silently skip a failed download. Every dataset ends in exactly one state:
    OK / SKIPPED_UP_TO_DATE / MANUAL_REQUIRED / FAILED — printed in a final summary,
    and the process exits non-zero if anything FAILED.

This script deliberately does NOT fabricate any data. Datasets with no anonymous direct
URL (Kaggle mirror, IEEE DataPort) are reported as MANUAL_REQUIRED with copy-paste steps.

Usage:
  python scripts/download_microscopy.py --list
  python scripts/download_microscopy.py --dataset nih_malaria_cell
  python scripts/download_microscopy.py --all
  python scripts/download_microscopy.py --all --dry-run    # print plan, download nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANIFEST = ROOT / "data" / "microscopy" / "dataset_manifest.json"
RAW_DIR = ROOT / "data" / "microscopy" / "raw"

CHUNK = 1024 * 256  # 256 KiB streaming chunks

# Terminal states
OK = "OK"
SKIPPED = "SKIPPED_UP_TO_DATE"
MANUAL = "MANUAL_REQUIRED"
FAILED = "FAILED"


def load_manifest() -> dict:
    if not MANIFEST.exists():
        sys.exit(f"ERROR: manifest not found at {MANIFEST}")
    with MANIFEST.open() as fh:
        return json.load(fh)


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TiB"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def stream_download(url: str, dest: Path, expected_sha: str | None) -> None:
    """Resumable streaming download with Range support. Raises on any failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "medgemma-vet/1.0"})
    if existing:
        req.add_header("Range", f"bytes={existing}-")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = resp.status
            # If server ignored Range (200 not 206) restart from scratch.
            mode = "ab"
            if existing and status == 200:
                existing = 0
                mode = "wb"
            total = resp.length or 0
            grand = total + existing
            done = existing
            with tmp.open(mode) as out:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if grand:
                        pct = 100 * done / grand
                        print(f"\r    {human(done)}/{human(grand)} ({pct:4.1f}%)",
                              end="", flush=True)
                    else:
                        print(f"\r    {human(done)}", end="", flush=True)
            print()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} for {url}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"network error for {url}: {e.reason}") from e

    observed = sha256_of(tmp)
    if expected_sha:
        if observed.lower() != expected_sha.lower():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"checksum mismatch: expected {expected_sha}, got {observed}")
    tmp.rename(dest)
    (dest.parent / "OBSERVED_SHA256").write_text(f"{observed}  {dest.name}\n")


def handle_direct(ds: dict, dry_run: bool) -> tuple[str, str]:
    """Datasets exposing an anonymous direct URL."""
    name = ds["id"]
    urls = ds.get("download_urls") or ([ds["source_url"]] if ds.get("_direct") else [])
    if not urls:
        return MANUAL, "no anonymous direct URL"
    out_dir = RAW_DIR / name
    marker = out_dir / ".complete"
    if marker.exists():
        return SKIPPED, "already downloaded (.complete marker present)"
    if dry_run:
        return OK, f"[dry-run] would download {len(urls)} file(s) (~{ds.get('approx_download_mb','?')} MB)"
    try:
        for url in urls:
            fname = url.split("/")[-1].split("?")[0] or f"{name}.bin"
            dest = out_dir / fname
            if dest.exists():
                print(f"    {fname}: present, skipping")
                continue
            print(f"    fetching {url}")
            stream_download(url, dest, ds.get("sha256"))
        marker.write_text("ok\n")
        return OK, f"downloaded into {out_dir}"
    except Exception as e:  # noqa: BLE001 — surface EVERY failure, never swallow
        return FAILED, str(e)


def handle_kaggle(ds: dict, dry_run: bool) -> tuple[str, str]:
    """Kaggle-hosted mirror. Detect creds; instruct clearly rather than failing cryptically."""
    slug = ds["kaggle_mirror"].split("kaggle.com/datasets/")[-1].rstrip("/")
    out_dir = RAW_DIR / ds["id"]
    if (out_dir / ".complete").exists():
        return SKIPPED, "already downloaded"
    have_creds = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) \
        or (Path.home() / ".kaggle" / "kaggle.json").exists()
    try:
        import kaggle  # noqa: F401
        have_kaggle_pkg = True
    except Exception:  # noqa: BLE001
        have_kaggle_pkg = False

    if not (have_creds and have_kaggle_pkg):
        msg = (
            "Kaggle credentials/package required.\n"
            "      1) pip install kaggle\n"
            "      2) Create an API token at https://www.kaggle.com/settings -> 'Create New Token'\n"
            "         Save it to ~/.kaggle/kaggle.json (chmod 600) OR export KAGGLE_USERNAME / KAGGLE_KEY\n"
            f"      3) kaggle datasets download -d {slug} -p {out_dir} --unzip\n"
            f"      (Primary NIH source data.lhncbc.nlm.nih.gov is anonymous; use --dataset nih_malaria_cell instead if the NLM host is up.)"
        )
        return MANUAL, msg
    if dry_run:
        return OK, f"[dry-run] would run kaggle datasets download -d {slug}"
    try:
        import kaggle
        out_dir.mkdir(parents=True, exist_ok=True)
        kaggle.api.dataset_download_files(slug, path=str(out_dir), unzip=True, quiet=False)
        (out_dir / ".complete").write_text("ok\n")
        return OK, f"downloaded via Kaggle into {out_dir}"
    except Exception as e:  # noqa: BLE001
        return FAILED, f"kaggle download failed: {e}"


def handle_gated(ds: dict, _dry_run: bool) -> tuple[str, str]:
    """IEEE DataPort / registration-gated sources — always MANUAL with instructions."""
    msg = (
        f"Gated source — no anonymous download.\n"
        f"      Register + accept terms at: {ds['source_url']}\n"
        f"      Challenge page: {ds.get('challenge_page','(n/a)')}\n"
        f"      Then place the extracted files under: {RAW_DIR / ds['id']}/\n"
        f"      Licence must be confirmed on the landing page before any redistribution."
    )
    return MANUAL, msg


def route(ds: dict) -> str:
    if ds["id"] == "nih_malaria_cell":
        return "nih"
    if ds.get("kaggle_mirror") and not ds.get("download_urls"):
        return "kaggle"
    if ds.get("download_urls"):
        return "direct"
    if "ieee-dataport" in (ds.get("source_url") or ""):
        return "gated"
    if ds.get("figshare_doi"):
        return "figshare"
    if ds.get("mendeley_doi"):
        return "mendeley"
    return "gated"


def handle_nih(ds: dict, dry_run: bool) -> tuple[str, str]:
    ds = {**ds, "download_urls": [ds["source_url"]]}
    return handle_direct(ds, dry_run)


def handle_figshare(ds: dict, dry_run: bool) -> tuple[str, str]:
    # figshare DOIs resolve to an article; the file bundle needs the figshare API to
    # enumerate download URLs. We surface the DOI + API hint rather than guessing a URL.
    out_dir = RAW_DIR / ds["id"]
    if (out_dir / ".complete").exists():
        return SKIPPED, "already downloaded"
    if dry_run:
        return OK, f"[dry-run] would resolve figshare {ds['figshare_doi']} via API and stream files"
    try:
        article_id = ds["figshare_doi"].split(".")[-1]
        api = f"https://api.figshare.com/v2/articles/{article_id}/files"
        req = urllib.request.Request(api, headers={"User-Agent": "medgemma-vet/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            files = json.load(resp)
        for f in files:
            dest = out_dir / f["name"]
            if dest.exists():
                continue
            print(f"    fetching {f['name']} ({human(f.get('size', 0))})")
            stream_download(f["download_url"], dest, f.get("supplied_md5") and None)
        (out_dir / ".complete").write_text("ok\n")
        return OK, f"downloaded figshare bundle into {out_dir}"
    except Exception as e:  # noqa: BLE001
        return FAILED, (f"figshare API/download failed ({e}). "
                        f"Resolve manually at https://doi.org/{ds['figshare_doi']}")


def handle_mendeley(ds: dict, dry_run: bool) -> tuple[str, str]:
    """Mendeley Data — anonymously downloadable via its public API (enumerate then stream)."""
    out_dir = RAW_DIR / ds["id"]
    if (out_dir / ".complete").exists():
        return SKIPPED, "already downloaded"
    doi_id, version = ds["mendeley_doi"].split("/")[-1].split(".")[0], ds["mendeley_doi"].split(".")[-1]
    api = (f"https://data.mendeley.com/public-api/datasets/{doi_id}/files"
           f"?folder_id=root&version={version}")
    if dry_run:
        return OK, f"[dry-run] would enumerate {api} and stream files (~{ds.get('approx_download_mb','?')} MB)"
    try:
        req = urllib.request.Request(api, headers={"User-Agent": "medgemma-vet/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            files = json.load(resp)
        if not files:
            raise RuntimeError("Mendeley API returned no files")
        for f in files:
            url = f.get("content_details", {}).get("download_url") or f.get("download_url")
            dest = out_dir / f["filename"]
            if dest.exists():
                continue
            print(f"    fetching {f['filename']}")
            stream_download(url, dest, None)
        (out_dir / ".complete").write_text("ok\n")
        return OK, f"downloaded Mendeley bundle into {out_dir}"
    except Exception as e:  # noqa: BLE001
        return MANUAL, (f"Mendeley API auto-fetch unavailable ({e}). "
                        f"Download manually (anonymous, no login) from {ds['source_url']} "
                        f"-> 'Download All', then extract into {out_dir}/")


HANDLERS = {
    "nih": handle_nih,
    "direct": handle_direct,
    "kaggle": handle_kaggle,
    "gated": handle_gated,
    "figshare": handle_figshare,
    "mendeley": handle_mendeley,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="download one dataset by manifest id")
    ap.add_argument("--all", action="store_true", help="download all auto-fetchable datasets")
    ap.add_argument("--list", action="store_true", help="list datasets and their access mode")
    ap.add_argument("--dry-run", action="store_true", help="print plan, download nothing")
    args = ap.parse_args()

    manifest = load_manifest()
    datasets = {d["id"]: d for d in manifest["datasets"]}

    if args.list or not (args.dataset or args.all):
        print(f"Resolution target: {manifest['resolution_target']['pixels']} "
              f"{manifest['resolution_target']['format']}\n")
        print(f"{'id':<26}{'mode':<10}{'~MB':>8}  licence")
        print("-" * 90)
        for d in manifest["datasets"]:
            print(f"{d['id']:<26}{route(d):<10}{str(d.get('approx_download_mb','?')):>8}  "
                  f"{d['licence'][:44]}")
        if not (args.dataset or args.all):
            print("\nChoose --dataset <id> or --all. Add --dry-run to preview.")
        return 0

    targets = [datasets[args.dataset]] if args.dataset else manifest["datasets"]
    if args.dataset and args.dataset not in datasets:
        sys.exit(f"unknown dataset id: {args.dataset}")

    results: list[tuple[str, str, str]] = []
    for ds in targets:
        mode = route(ds)
        print(f"\n=== {ds['id']}  ({mode}) ===")
        state, detail = HANDLERS[mode](ds, args.dry_run)
        print(f"  -> {state}: {detail}")
        results.append((ds["id"], state, detail))

    print("\n" + "=" * 60 + "\nSUMMARY")
    any_failed = False
    for ds_id, state, _ in results:
        print(f"  {state:<20} {ds_id}")
        if state == FAILED:
            any_failed = True
    if any_failed:
        print("\nOne or more downloads FAILED (see above). Exiting non-zero.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
