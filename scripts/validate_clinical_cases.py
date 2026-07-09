#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_clinical_cases.py
==========================

Standalone validator for the SYNTHETIC clinical-text dataset (spec §4.1). It re-reads the
JSONL split files independently of the generator and asserts:

  * schema conformance (all §4.1 fields, correct types/ranges)
  * §0 provenance fields present on every record
  * weight-vs-species/age plausibility
  * culture & sensitivity null-rule (populated only for Bovine Mastitis)
  * no case_id collisions across the train/val/test splits
  * priority-four present in all splits; no test diagnosis unseen in train
  * probe records recommend a REFERRAL (never a drug)

It prints a class-distribution / balance report and a PASS/FAIL summary table.

    python3 scripts/validate_clinical_cases.py --dir data/clinical_text

-------------------------------------------------------------------------------------------
   SYNTHETIC PRE-VALIDATION DATA — NOT VETERINARY ADVICE. Requires Cherehani Labs sign-off.
-------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

# Reuse the canonical constants and per-record validator from the generator so the two
# scripts cannot drift apart.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_clinical_cases import (  # noqa: E402
    PRIORITY_FOUR,
    SchemaError,
    validate_record,
    weight_plausible,
)

SPLIT_FILES = {
    "train": "cases_train.jsonl",
    "val": "cases_val.jsonl",
    "test": "cases_test.jsonl",
}
PROBE_FILE = "hallucination_probe.jsonl"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{path}:{ln} invalid JSON: {exc}")
    return records


def check_records(records: List[Dict[str, Any]], is_probe: bool) -> List[str]:
    errors: List[str] = []
    for rec in records:
        try:
            validate_record(rec, is_probe=is_probe)
        except SchemaError as exc:
            errors.append(str(exc))
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate synthetic clinical-case JSONL files.")
    ap.add_argument("--dir", default="data/clinical_text", help="directory with the JSONL files")
    args = ap.parse_args()

    results: List[Tuple[str, bool, str]] = []   # (check name, passed, detail)
    all_ok = True

    def record(name: str, passed: bool, detail: str = "") -> None:
        nonlocal all_ok
        results.append((name, passed, detail))
        if not passed:
            all_ok = False

    # ---- load ---------------------------------------------------------------------------
    splits: Dict[str, List[Dict[str, Any]]] = {}
    for split, fname in SPLIT_FILES.items():
        path = os.path.join(args.dir, fname)
        if not os.path.exists(path):
            record(f"file present: {fname}", False, "MISSING")
            continue
        splits[split] = load_jsonl(path)
        record(f"file present: {fname}", True, f"{len(splits[split])} records")

    probe_path = os.path.join(args.dir, PROBE_FILE)
    probe: List[Dict[str, Any]] = []
    if os.path.exists(probe_path):
        probe = load_jsonl(probe_path)
        record(f"file present: {PROBE_FILE}", True, f"{len(probe)} records")
    else:
        record(f"file present: {PROBE_FILE}", False, "MISSING")

    # ---- per-record schema --------------------------------------------------------------
    for split, recs in splits.items():
        errs = check_records(recs, is_probe=False)
        record(f"schema conformance: {split}", not errs,
               "all valid" if not errs else f"{len(errs)} error(s): {errs[:3]}")

    probe_errs = check_records(probe, is_probe=True)
    record("schema conformance: probe", not probe_errs,
           "all valid" if not probe_errs else f"{len(probe_errs)} error(s): {probe_errs[:3]}")

    # ---- provenance on every record -----------------------------------------------------
    prov_bad = 0
    for recs in list(splits.values()) + [probe]:
        for r in recs:
            if not (r.get("provenance") == "synthetic" and r.get("clinical_use") is False
                    and r.get("requires_expert_validation") is True):
                prov_bad += 1
    record("provenance fields on every record", prov_bad == 0,
           "all present" if prov_bad == 0 else f"{prov_bad} record(s) missing provenance")

    # ---- weight plausibility ------------------------------------------------------------
    w_bad = 0
    for recs in list(splits.values()) + [probe]:
        for r in recs:
            if not weight_plausible(r["species"], r["age_months"], r["weight_kg"]):
                w_bad += 1
    record("weight-vs-species/age plausibility", w_bad == 0,
           "all plausible" if w_bad == 0 else f"{w_bad} implausible weight(s)")

    # ---- C&S null-rule ------------------------------------------------------------------
    cs_bad: List[str] = []
    for recs in list(splits.values()) + [probe]:
        for r in recs:
            has_cs = r.get("culture_sensitivity") is not None
            is_mastitis = r.get("diagnosis") == "Bovine Mastitis"
            if has_cs and not is_mastitis:
                cs_bad.append(f"{r['case_id']} has C&S but is not mastitis")
            if is_mastitis and not has_cs:
                cs_bad.append(f"{r['case_id']} is mastitis but lacks C&S")
    record("C&S null-rule (mastitis-only)", not cs_bad,
           "correct" if not cs_bad else f"{len(cs_bad)} violation(s): {cs_bad[:3]}")

    # ---- no case_id collisions across splits --------------------------------------------
    seen: Dict[str, str] = {}
    collisions: List[str] = []
    for split, recs in splits.items():
        for r in recs:
            cid = r["case_id"]
            if cid in seen:
                collisions.append(f"{cid} in both {seen[cid]} and {split}")
            else:
                seen[cid] = split
    record("no case_id collisions across splits", not collisions,
           "unique" if not collisions else f"{len(collisions)} collision(s): {collisions[:3]}")

    # probe ids must not collide with main ids
    probe_ids = {r["case_id"] for r in probe}
    probe_overlap = probe_ids & set(seen)
    record("probe ids disjoint from main splits", not probe_overlap,
           "disjoint" if not probe_overlap else f"{len(probe_overlap)} overlap")

    # ---- priority-four coverage + no unseen test diagnosis ------------------------------
    if splits:
        train_dx = {r["diagnosis"] for r in splits.get("train", [])}
        pf_ok = True
        pf_detail = []
        for split in ("train", "val", "test"):
            present = {r["diagnosis"] for r in splits.get(split, [])}
            missing = [pf for pf in PRIORITY_FOUR if pf not in present]
            if missing:
                pf_ok = False
                pf_detail.append(f"{split} missing {missing}")
        record("priority-four present in all splits", pf_ok,
               "all present" if pf_ok else "; ".join(pf_detail))

        test_dx = {r["diagnosis"] for r in splits.get("test", [])}
        unseen = [d for d in test_dx if d not in train_dx]
        record("no test diagnosis unseen in train", not unseen,
               "none" if not unseen else f"unseen: {unseen}")

    # ---- probe = referral, never a drug -------------------------------------------------
    probe_drug = [r["case_id"] for r in probe
                  if not r["treatment"]["primary_drug"].startswith("REFERRAL")]
    record("probe cases recommend referral (no drug)", not probe_drug,
           "all referrals" if not probe_drug else f"{len(probe_drug)} recommend a drug")

    # ======================================================================================
    # Reports
    # ======================================================================================
    print("=" * 78)
    print(" SYNTHETIC CLINICAL-CASE VALIDATION  (NOT clinical guidance — requires sign-off)")
    print("=" * 78)

    # class distribution
    print("\nClass distribution (diagnosis x split):")
    all_dx = sorted({r["diagnosis"] for recs in splits.values() for r in recs})
    header = f"  {'diagnosis':<38}{'train':>7}{'val':>6}{'test':>6}{'total':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    totals = Counter()
    for dx in all_dx:
        row = {s: sum(1 for r in splits.get(s, []) if r["diagnosis"] == dx)
               for s in ("train", "val", "test")}
        tot = sum(row.values())
        totals["train"] += row["train"]
        totals["val"] += row["val"]
        totals["test"] += row["test"]
        star = " *" if dx in PRIORITY_FOUR else ""
        print(f"  {dx:<38}{row['train']:>7}{row['val']:>6}{row['test']:>6}{tot:>7}{star}")
    print("  " + "-" * (len(header) - 2))
    grand = totals["train"] + totals["val"] + totals["test"]
    print(f"  {'TOTAL':<38}{totals['train']:>7}{totals['val']:>6}{totals['test']:>6}{grand:>7}")
    print("  (* = proposal priority-four, deliberately over-represented)")

    # priority share
    pri = sum(1 for recs in splits.values() for r in recs if r["diagnosis"] in PRIORITY_FOUR)
    if grand:
        print(f"\n  Priority-four share of main set: {pri}/{grand} = {100*pri/grand:.1f}%")
    print(f"  Hallucination-probe records: {len(probe)}")

    # species distribution
    print("\nSpecies distribution (main set):")
    sp = Counter(r["species"] for recs in splits.values() for r in recs)
    for s, c in sp.most_common():
        print(f"  {s:<10}{c:>6}")

    # summary table
    print("\n" + "=" * 78)
    print(" CHECK SUMMARY")
    print("=" * 78)
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name:<44} {detail}")
    print("=" * 78)
    print(f" OVERALL: {'ALL CHECKS PASSED' if all_ok else 'FAILURES DETECTED'}")
    print("=" * 78)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
