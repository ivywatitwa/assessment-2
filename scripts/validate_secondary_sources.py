#!/usr/bin/env python3
"""Validate the secondary-data boundary and write an auditable report.

This validator deliberately does not treat synthetic records or unverified local
images as secondary data. It checks the source register, local intake manifest,
and optional revised image/text manifests without downloading anything.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
REGISTER = ROOT / "data" / "secondary" / "sources" / "source_register.json"
LOCAL_MANIFEST = ROOT / "data" / "microscopy" / "local_intake_manifest.jsonl"

REQUIRED_SOURCE_FIELDS = {
    "source_id", "title", "source_type", "intended_role", "licence",
    "eligibility_status", "notes",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected an object")
        rows.append(value)
    return rows


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def validate_register(register: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if register.get("study_policy") not in {"secondary_data_only", "secondary_data_only_with_synthetic_auxiliary"}:
        errors.append("study_policy must use a documented secondary-data policy")
    sources = register.get("sources")
    if not isinstance(sources, list) or not sources:
        return ["sources must be a non-empty list"]

    seen: set[str] = set()
    for index, source in enumerate(sources):
        prefix = f"sources[{index}]"
        if not isinstance(source, dict):
            errors.append(f"{prefix} must be an object")
            continue
        missing = REQUIRED_SOURCE_FIELDS - source.keys()
        errors.extend(f"{prefix} missing {field}" for field in sorted(missing))
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            errors.append(f"{prefix}.source_id must be non-empty")
        elif source_id in seen:
            errors.append(f"duplicate source_id: {source_id}")
        else:
            seen.add(source_id)
        status = str(source.get("eligibility_status", ""))
        role = str(source.get("intended_role", ""))
        if role == "excluded" and not status.startswith("excluded_"):
            errors.append(f"{prefix}: excluded role must have excluded_* status")
        if role not in {"excluded", "synthetic_auxiliary_t2_development_and_training"} and status.startswith("excluded_"):
            errors.append(f"{prefix}: non-excluded role has excluded_* status")
        url = source.get("url", "")
        if url and not is_url(url):
            errors.append(f"{prefix}.url is not an http(s) URL")
        if source.get("source_type") == "synthetic_generated" and role not in {"excluded", "synthetic_auxiliary_t2_development_and_training"}:
            errors.append(f"{prefix}: synthetic source must be excluded or auxiliary-only")
        if role == "synthetic_auxiliary_t2_development_and_training":
            if status != "auxiliary_synthetic_prevalidation_only":
                errors.append(f"{prefix}: synthetic auxiliary status is incorrect")
            if source.get("clinical_use") is not False or source.get("requires_expert_validation") is not True:
                errors.append(f"{prefix}: synthetic auxiliary must be clinical_use=false and requires_expert_validation=true")
        if source.get("source_type") == "local_unverified" and role != "excluded":
            errors.append(f"{prefix}: local unverified source cannot be eligible")
    return errors


def audit_local_intake() -> dict[str, Any]:
    rows = load_jsonl(LOCAL_MANIFEST)
    included = [r for r in rows if r.get("included_in_t1") is True]
    status_counts = Counter(str(r.get("label_status", "missing")) for r in rows)
    return {
        "manifest_exists": LOCAL_MANIFEST.exists(),
        "records": len(rows),
        "included_in_t1_records": len(included),
        "label_status_counts": dict(sorted(status_counts.items())),
        "policy_result": "PASS" if not included else "FAIL",
    }


def audit_optional_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve()
    rows = load_jsonl(path)
    prohibited = [
        r for r in rows
        if r.get("provenance") in {"synthetic", "local_unverified", "synthetic_generated"}
        or r.get("source_dataset") == "local_intake_unverified"
        or r.get("included_in_t1") is True
        or str(r.get("image_path", "")).startswith("PLAN-ONLY://")
    ]
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "exists": path.exists(),
        "records": len(rows),
        "prohibited_records": len(prohibited),
        "policy_result": "PASS" if not prohibited else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", type=Path, default=REGISTER)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "secondary" / "reports" / "source_audit.json")
    parser.add_argument("--image-manifest", type=Path)
    parser.add_argument("--text-manifest", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    try:
        register = json.loads(args.register.read_text(encoding="utf-8"))
        errors.extend(validate_register(register))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"register error: {exc}")
        register = {}

    report: dict[str, Any] = {
        "study_policy": register.get("study_policy", "unknown"),
        "register": str(args.register.relative_to(ROOT)) if args.register.is_absolute() else str(args.register),
        "source_count": len(register.get("sources", [])),
        "source_type_counts": dict(Counter(s.get("source_type", "missing") for s in register.get("sources", []))),
        "local_intake": audit_local_intake(),
        "optional_manifests": [],
        "errors": errors,
    }
    if report["local_intake"]["policy_result"] != "PASS":
        errors.append("local intake contains a record marked included_in_t1=true")
    for manifest in (args.image_manifest, args.text_manifest):
        if manifest:
            result = audit_optional_manifest(manifest)
            report["optional_manifests"].append(result)
            if result["policy_result"] != "PASS":
                errors.append(f"prohibited records in {manifest}")
    report["status"] = "PASS" if not errors else "FAIL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
