#!/usr/bin/env python3
"""Compare raw T1 predictions on exactly the same grouped split.

Archive paths may point to original evaluation ZIPs; new runs use JSONL paths.
Missing runs are never replaced with simulated outputs.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def read_predictions(path: Path) -> list[dict]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            matches = [name for name in archive.namelist() if name.endswith("/t1_predictions.jsonl")]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one prediction file in {path}: {matches}")
            content = archive.read(matches[0]).decode("utf-8")
    else:
        content = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--balanced", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from evaluate_t1 import evaluate

    original = read_predictions(args.original)
    if not original:
        raise ValueError("Original predictions are empty")
    reference = {row["record_id"]: (row["gold_label"], row["group_id"]) for row in original}
    if len(reference) != len(original):
        raise ValueError("Original predictions have duplicate record IDs")

    runs = {"original_adapter": original}
    for name, path in (("base_model", args.base), ("balanced_adapter", args.balanced)):
        rows = read_predictions(path)
        observed = {row["record_id"]: (row["gold_label"], row["group_id"]) for row in rows}
        if len(observed) != len(rows) or observed != reference:
            raise ValueError(f"{name}: record IDs, gold labels or groups differ from original")
        runs[name] = rows

    baseline = [
        {**row, "predicted_label": "trypanosome_present"}
        for row in original
    ]
    results = {name: evaluate(rows) for name, rows in {**runs, "always_positive": baseline}.items()}
    result = {
        "split_records": len(reference),
        "matrix_axes": "rows=true, columns=predicted; label order given for each run",
        "runs": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
