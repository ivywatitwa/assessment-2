#!/usr/bin/env python3
"""Evaluate T1 predictions without silently converting invalid labels to correct ones.

Prediction JSONL fields: `record_id`, `group_id`, `gold_label`, `predicted_label`.
Extra fields are preserved in the per-record output when the caller stores raw
predictions separately.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

def load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not all(key in row for key in ("record_id", "group_id", "gold_label", "predicted_label")):
            raise ValueError(f"{path}:{line_no}: missing required prediction field")
        rows.append(row)
    return rows


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [str(row["gold_label"]) for row in rows]
    predicted = [str(row["predicted_label"]) for row in rows]
    gold_labels = sorted(set(gold))
    labels = sorted(set(gold) | set(predicted))
    positions = {label: i for i, label in enumerate(labels)}
    confusion = [[0] * len(labels) for _ in labels]
    for actual, guess in zip(gold, predicted):
        confusion[positions[actual]][positions[guess]] += 1
    recalls = {
        label: confusion[positions[label]][positions[label]] / sum(confusion[positions[label]])
        for label in gold_labels
    }
    class_report = {}
    for label in labels:
        i = positions[label]
        tp = confusion[i][i]
        support = sum(confusion[i])
        predicted_count = sum(row[i] for row in confusion)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_report[label] = {"precision": precision, "recall": recall, "f1-score": f1, "support": support}
    if rows:
        class_report["accuracy"] = sum(a == p for a, p in zip(gold, predicted)) / len(rows)
        for average in ("macro avg", "weighted avg"):
            weights = [1 if average == "macro avg" else class_report[label]["support"] for label in labels]
            divisor = sum(weights)
            class_report[average] = {
                key: sum(class_report[label][key] * weight for label, weight in zip(labels, weights)) / divisor
                if divisor else 0.0
                for key in ("precision", "recall", "f1-score")
            }
            class_report[average]["support"] = len(rows)
    macro_f1 = sum(class_report[label]["f1-score"] for label in gold_labels) / len(gold_labels) if gold_labels else None
    return {
        "records": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "accuracy": sum(actual == guess for actual, guess in zip(gold, predicted)) / len(rows) if rows else None,
        "macro_f1": macro_f1,
        "labels": labels,
        "confusion_matrix": confusion,
        "per_class_recall": recalls,
        "balanced_accuracy": sum(recalls.values()) / len(recalls) if recalls else 0.0,
        "negative_control_recall": recalls.get("negative_control", 0.0),
        "invalid_outputs": sum(label not in gold_labels for label in predicted),
        "classification_report": class_report,
        "class_counts": dict(Counter(gold)),
        "status": "PASS" if rows else "NO_DATA",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(load(args.input))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
