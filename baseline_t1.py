#!/usr/bin/env python3
"""
baseline_t1.py - a class-weighted reference classifier on the SAME grouped Tryp splits.

Why: the MedGemma adapter predicted "positive" for every image. Without a comparator a marker
cannot tell whether the task is unlearnable on these splits or whether the adapter's training
objective (one epoch, 30:1 imbalance, no re-weighting) caused the collapse. A frozen ImageNet
ResNet-18 + class-balanced logistic regression answers that in ~10 minutes on a laptop CPU/MPS.

It reuses your split files unchanged, never trains on validation/test, picks the decision threshold
on VALIDATION only, and reports test once.

Usage (from the project root):
    pip install torch torchvision scikit-learn pillow
    python3 baseline_t1.py --check-inputs --image-source prepared
    python3 baseline_t1.py --image-source prepared --out data/secondary/reports/baseline_t1.json

If your split files use different field names, pass them explicitly, e.g.
    --image-key normalized_path --label-key benchmark_label --group-key group_id
The script prints the keys it finds in the first record so you can check.
"""
import argparse
import json
from pathlib import Path

POS = "trypanosome_present"
NEG = "negative_control"
CAND_IMG = ["normalized_path", "prepared_path", "processed_path", "image_path", "png_path", "path", "file_path", "original_path"]
CAND_LABEL = ["benchmark_label", "gold_label", "label", "binary_label"]
CAND_GROUP = ["group_id", "source_video_group", "source_group", "video_id"]


def load_split(d, name):
    for fn in (f"{name}.jsonl", f"{name}_images.jsonl", f"images_{name}.jsonl"):
        p = d / fn
        if p.exists():
            return [json.loads(l) for l in p.open() if l.strip()]
    hits = sorted(d.glob(f"*{name}*.jsonl"))
    if hits:
        return [json.loads(l) for l in hits[0].open() if l.strip()]
    raise SystemExit(f"No {name} split found in {d} (looked for {name}.jsonl etc.). Files: {[x.name for x in d.iterdir()]}")


def pick(rec, forced, cands, what):
    if forced:
        return forced
    for k in cands:
        if k in rec:
            return k
    raise SystemExit(f"Could not find the {what} field. Keys in first record: {sorted(rec)}. Pass --{what}-key.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default="data/secondary/splits/images")
    ap.add_argument("--root", default=".", help="prefix for relative image paths")
    ap.add_argument("--image-key")
    ap.add_argument("--label-key")
    ap.add_argument("--group-key")
    ap.add_argument("--out", default="baseline_t1.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image-source", choices=("auto", "raw", "prepared"), default="auto",
                    help="auto tries manifest path first, then the prepared split PNG; prepared matches MedGemma's image input")
    ap.add_argument("--check-inputs", action="store_true", help="verify all image paths and group splits without model dependencies")
    ap.add_argument("--validation-only", action="store_true", help="fit and select threshold on train/val without evaluating test")
    a = ap.parse_args()

    d = Path(a.splits_dir)
    splits = {s: load_split(d, s) for s in ("train", "validation" if list(d.glob("*validation*")) else "val", "test")}
    names = list(splits)
    first = splits[names[0]][0]
    ik = pick(first, a.image_key, CAND_IMG, "image")
    lk = pick(first, a.label_key, CAND_LABEL, "label")
    gk = pick(first, a.group_key, CAND_GROUP, "group")
    print(f"Using image={ik!r} label={lk!r} group={gk!r}")

    # leakage guard: groups must not cross splits
    gsets = {s: {r[gk] for r in recs} for s, recs in splits.items()}
    for i, s in enumerate(names):
        for t in names[i + 1:]:
            assert not gsets[s] & gsets[t], f"group leakage between {s} and {t}"

    root = Path(a.root)
    resolved = {}
    for s in names:
        resolved[s] = []
        for r in splits[s]:
            if r[lk] not in (POS, NEG):
                raise ValueError(f"Unexpected label for {r.get('record_id')}: {r[lk]}")
            if "record_id" not in r:
                raise ValueError("Split record is missing record_id for prepared-image lookup")
            source = Path(r[ik])
            raw = source if source.is_absolute() else root / source
            prepared = root / "data/secondary/processed/microscopy/tryp" / r.get("split", s) / r[lk] / f"{r['record_id']}.png"
            candidates = {"raw": (raw,), "prepared": (prepared,), "auto": (raw, prepared)}[a.image_source]
            selected = next((p for p in candidates if p.is_file()), None)
            if selected is None:
                raise FileNotFoundError(f"No image for {r['record_id']} in {s}; checked: {candidates}")
            resolved[s].append(selected)
    print("Verified images:", {s: len(paths) for s, paths in resolved.items()}, "source:", a.image_source)
    if a.check_inputs:
        return

    import numpy as np
    import torch
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
    from torchvision import models

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    net = models.resnet18(weights=weights)
    net.fc = torch.nn.Identity()
    net.eval().to(dev)
    tf = weights.transforms()

    def feats(paths):
        out = []
        with torch.no_grad():
            for i in range(0, len(paths), 64):
                batch = []
                for p in paths[i:i + 64]:
                    batch.append(tf(Image.open(p).convert("RGB")))
                out.append(net(torch.stack(batch).to(dev)).cpu().numpy())
                print(f"  {i + len(batch)}/{len(paths)}", end="\r")
        print()
        return np.concatenate(out)

    X, y = {}, {}
    for s in names[:2] if a.validation_only else names:
        print(f"Extracting features: {s}")
        X[s] = feats(resolved[s])
        y[s] = np.array([1 if r[lk] == POS else 0 for r in splits[s]])

    clf = LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0, random_state=a.seed)
    clf.fit(X["train"], y["train"])

    val = names[1]
    pv = clf.predict_proba(X[val])[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    thr = float(grid[np.argmax([balanced_accuracy_score(y[val], (pv >= t).astype(int)) for t in grid])])

    def report(s, t):
        p = clf.predict_proba(X[s])[:, 1]
        yp = (p >= t).astype(int)
        cm = confusion_matrix(y[s], yp, labels=[0, 1]).tolist()
        neg_groups = sorted({r[gk] for r in splits[s] if r[lk] == NEG})
        return {
            "records": int(len(yp)), "negatives": int((y[s] == 0).sum()), "negative_groups": neg_groups,
            "accuracy": round(float((yp == y[s]).mean()), 4),
            "macro_f1": round(float(f1_score(y[s], yp, average="macro")), 4),
            "negative_recall": round(float(recall_score(y[s], yp, pos_label=0)), 4),
            "positive_recall": round(float(recall_score(y[s], yp, pos_label=1)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y[s], yp)), 4),
            "confusion_matrix_[neg,pos]": cm,
            "always_positive_accuracy": round(float(y[s].mean()), 4),
        }

    reported = (val,) if a.validation_only else (val, "test")
    res = {
        "model": "ResNet-18 (ImageNet, frozen) + logistic regression, class_weight=balanced",
        "seed": a.seed, "threshold_selected_on": val, "threshold": thr,
        "image_source": a.image_source,
        "fields": {"image": ik, "label": lk, "group": gk},
        **{s: report(s, thr) for s in reported},
        "default_threshold_0.5": {s: report(s, 0.5) for s in reported},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    print("\nTable row for the dissertation (Table 4b):")
    print("| Split | Records | Accuracy | Macro-F1 | Negative recall | Balanced accuracy |")
    for s in reported:
        r = res[s]
        print(f"| {s} | {r['records']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['negative_recall']:.4f} | {r['balanced_accuracy']:.4f} |")


if __name__ == "__main__":
    main()
