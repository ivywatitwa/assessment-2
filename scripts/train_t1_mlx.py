#!/usr/bin/env python3
"""Class-balanced multimodal T1 QLoRA on Apple Silicon via mlx-vlm 0.7.3.

Run from mlx_t1_balanced_train.ipynb. The prepared training and validation
images must already be present under --root. Test data is never loaded.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

MODEL_ID = "google/medgemma-4b-it"
MODEL_REVISION = "290cda5eeccbee130f987c4ad74a59ae6f196408"
LABELS = ("negative_control", "trypanosome_present")


def load_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        row
        for line in path.open(encoding="utf-8")
        if (row := json.loads(line)).get("task") == "T1_microscopy" and row.get("image")
    ]


def balance_records(records: list[dict], seed: int) -> list[dict]:
    """Match the Colab trainer's seeded, interleaved minority cycling."""
    by_label = {label: [] for label in LABELS}
    for record in records:
        by_label[record["messages"][1]["content"]].append(record)
    if any(not rows for rows in by_label.values()):
        raise ValueError("Both public T1 classes are required")
    rng = random.Random(seed)
    for rows in by_label.values():
        rng.shuffle(rows)
    target = max(map(len, by_label.values()))
    return [
        by_label[label][i % len(by_label[label])]
        for i in range(target)
        for label in LABELS
    ]


def check_images(root: Path, records: list[dict]) -> None:
    missing = [row["image"] for row in records if not (root / row["image"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} prepared images missing; first: {missing[0]}")


def messages(record: dict) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": record["messages"][0]["content"]},
            ],
        },
        {"role": "assistant", "content": record["messages"][1]["content"]},
    ]


def model_path_ok(model_dir: Path) -> None:
    source = model_dir / "model_source.json"
    if not source.exists():
        raise FileNotFoundError(f"Convert the pinned model in the notebook first: {source}")
    info = json.loads(source.read_text(encoding="utf-8"))
    if (info.get("model_id"), info.get("revision"), info.get("quantization")) != (
        MODEL_ID, MODEL_REVISION, "mlx-4bit-affine"
    ):
        raise ValueError(f"Wrong model conversion provenance: {info}")


def make_dataset(root: Path, records: list[dict], model, processor, max_seq_length: int):
    import mlx.core as mx
    import numpy as np
    from mlx_vlm.trainer.datasets import VisionDataset
    from mlx_vlm.models.base import to_mlx
    from mlx_vlm.utils import prepare_inputs, process_inputs_with_fallback

    class ImageFirstDataset(VisionDataset):
        """Keep the Colab image-first Gemma prompt (the MLX helper puts it last)."""

        def process(self, item):
            image = item["images"]
            conversation = item["messages"]
            texts = [
                self.processor.tokenizer.apply_chat_template(
                    turns, tokenize=False, add_generation_prompt=generation
                )
                for turns, generation in ((conversation, False), (conversation[:1], True))
            ]

            def encode(text):
                try:
                    return process_inputs_with_fallback(
                        processor=self.processor, prompts=[text], images=image,
                        audio=None, add_special_tokens=False,
                    )
                except Exception:
                    return prepare_inputs(
                        processor=self.processor, prompts=[text], images=image,
                        image_token_index=self.config["image_token_index"],
                        add_special_tokens=False,
                    )

            inputs = encode(texts[0])
            if "images" in inputs and "pixel_values" not in inputs:
                inputs["pixel_values"] = inputs.pop("images")
            inputs = to_mlx(inputs)
            prefix = encode(texts[1])
            prefix_len = np.array(prefix["input_ids"]).reshape(-1).shape[0]
            ids = inputs["input_ids"]
            if prefix_len >= ids.shape[-1]:
                raise ValueError("No assistant completion tokens after multimodal prompt")
            if ids.shape[-1] > max_seq_length:
                raise ValueError(
                    f"Image + completion has {ids.shape[-1]} tokens; raise --max-seq-length "
                    f"above {max_seq_length} to avoid silently truncating a training example"
                )
            return {
                "input_ids": ids,
                "attention_mask": inputs.get("attention_mask", mx.ones_like(ids)),
                "pixel_values": inputs["pixel_values"],
                "completion_mask": (mx.arange(ids.shape[-1])[None, :] >= prefix_len).astype(mx.int32),
                **{k: v for k, v in inputs.items() if k not in
                   ("input_ids", "attention_mask", "pixel_values", "completion_mask")},
            }

    rows = [
        {"images": [str((root / row["image"]).resolve())], "messages": messages(row)}
        for row in records
    ]
    return ImageFirstDataset(rows, vars(model.config), processor, train_on_completions=True)


def train(args, root: Path, source: list[dict], validation: list[dict]) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_map
    from mlx_vlm import load
    from mlx_vlm.trainer.sft_trainer import iterate_batches, vision_language_loss_fn
    from mlx_vlm.trainer.utils import (
        find_all_linear_names, get_peft_model, print_trainable_parameters, save_adapter,
    )

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty; choose another directory: {args.output}")
    if args.epochs < 1 or args.gradient_accumulation_steps < 1 or args.max_seq_length < 2:
        raise ValueError("epochs and accumulation must be positive; max sequence length >= 2")
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    balanced = balance_records(source, args.seed)
    selected = balanced[:args.max_train] if args.max_train is not None else balanced
    if not selected:
        raise ValueError("No training records")
    check_images(root, selected)
    print(f"Source T1: {len(source)} {dict(Counter(r['messages'][1]['content'] for r in source))}")
    print(f"Balanced T1 used: {len(selected)} {dict(Counter(r['messages'][1]['content'] for r in selected))}")
    model, processor = load(str(args.model_dir))
    if model.config.model_type != "gemma3":
        raise ValueError(f"Expected multimodal Gemma 3; got {model.config.model_type}")
    dataset = make_dataset(root, selected, model, processor, args.max_seq_length)
    probe = dataset[0]
    if probe["pixel_values"] is None or int(mx.sum(probe["completion_mask"]).item()) < 1:
        raise ValueError("Image or assistant completion not present in prepared example")
    if probe["input_ids"].shape[-1] > args.max_seq_length:
        raise ValueError("Image + text exceeds max-seq-length; raise it before training")

    model = get_peft_model(
        model, find_all_linear_names(model.language_model),
        rank=8, alpha=16, dropout=0.05, verbose=False,
    )
    print_trainable_parameters(model)
    optimizer = optim.AdamW(learning_rate=args.learning_rate)
    loss_and_grad = nn.value_and_grad(
        model, lambda batch: vision_language_loss_fn(model, batch, train_on_completions=True)
    )
    model.train()
    total_micro = args.epochs * len(selected)
    total_updates = math.ceil(total_micro / args.gradient_accumulation_steps)
    update = 0
    pending = 0
    grads_sum = None
    losses = []
    last_loss = None
    for epoch in range(args.epochs):
        # train=False traverses the already-interleaved balanced records exactly once;
        # the library's train=True shuffles and repeats indefinitely.
        for batch in iterate_batches(dataset, 1, args.max_seq_length, train=False):
            loss, grads = loss_and_grad(batch)
            mx.eval(loss, grads)
            value = loss.item()
            if not np.isfinite(value):
                raise FloatingPointError(f"Non-finite loss in epoch {epoch + 1}")
            last_loss = float(value)
            losses.append(last_loss)
            grads_sum = grads if grads_sum is None else tree_map(lambda a, b: a + b, grads_sum, grads)
            mx.eval(grads_sum)
            del loss, grads, batch
            pending += 1
            if pending == args.gradient_accumulation_steps or (epoch == args.epochs - 1 and update * args.gradient_accumulation_steps + pending == total_micro):
                # Flush the final partial accumulation (4 examples for 4,332 / 8).
                averaged = tree_map(lambda g: mx.clip(g / pending, -1.0, 1.0), grads_sum)
                optimizer.update(model, averaged)
                mx.eval(model.state, optimizer.state)
                update += 1
                if update == 1 or update % 10 == 0 or update == total_updates:
                    print(f"epoch={epoch + 1}/{args.epochs} update={update}/{total_updates} loss={sum(losses) / len(losses):.4f}", flush=True)
                grads_sum, pending, losses = None, 0, []
                mx.clear_cache()

    if update != total_updates or pending:
        raise RuntimeError(f"Incomplete optimizer steps: {update} of {total_updates}")
    args.output.mkdir(parents=True, exist_ok=True)
    save_adapter(model, args.output / "adapters.safetensors")
    (args.output / "training_summary.json").write_text(json.dumps({
        "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "base_model": str(args.model_dir), "backend": "mlx-vlm", "quantization": "mlx-4bit-affine",
        "task": "public_t1_only", "sampling": {
            "class_balanced": True, "seed": args.seed,
            "source_train_records": len(source),
            "source_class_counts": dict(Counter(r["messages"][1]["content"] for r in source)),
            "effective_class_counts": dict(Counter(r["messages"][1]["content"] for r in selected)),
        },
        "train_records": len(selected), "validation_records": len(validation),
        "epochs": args.epochs, "optimizer_updates": update,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate, "batch_size": 1,
        "training_loss": "assistant_completion_only", "final_loss": last_loss,
        "evaluation_split": "val",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved MLX adapter: {args.output}")


def parse_label(text: str) -> str:
    normalized = text.lower().replace(" ", "_")
    return next((label for label in reversed(LABELS) if label in normalized), "invalid_output")


def evaluate(args, root: Path, validation: list[dict]) -> None:
    from mlx_vlm import load
    from mlx_vlm.generate import generate
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    if not (args.output / "adapters.safetensors").is_file():
        raise FileNotFoundError(f"Train the MLX adapter first: {args.output}")
    if args.max_val is not None:
        validation = validation[:args.max_val]
    if not validation:
        raise ValueError("No validation examples")
    check_images(root, validation)
    model, processor = load(str(args.model_dir), adapter_path=str(args.output))
    model.eval()
    predictions = []
    for index, row in enumerate(validation, 1):
        prompt = processor.tokenizer.apply_chat_template(
            messages(row)[:1], tokenize=False, add_generation_prompt=True,
        )
        response = generate(
            model, processor, prompt, image=[str((root / row["image"]).resolve())],
            max_tokens=args.max_new_tokens, temperature=0.0, verbose=False,
        )
        predictions.append({
            "record_id": row["meta"]["record_id"],
            "group_id": row["meta"]["group_id"],
            "gold_label": row["messages"][1]["content"],
            "predicted_label": parse_label(response.text), "raw_output": response.text.strip(),
        })
        if index == 1 or index % 25 == 0:
            print(f"evaluated {index}/{len(validation)}", flush=True)

    gold = [row["gold_label"] for row in predictions]
    predicted = [row["predicted_label"] for row in predictions]
    labels = sorted(set(gold) | set(predicted))
    confusion = confusion_matrix(gold, predicted, labels=labels)
    recalls = {
        label: float(confusion[i, i] / confusion[i].sum())
        for i, label in enumerate(labels) if label in set(gold)
    }
    metrics = {
        "records": len(predictions), "groups": len({row["group_id"] for row in predictions}),
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, labels=list(LABELS), average="macro", zero_division=0)),
        "labels": labels, "confusion_matrix": confusion.tolist(),
        "per_class_recall": recalls, "balanced_accuracy": sum(recalls.values()) / len(recalls),
        "negative_control_recall": recalls.get("negative_control", 0.0),
        "invalid_outputs": predicted.count("invalid_output"), "status": "PASS",
    }
    (args.output / "t1_predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
    )
    (args.output / "t1_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/mlx-medgemma-4b-it-4bit"))
    parser.add_argument("--output", type=Path, default=Path("adapters/medgemma-t1-mlx-balanced"))
    parser.add_argument("--mode", choices=("check", "train", "eval"), default="check")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int, default=488)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    source = load_records(root / "data/secondary/instruction_tuning_mixed/train.jsonl")
    validation = load_records(root / "data/secondary/instruction_tuning_mixed/val.jsonl")
    if not source or not validation:
        raise ValueError("Both training and validation T1 records are needed")
    if args.mode == "check":
        check_images(root, source + validation)
        balanced = balance_records(source, args.seed)
        print(f"T1 source={len(source)}, balanced={len(balanced)}, val={len(validation)}")
        print(f"Balanced class counts: {dict(Counter(r['messages'][1]['content'] for r in balanced))}")
        return
    model_path_ok(args.model_dir)
    if args.mode == "train":
        train(args, root, source, validation)
    else:
        evaluate(args, root, validation)


if __name__ == "__main__":
    main()
