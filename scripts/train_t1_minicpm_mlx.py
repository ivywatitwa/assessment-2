#!/usr/bin/env python3
"""Balanced image-text T1 LoRA training for MiniCPM-V 4.6 with MLX-VLM."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any


MODEL_ID = "openbmb/MiniCPM-V-4.6"
MODEL_REVISION = "36f34a661a4bd35d0dc2294cb044d2584646c7d3"
LABELS = ("negative_control", "trypanosome_present")
IMAGE_MODE = "4x"


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(row["label"] for row in rows))


def prompt_from(row: dict[str, Any]) -> str:
    return next(
        part["text"]
        for part in row["messages"][0]["content"]
        if part.get("type") == "text"
    )


def validate_rows(root: Path, rows: list[dict[str, Any]], split: str) -> None:
    for index, row in enumerate(rows):
        if not isinstance(row.get("images"), list) or len(row["images"]) != 1:
            raise ValueError(f"{split}[{index}] must have one path in its images column")
        if len(row.get("messages", [])) != 2:
            raise ValueError(f"{split}[{index}] must contain one user/assistant turn")
        image_path = row["images"][0]
        user_content = row["messages"][0].get("content", [])
        image_parts = [part for part in user_content if part.get("type") == "image"]
        text_parts = [part for part in user_content if part.get("type") == "text"]
        if len(image_parts) != 1 or image_parts[0].get("image") != image_path:
            raise ValueError(f"{split}[{index}] message image does not match its images column")
        if len(text_parts) != 1 or not text_parts[0].get("text"):
            raise ValueError(f"{split}[{index}] must have one non-empty user text prompt")
        assistant_content = row["messages"][1].get("content", [])
        target = assistant_content[0].get("text") if assistant_content else None
        if target not in LABELS or target != row.get("label"):
            raise ValueError(f"{split}[{index}] has an invalid target label")
        resolved = root / image_path
        if not resolved.is_file():
            raise FileNotFoundError(resolved)


def configure_image_mode(model, processor) -> None:
    # Keep the processor's image placeholder count and model-side feature merger
    # in agreement. 4x retains more visual detail for microscopy than 16x.
    processor.image_processor.downsample_mode = IMAGE_MODE
    if hasattr(processor, "video_processor"):
        processor.video_processor.downsample_mode = IMAGE_MODE
    model.config.downsample_mode = IMAGE_MODE


def check(args, root: Path) -> None:
    dataset_info = json.loads((args.data_dir / "dataset_info.json").read_text(encoding="utf-8"))
    expected_prompt = dataset_info["prompt"]
    if dataset_info.get("schema") != "mlx-vlm-lora-images-messages-jsonl":
        raise ValueError(f"Unexpected dataset schema: {dataset_info.get('schema')!r}")
    for split in ("train", "val"):
        rows = read_rows(args.data_dir / f"{split}.jsonl")
        validate_rows(root, rows, split)
        if any(prompt_from(row) != expected_prompt for row in rows):
            raise ValueError(f"{split} contains a stale or inconsistent user prompt")
        print(f"{split}: {len(rows)}; {counts(rows)}; all image paths resolve")


def train(args, root: Path) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_map
    from datasets import load_dataset
    from mlx_vlm import load
    from mlx_vlm.trainer.datasets import VisionDataset
    from mlx_vlm.trainer.sft_trainer import iterate_batches, vision_language_loss_fn
    from mlx_vlm.trainer.utils import (
        find_all_linear_names,
        get_peft_model,
        print_trainable_parameters,
        save_adapter,
    )

    train_path = args.data_dir / "train.jsonl"
    rows = read_rows(train_path)
    validate_rows(root, rows, "train")
    dataset_info = json.loads((args.data_dir / "dataset_info.json").read_text(encoding="utf-8"))
    if dataset_info.get("schema") != "mlx-vlm-lora-images-messages-jsonl":
        raise ValueError(f"Unexpected dataset schema: {dataset_info.get('schema')!r}")
    if dataset_info.get("model_id") != MODEL_ID:
        raise ValueError(f"Dataset was prepared for another model: {dataset_info.get('model_id')!r}")
    if any(prompt_from(row) != dataset_info.get("prompt") for row in rows):
        raise ValueError("Training rows have inconsistent prompts")
    if args.max_train is not None:
        if args.max_train < 1:
            raise ValueError("--max-train must be positive")
        rows = rows[:args.max_train]
    if not rows:
        raise ValueError("No training records selected")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty; select a clean directory: {args.output}")
    if args.epochs != 1:
        raise ValueError("This balanced export is defined for one complete pass (--epochs 1)")
    if len(rows) % args.gradient_accumulation_steps:
        raise ValueError(
            "Selected rows must divide evenly by gradient accumulation; "
            "use 4 for the 4,332-row full run"
        )

    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model, processor = load(str(args.model_dir))
    if model.config.model_type != "minicpmv4_6":
        raise ValueError(f"Expected minicpmv4_6, got {model.config.model_type}")
    configure_image_mode(model, processor)

    hf_rows = load_dataset("json", data_files=str(train_path), split="train")
    if args.max_train is not None:
        hf_rows = hf_rows.select(range(len(rows)))
    # MLX-VLM's MiniCPM-V 4.6 ModelConfig currently drops image_token_id
    # while parsing config.json, although the processor retains the correct id.
    dataset_config = vars(model.config).copy()
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        raise ValueError("MiniCPM processor is missing image_token_id")
    dataset_config["image_token_id"] = int(image_token_id)
    dataset = VisionDataset(
        hf_rows,
        dataset_config,
        processor,
        train_on_completions=True,
    )
    probe = dataset[0]
    if probe.get("pixel_values") is None or len(probe["pixel_values"]) == 0:
        raise ValueError("MLX-VLM did not produce image pixel_values for the first record")
    if probe["input_ids"].shape[-1] > args.max_seq_length:
        raise ValueError(
            f"First multimodal example has {probe['input_ids'].shape[-1]} tokens, "
            f"over --max-seq-length {args.max_seq_length}"
        )

    model = get_peft_model(
        model,
        find_all_linear_names(model.language_model),
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        verbose=False,
    )
    print_trainable_parameters(model)
    optimizer = optim.AdamW(learning_rate=args.learning_rate)
    loss_and_grad = nn.value_and_grad(
        model,
        lambda batch: vision_language_loss_fn(
            model, batch, train_on_completions=True
        ),
    )
    model.train()
    total_micro = len(rows)
    total_updates = math.ceil(total_micro / args.gradient_accumulation_steps)
    update = pending = 0
    grads_sum = None
    losses: list[float] = []
    final_loss: float | None = None

    for batch in iterate_batches(dataset, 1, args.max_seq_length, train=False):
        loss, grads = loss_and_grad(batch)
        mx.eval(loss, grads)
        value = float(loss.item())
        if not np.isfinite(value):
            raise FloatingPointError(f"Non-finite loss at example {update * args.gradient_accumulation_steps + pending + 1}")
        final_loss = value
        losses.append(value)
        grads_sum = grads if grads_sum is None else tree_map(lambda a, b: a + b, grads_sum, grads)
        mx.eval(grads_sum)
        del loss, grads, batch
        pending += 1

        if pending == args.gradient_accumulation_steps:
            averaged = tree_map(lambda grad: mx.clip(grad / pending, -1.0, 1.0), grads_sum)
            optimizer.update(model, averaged)
            mx.eval(model.state, optimizer.state)
            update += 1
            if update == 1 or update % 10 == 0 or update == total_updates:
                print(
                    f"update={update}/{total_updates} "
                    f"loss={sum(losses) / len(losses):.5f}",
                    flush=True,
                )
            grads_sum, pending, losses = None, 0, []
            mx.clear_cache()

    if update != total_updates or pending:
        raise RuntimeError(f"Incomplete optimizer updates: {update}/{total_updates}, pending={pending}")

    args.output.mkdir(parents=True, exist_ok=True)
    save_adapter(model, args.output / "adapters.safetensors")
    (args.output / "training_summary.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "model_dir": str(args.model_dir),
                "mlx_vlm_model_type": model.config.model_type,
                "image_mode": IMAGE_MODE,
                "schema": "mlx-vlm-lora-images-messages-jsonl",
                "train_records": len(rows),
                "effective_class_counts": counts(rows),
                "epochs": args.epochs,
                "optimizer_updates": update,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "training_loss": "assistant_completion_only",
                "prompt": prompt_from(rows[0]),
                "final_loss": final_loss,
                "dataset": str(train_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved adapter to {args.output}")


def parse_label(text: str) -> str:
    normalized = text.lower().replace(" ", "_")
    for label in LABELS:
        if label in normalized:
            return label
    return "invalid_output"


def evaluate(args, root: Path) -> None:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    rows = read_rows(args.data_dir / f"{args.split}.jsonl")
    validate_rows(root, rows, args.split)
    limit = args.max_val if args.split == "val" else args.max_test
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No {args.split} rows selected")
    if not (args.output / "adapters.safetensors").is_file():
        raise FileNotFoundError(f"No adapter at {args.output}; train first")

    model, processor = load(str(args.model_dir), adapter_path=str(args.output))
    configure_image_mode(model, processor)
    predictions = []
    for index, row in enumerate(rows, 1):
        image = row["images"][0]
        prompt = apply_chat_template(
            processor,
            model.config,
            row["messages"][:1],
            add_generation_prompt=True,
            num_images=1,
            enable_thinking=False,
        )
        response = generate(
            model,
            processor,
            prompt,
            image=[str(root / image)],
            max_tokens=args.max_new_tokens,
            temperature=0.0,
            verbose=False,
        )
        raw = response.text.strip()
        gold = row["label"]
        predictions.append(
            {
                "record_id": row.get("source_record_id"),
                "group_id": row.get("group_id"),
                "gold_label": gold,
                "predicted_label": parse_label(raw),
                "raw_output": raw,
            }
        )
        if index == 1 or index % 25 == 0:
            print(f"evaluated {index}/{len(rows)}", flush=True)

    gold = [row["gold_label"] for row in predictions]
    predicted = [row["predicted_label"] for row in predictions]
    labels = list(LABELS)
    matrix = confusion_matrix(gold, predicted, labels=labels)
    per_class_recall = {
        label: float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else 0.0
        for i, label in enumerate(labels)
    }
    metrics = {
        "records": len(predictions),
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, labels=labels, average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "per_class_recall": per_class_recall,
        "balanced_accuracy": sum(per_class_recall.values()) / len(labels),
        "negative_control_recall": per_class_recall["negative_control"],
        "invalid_outputs": sum(row["predicted_label"] == "invalid_output" for row in predictions),
        "status": "PASS",
        "model_id": MODEL_ID,
        "image_mode": IMAGE_MODE,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_prefix = "t1" if args.split == "val" else "t1_test"
    (args.output / f"{output_prefix}_predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
    )
    (args.output / f"{output_prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--data-dir", type=Path, default=Path("data/secondary/mlx_t1_minicpmv46"))
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/mlx-openbmb-minicpm-v46-4bit"))
    parser.add_argument("--output", type=Path, default=Path("adapters/minicpm-v46-t1-balanced-prompt-v1"))
    parser.add_argument("--mode", choices=("check", "train", "eval"), default="check")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max-val", type=int, default=488)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    root = args.root.resolve()
    args.data_dir = args.data_dir if args.data_dir.is_absolute() else root / args.data_dir
    args.model_dir = args.model_dir if args.model_dir.is_absolute() else root / args.model_dir
    args.output = args.output if args.output.is_absolute() else root / args.output
    os.chdir(root)

    if args.mode == "check":
        check(args, root)
    elif args.mode == "train":
        train(args, root)
    else:
        evaluate(args, root)


if __name__ == "__main__":
    main()
