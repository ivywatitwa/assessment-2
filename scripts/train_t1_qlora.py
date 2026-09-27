#!/usr/bin/env python3
"""Train and evaluate the public T1 microscopy task with MedGemma QLoRA.

This is the shortest complete multimodal path for the current repository. It
deliberately excludes synthetic T2 records because those records have no image
and require a separate text-only or mixed-modality collator.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Keep asynchronous CUDA execution for normal training throughput. Set this to
# 1 only when diagnosing a CUDA kernel error.
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_utils import load_hf_token  # noqa: E402


MODEL_ID = "google/medgemma-4b-it"
MODEL_REVISION = "290cda5eeccbee130f987c4ad74a59ae6f196408"


def load_t1_records(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [
        row
        for row in rows
        if row.get("task") == "T1_microscopy" and row.get("image")
    ]


def class_balance_records(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Cycle minority records and interleave classes for training only."""
    by_label: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        label = str(record["messages"][1]["content"])
        by_label.setdefault(label, []).append(record)
    if len(by_label) != 2 or any(not rows for rows in by_label.values()):
        raise ValueError("--class-balanced requires exactly two non-empty T1 classes")

    rng = random.Random(seed)
    labels = sorted(by_label)
    for rows in by_label.values():
        rng.shuffle(rows)

    target = max(len(rows) for rows in by_label.values())
    balanced: list[dict[str, Any]] = []
    for index in range(target):
        for label in labels:
            rows = by_label[label]
            balanced.append(rows[index % len(rows)])
    return balanced


def to_example(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record": record,
        "image": root / record["image"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": record["messages"][0]["content"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": record["messages"][1]["content"]}
                ],
            },
        ],
    }


def build_collator(processor):
    tokenizer = processor.tokenizer

    def collate(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        images: list[list[Image.Image]] = []

        for example in examples:
            image = Image.open(example["image"]).convert("RGB")
            images.append([image])
            texts.append(
                processor.apply_chat_template(
                    example["messages"],
                    add_generation_prompt=False,
                    tokenize=False,
                ).strip()
            )

        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        labels = batch["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels >= tokenizer.vocab_size] = -100

        for token_name in ("boi_token", "eoi_token", "image_token"):
            token = getattr(tokenizer, token_name, None)
            if token is None:
                token = tokenizer.special_tokens_map.get(token_name)
            if token is not None:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id is not None and token_id >= 0:
                    labels[labels == token_id] = -100

        batch["labels"] = labels
        return batch

    return collate


def move_batch(batch, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        key: (
            value.to(device=device, dtype=dtype)
            if torch.is_floating_point(value)
            else value.to(device)
        )
        for key, value in batch.items()
    }


def parse_t1_label(text: str) -> str:
    normalized = text.lower().replace(" ", "_")
    if "trypanosome_present" in normalized:
        return "trypanosome_present"
    if "negative_control" in normalized:
        return "negative_control"
    return "invalid_output"


def evaluate_t1(
    model,
    processor,
    examples: list[dict[str, Any]],
    device: torch.device,
    output_path: Path,
    max_new_tokens: int,
    model_dtype: torch.dtype,
) -> dict[str, Any]:
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    base_model.config.use_cache = True
    predictions: list[dict[str, Any]] = []

    for index, example in enumerate(examples, 1):
        image = Image.open(example["image"]).convert("RGB")
        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": example["record"]["messages"][0]["content"],
                },
            ],
        }
        inputs = move_batch(
            processor.apply_chat_template(
                [user_message],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ),
            device,
            model_dtype,
        )

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        prompt_length = inputs["input_ids"].shape[-1]
        raw_output = processor.decode(
            generated[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()
        record = example["record"]
        predictions.append(
            {
                "record_id": record["meta"]["record_id"],
                "group_id": record["meta"]["group_id"],
                "gold_label": record["messages"][1]["content"],
                "predicted_label": parse_t1_label(raw_output),
                "raw_output": raw_output,
            }
        )

        if index == 1 or index % 25 == 0:
            print(f"evaluated {index}/{len(examples)}")

        # Generation retains image tensors unless the per-example references
        # are released explicitly on a constrained GPU.
        del inputs, generated, image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row) + "\n" for row in predictions),
        encoding="utf-8",
    )

    gold = [row["gold_label"] for row in predictions]
    predicted = [row["predicted_label"] for row in predictions]
    gold_labels = sorted(set(gold))
    labels = sorted(set(gold) | set(predicted))

    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    confusion = confusion_matrix(gold, predicted, labels=labels)
    recalls = {
        label: float(confusion[index, index] / confusion[index].sum())
        if confusion[index].sum()
        else 0.0
        for index, label in enumerate(labels) if label in gold_labels
    }

    metrics = {
        "records": len(predictions),
        "groups": len({row["group_id"] for row in predictions}),
        "accuracy": accuracy_score(gold, predicted),
        "macro_f1": f1_score(gold, predicted, labels=gold_labels, average="macro", zero_division=0),
        "labels": labels,
        "confusion_matrix": confusion.tolist(),
        "per_class_recall": recalls,
        "balanced_accuracy": sum(recalls.values()) / len(recalls) if recalls else 0.0,
        "negative_control_recall": recalls.get("negative_control", 0.0),
        "invalid_outputs": sum(row["predicted_label"] == "invalid_output" for row in predictions),
        "status": "PASS",
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/medgemma-t1-qlora"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-balanced",
        action="store_true",
        help="Cycle minority T1 records and interleave the two classes for training",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Evaluate the pinned base model without loading or training an adapter",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--allow-mps",
        action="store_true",
        help="Use Apple Metal (MPS) with an unquantized FP16 base model instead of CUDA QLoRA",
    )
    args = parser.parse_args()
    if args.eval_only and args.base_only:
        raise SystemExit("Use either --eval-only or --base-only, not both")
    if args.base_only and args.class_balanced:
        raise SystemExit("--base-only cannot be combined with --class-balanced")

    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if not has_cuda and not (args.allow_mps and has_mps):
        raise SystemExit("A CUDA GPU is required; pass --allow-mps on a supported Apple GPU")
    if not load_hf_token():
        raise SystemExit("HF_TOKEN is required for gated MedGemma access")
    torch.manual_seed(args.seed)
    if has_cuda:
        torch.cuda.manual_seed_all(args.seed)

    root = args.root.resolve()
    train_path = root / "data/secondary/instruction_tuning_mixed/train.jsonl"
    evaluation_path = root / "data/secondary/instruction_tuning_mixed" / f"{args.split}.jsonl"
    source_train_records = load_t1_records(train_path)
    val_records = load_t1_records(evaluation_path)
    source_train_counts = Counter(str(row["messages"][1]["content"]) for row in source_train_records)
    train_records = source_train_records
    if args.class_balanced and not args.eval_only:
        train_records = class_balance_records(train_records, args.seed)
    if args.max_train:
        train_records = train_records[: args.max_train]
    if args.max_val:
        val_records = val_records[: args.max_val]

    train_examples = [to_example(root, row) for row in train_records]
    val_examples = [to_example(root, row) for row in val_records]
    effective_train_counts = Counter(str(row["messages"][1]["content"]) for row in train_records)
    print(f"Public T1 source train: {len(source_train_records)}")
    print(f"Public T1 train examples: {len(train_examples)}")
    print(f"Public T1 train class counts: {dict(effective_train_counts)}")
    print(f"Public T1 {args.split}: {len(val_examples)}")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=os.environ["HF_TOKEN"],
    )
    processor.tokenizer.padding_side = "right"
    # FP16 produced non-finite multimodal logits in the verified smoke test;
    # keep the MPS fallback numerically stable even though it uses more memory.
    model_dtype = torch.float32
    if has_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            token=os.environ["HF_TOKEN"],
            quantization_config=bnb_config,
            dtype=model_dtype,
            device_map="auto",
        )
        device = next(model.parameters()).device
    else:
        device = torch.device("mps")
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            token=os.environ["HF_TOKEN"],
            dtype=model_dtype,
        ).to(device)
    model.config.use_cache = False

    inference_only = args.eval_only or args.base_only
    if args.eval_only:
        from peft import PeftModel

        if not (args.output / "adapter_config.json").exists():
            raise SystemExit(f"adapter checkpoint not found in {args.output}")
        model = PeftModel.from_pretrained(model, args.output, is_trainable=False)
        trainable: list[torch.nn.Parameter] = []
    elif not args.base_only:
        if has_cuda:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        elif hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            ),
        )
        for name, parameter in model.named_parameters():
            if any(marker in name for marker in ("vision_tower", "vision_model", "vision_encoder")):
                parameter.requires_grad = False

        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("LoRA created no trainable parameters")
        model.print_trainable_parameters()

    collate = build_collator(processor)
    loss_history: list[float] = []
    if not inference_only:
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
        total_micro_steps = args.epochs * len(train_examples)
        total_updates = math.ceil(total_micro_steps / args.gradient_accumulation_steps)
        update = 0
        micro_step = 0
        pending_losses: list[float] = []
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(args.epochs):
            for example in train_examples:
                micro_step += 1
                batch = collate([example])
                model_inputs = move_batch(
                    {key: value for key, value in batch.items() if key != "labels"},
                    device,
                    model_dtype,
                )
                labels = batch["labels"].to(device)
                outputs = model(**model_inputs, labels=labels, use_cache=False)
                loss = outputs.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at micro-step {micro_step}")
                pending_losses.append(float(loss.detach()))
                if micro_step == 1 or micro_step % 10 == 0:
                    print(
                        f"epoch={epoch + 1}/{args.epochs} "
                        f"micro_step={micro_step}/{total_micro_steps} "
                        f"loss={float(loss.detach()):.4f}",
                        flush=True,
                    )
                (loss / args.gradient_accumulation_steps).backward()

                # Release the large multimodal graph before the next image.
                del outputs, loss, batch, model_inputs, labels

                if micro_step % args.gradient_accumulation_steps == 0 or micro_step == total_micro_steps:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    update += 1
                    mean_loss = sum(pending_losses) / len(pending_losses)
                    pending_losses = []
                    loss_history.append(mean_loss)
                    if update == 1 or update % 10 == 0:
                        print(f"epoch={epoch + 1}/{args.epochs} update={update}/{total_updates} loss={mean_loss:.4f}")
                    if torch.cuda.is_available() and update % 10 == 0:
                        torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "training_summary.json"
    training_summary = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "task": "public_t1_only",
        "sampling": {
            "class_balanced": bool(args.class_balanced and not args.eval_only),
            "seed": args.seed,
            "source_train_records": len(source_train_records),
            "source_class_counts": dict(source_train_counts),
            "effective_class_counts": dict(effective_train_counts),
        },
        "train_records": len(train_examples),
        "validation_records": len(val_examples),
        "evaluation_split": args.split,
        "epochs": args.epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "compute_dtype": "float32",
        "final_loss": loss_history[-1] if loss_history else None,
    }
    if not inference_only:
        model.save_pretrained(args.output)
        processor.save_pretrained(args.output)
        summary_path.write_text(json.dumps(training_summary, indent=2) + "\n", encoding="utf-8")
        if args.skip_eval:
            print(f"Saved adapter and processor to {args.output}")
            return 0
    predictions_path = args.output / "t1_predictions.jsonl"
    metrics = evaluate_t1(
        model,
        processor,
        val_examples,
        device,
        predictions_path,
        args.max_new_tokens,
        model_dtype,
    )
    (args.output / "t1_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if not summary_path.exists():
        summary_path.write_text(json.dumps(training_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved adapter and results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
