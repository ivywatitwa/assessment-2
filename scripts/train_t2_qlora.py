#!/usr/bin/env python3
"""Train a separate text-only T2 QLoRA adapter on synthetic auxiliary cases.

This adapter is intentionally separate from the public T1 image adapter. T2
records are synthetic, so all resulting metrics are auxiliary pre-validation
diagnostics and are not clinical evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_utils import load_hf_token  # noqa: E402


MODEL_ID = "google/medgemma-4b-it"
MODEL_REVISION = "290cda5eeccbee130f987c4ad74a59ae6f196408"


def load_t2_records(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("task") == "T2_treatment" and row.get("image") is None]


def move_batch(batch, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        key: (
            value.to(device=device, dtype=dtype)
            if torch.is_floating_point(value)
            else value.to(device)
        )
        for key, value in batch.items()
    }


def make_batch(processor, row: dict[str, Any]) -> dict[str, torch.Tensor]:
    user = [{"role": "user", "content": row["messages"][0]["content"]}]
    full = user + [{"role": "assistant", "content": row["messages"][1]["content"]}]
    prompt_batch = processor.apply_chat_template(
        user,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    full_batch = processor.apply_chat_template(
        full,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_length = prompt_batch["input_ids"].shape[-1]
    labels = full_batch["input_ids"].clone()
    labels[:, :prompt_length] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels >= processor.tokenizer.vocab_size] = -100
    full_batch["labels"] = labels
    return full_batch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for T2 QLoRA training")
    if not load_hf_token():
        raise SystemExit("HF_TOKEN is required for gated MedGemma access")

    train_path = args.root / "data/secondary/instruction_tuning_synthetic/train.jsonl"
    val_path = args.root / "data/secondary/instruction_tuning_synthetic/val.jsonl"
    train_records = load_t2_records(train_path)
    val_records = load_t2_records(val_path)
    if args.max_train:
        train_records = train_records[: args.max_train]
    print(f"Synthetic T2 train: {len(train_records)}", flush=True)
    print(f"Synthetic T2 validation: {len(val_records)}", flush=True)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=os.environ["HF_TOKEN"],
    )
    processor.tokenizer.padding_side = "right"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=os.environ["HF_TOKEN"],
        quantization_config=quantization,
        dtype=torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA created no trainable parameters")
    model.print_trainable_parameters()

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    total_micro_steps = args.epochs * len(train_records)
    total_updates = math.ceil(total_micro_steps / args.gradient_accumulation_steps)
    update = 0
    micro_step = 0
    pending_losses: list[float] = []
    loss_history: list[float] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        for row in train_records:
            micro_step += 1
            batch = make_batch(processor, row)
            batch = move_batch(batch, device, torch.float32)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at micro-step {micro_step}")
            pending_losses.append(float(loss.detach()))
            (loss / args.gradient_accumulation_steps).backward()
            del outputs, loss, batch

            if micro_step % args.gradient_accumulation_steps == 0 or micro_step == total_micro_steps:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                mean_loss = sum(pending_losses) / len(pending_losses)
                pending_losses = []
                loss_history.append(mean_loss)
                if update == 1 or update % 10 == 0:
                    print(
                        f"epoch={epoch + 1}/{args.epochs} update={update}/{total_updates} loss={mean_loss:.4f}",
                        flush=True,
                    )
                if update % 10 == 0:
                    torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    (args.output / "training_summary.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "task": "synthetic_t2_auxiliary_only",
                "train_records": len(train_records),
                "validation_records": len(val_records),
                "epochs": args.epochs,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "compute_dtype": "float32",
                "final_loss": loss_history[-1] if loss_history else None,
                "clinical_use": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved T2 adapter to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
