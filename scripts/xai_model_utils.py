#!/usr/bin/env python3
"""Shared model loading and score utilities for bounded XAI experiments."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_utils import load_hf_token  # noqa: E402


MODEL_ID = "google/medgemma-4b-it"
MODEL_REVISION = "290cda5eeccbee130f987c4ad74a59ae6f196408"


def load_medgemma(adapter: Path, allow_mps: bool = False):
    if not load_hf_token():
        raise SystemExit("HF_TOKEN is required for gated MedGemma access")

    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if not has_cuda and not (allow_mps and has_mps):
        raise SystemExit("A CUDA GPU is required; pass --allow-mps on a supported Apple GPU")

    token = os.environ["HF_TOKEN"]
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=token,
    )
    processor.tokenizer.padding_side = "right"

    model_dtype = torch.float32
    if has_cuda:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            token=token,
            quantization_config=quantization,
            dtype=model_dtype,
            device_map="auto",
        )
        device = next(model.parameters()).device
    else:
        device = torch.device("mps")
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            token=token,
            dtype=model_dtype,
        ).to(device)

    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"Adapter checkpoint not found: {adapter}")
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    model.get_base_model().config.use_cache = True
    return model, processor, device, model_dtype


def move_batch(batch: Any, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        key: (
            value.to(device=device, dtype=dtype)
            if torch.is_floating_point(value)
            else value.to(device)
        )
        for key, value in batch.items()
    }


def clear_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def score_target(
    model,
    processor,
    messages: list[dict[str, Any]],
    target: str,
    device: torch.device,
    model_dtype: torch.dtype,
) -> float:
    full_messages = messages + [
        {"role": "assistant", "content": [{"type": "text", "text": target}]}
    ]
    prompt_batch = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    full_batch = processor.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_length = prompt_batch["input_ids"].shape[-1]
    full_batch = move_batch(full_batch, device, model_dtype)
    input_ids = full_batch["input_ids"]
    target_ids = input_ids[:, prompt_length:]
    if target_ids.numel() == 0:
        return 0.0

    with torch.inference_mode():
        outputs = model(**full_batch, use_cache=True)
        logits = outputs.logits[:, prompt_length - 1 : -1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        token_scores = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    score = float(token_scores.sum().cpu())
    del prompt_batch, full_batch, outputs, logits, log_probs, token_scores
    clear_cache(device)
    return score


def generate_target(
    model,
    processor,
    messages: list[dict[str, Any]],
    device: torch.device,
    model_dtype: torch.dtype,
    max_new_tokens: int,
) -> str:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = move_batch(inputs, device, model_dtype)
    prompt_length = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    text = processor.decode(generated[0][prompt_length:], skip_special_tokens=True).strip()
    del inputs, generated
    clear_cache(device)
    return text
