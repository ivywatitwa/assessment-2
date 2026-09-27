#!/usr/bin/env python3
"""Export an mlx-vlm LoRA adapter as a standard Transformers/PEFT adapter.

The LoRA matrices are transposed because mlx-vlm stores A=(in, rank),
B=(rank, out), while PEFT stores A=(rank, in), B=(out, rank).
This converts the adapter format only; it does not convert the MLX base model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

MODEL_ID = "google/medgemma-4b-it"
MODEL_REVISION = "290cda5eeccbee130f987c4ad74a59ae6f196408"
TARGET_MODULES = ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"]


def convert(input_dir: Path, output_dir: Path) -> None:
    config_path = input_dir / "adapter_config.json"
    weights_path = input_dir / "adapters.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Expected MLX adapter_config.json and adapters.safetensors in {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    mlx_config = json.loads(config_path.read_text(encoding="utf-8"))
    if mlx_config.get("fine_tune_type") != "lora":
        raise ValueError("Only standard LoRA adapters can be exported")
    params = mlx_config.get("lora_parameters", {})
    rank = int(params.get("rank", 0))
    alpha = float(params.get("scale", 0)) * rank
    if rank < 1 or alpha <= 0:
        raise ValueError(f"Invalid MLX LoRA parameters: {params}")

    mlx_weights = load_file(str(weights_path))
    peft_weights: dict[str, np.ndarray] = {}
    seen: set[tuple[str, str]] = set()
    for key, value in mlx_weights.items():
        if key.endswith(".lora_a"):
            side = "A"
        elif key.endswith(".lora_b"):
            side = "B"
        else:
            raise ValueError(f"Unexpected MLX adapter tensor name: {key}")
        module = key.rsplit(".lora_", 1)[0]
        pair = (module, side)
        if pair in seen:
            raise ValueError(f"Duplicate LoRA tensor: {key}")
        seen.add(pair)
        # HF Gemma3ForConditionalGeneration exposes these paths below
        # model.language_model.model.layers...; PEFT prefixes them with
        # base_model.model when saving a causal-LM PEFT adapter.
        peft_key = f"base_model.model.{module}.lora_{side}.weight"
        peft_weights[peft_key] = np.ascontiguousarray(value.T)

    modules = {module for module, _ in seen}
    if not modules or any((module, side) not in seen for module in modules for side in ("A", "B")):
        raise ValueError("Every adapted module must have both LoRA A and B tensors")
    if len(modules) != len(mlx_config.get("lora_parameters", {}).get("keys", [])):
        raise ValueError("Tensor module set does not match adapter_config.json")
    for module in modules:
        a = peft_weights[f"base_model.model.{module}.lora_A.weight"]
        b = peft_weights[f"base_model.model.{module}.lora_B.weight"]
        if a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError(f"Rank/shape mismatch for {module}: A{a.shape}, B{b.shape}, r={rank}")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(peft_weights, str(output_dir / "adapter_model.safetensors"))
    peft_config = {
        "base_model_name_or_path": MODEL_ID,
        "revision": MODEL_REVISION,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": alpha,
        "lora_dropout": float(params.get("dropout", 0.0)),
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": TARGET_MODULES,
        "task_type": "CAUSAL_LM",
    }
    (output_dir / "adapter_config.json").write_text(
        json.dumps(peft_config, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "source_backend": "mlx-vlm",
        "source_model": MODEL_ID,
        "source_revision": MODEL_REVISION,
        "source_base_quantization": "MLX affine 4-bit, group size 64",
        "target_base_quantization": "original Hugging Face checkpoint (unquantized unless caller quantizes it)",
        "note": "PEFT-compatible tensor export. Outputs can differ from MLX because the trained base used MLX 4-bit quantization.",
    }
    (output_dir / "mlx_export_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {len(modules)} LoRA modules ({len(peft_weights)} tensors) to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="MLX adapter directory")
    parser.add_argument("--output", type=Path, required=True, help="New PEFT adapter directory")
    args = parser.parse_args()
    convert(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
