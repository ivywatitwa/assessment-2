# MLX-VLM MiniCPM-V 4.6 T1 dataset

This export follows the current MLX-VLM LoRA JSON dataset structure: each JSONL
row contains an `images` list and model-formatted `messages`. Image paths are
relative to the repository root; the trainer changes its working directory to
`--root` before MLX-VLM opens them.

- `train.jsonl`: 4,332 rows, balanced to 2,166 per class by cycling minority
  training records (seed 42). This duplicates existing negative examples; it
  does not add negative image diversity.
- `val.jsonl`: the original 488-record validation split.
- `test.jsonl`: the original 477-record held-out split. The trained adapter was
  evaluated on this split after validation; results are summarised in
  `docs/T1_MINICPM_RESULTS.md`.
- `dataset_info.json`: schema, model, prompt, source and balancing metadata.

Regenerate the export from the canonical mixed instruction splits with:

```bash
python scripts/build_mlx_t1_dataset.py --root .
```

The user prompt requests a specific Trypanosoma microscopy classification and
exactly one of `trypanosome_present` or `negative_control`.
