# Class-Balanced T1 Run Protocol

This protocol defines one additional public T1 training run. It does not alter the
frozen validation or test records and does not use the test split for model selection.

## Training

- Model: `google/medgemma-4b-it`
- Revision: `290cda5eeccbee130f987c4ad74a59ae6f196408`
- Epochs: `1`
- Learning rate: `2e-4`
- Gradient accumulation: `8`
- Compute dtype: FP32
- Seed: `42`
- Sampling: deterministic class balancing on the training split only
- Output directory: `/content/medgemma-t1-qlora-balanced`

Before the full pass, run the four-record smoke test in the training notebook. It
checks model loading, multimodal collation, loss, backpropagation and adapter
writing without spending the full run time. Do not interpret the smoke output as
an evaluation result.

The source training split contains 2,166 `trypanosome_present` records and 65
`negative_control` records. The balanced sampler cycles the minority records and
interleaves the two classes, producing 4,332 effective training examples. It does
not create new images or labels. The 65 negative records come from only four source
groups, so this run mitigates record-count imbalance but not source-group scarcity.

For the requested base-model comparator, evaluate the same pinned checkpoint on
the validation split without an adapter:

```bash
python scripts/train_t1_qlora.py \
  --root /content/medgemma-secondary \
  --output /content/medgemma-t1-base-validation \
  --base-only --split val --max-val 488 --max-new-tokens 8
```

This is an inference baseline, not a trained baseline. Preserve its metrics and
predictions separately from the adapter output. Do not use its test output to
tune the adapter.

## Selection Gate

Run validation immediately after training. Do not load or evaluate the test split
at this stage. Continue to the test evaluation only if the balanced run has both:

- positive negative-control recall; and
- validation macro-F1 above the existing `0.4917` result.

If either condition fails, retain the existing validation/test result in the thesis
and do not spend additional compute on another T1 run.

## Final Test

After the validation gate passes, freeze the adapter and all settings. Evaluate the
same adapter exactly once on the 477-record test split using
`colab_t1_balanced_test.ipynb`. Do not retrain, alter prompts or tune thresholds
after viewing test predictions. The full project names this split `test.jsonl`.
The compact test bundle deliberately stores the same held-out records in
`val.jsonl`, so the notebook's `val.jsonl` selection must not be changed when
using that bundle; the records' metadata and image paths still identify `test`.

The Colab training notebook is `colab_t1_balanced_train.ipynb`. It uploads the full
`medgemma-secondary-kaggle.tar` archive and overwrites the archive's stale trainer
with the updated `scripts/train_t1_qlora.py`. The archive is a project bundle name,
not a Kaggle attachment requirement.

The original adapter and result archives remain unchanged. The balanced adapter is
an additional experiment and must not replace the original result unless its
validation and one-time test outputs are both archived and documented.
