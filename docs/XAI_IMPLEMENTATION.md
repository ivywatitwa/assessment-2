# XAI Implementation

## Scope

The repository now contains executable, model-agnostic attribution fallbacks for the generative MedGemma architecture:

- T1 image tile occlusion in `scripts/attribute_t1_occlusion.py`.
- T2 clinical-field occlusion or KernelSHAP in `scripts/attribute_t2_occlusion.py`.
- Synthetic-only T2 generation in `scripts/run_t2_generation.py`.
- Synthetic T2 field-match and referral diagnostics in `scripts/evaluate_t2_generation.py`.
- Attribution summaries in `scripts/evaluate_xai.py`.

These methods are intentionally not named Grad-CAM or SHAP. MedGemma generates text and does not expose a verified scalar image-classification head or a fixed SHAP masking game in the current implementation. The occlusion methods provide an auditable fallback while avoiding unsupported explanation claims.

## T1 Image Attribution

The runner scores the fixed class margin:

```text
log P(trypanosome_present) - log P(negative_control)
```

It masks a regular image grid, recomputes the margin and records the score change for every tile. Positive score changes identify tiles supporting the positive class. If the prepared manifest is supplied, the evaluator also calculates positive attribution mass overlapping publisher-provided boxes and pointing-game accuracy.

Example:

```bash
python scripts/attribute_t1_occlusion.py \
  --root /path/to/bundle \
  --input /path/to/bundle/data/secondary/instruction_tuning_mixed/val.jsonl \
  --adapter /path/to/bundle/medgemma-t1-qlora \
  --manifest /path/to/bundle/data/secondary/manifests/images_prepared.jsonl \
  --output results/xai/t1_image_attributions.jsonl \
  --max-examples 4 \
  --grid-size 4
```

## T2 Attribution

The T2 runner treats the fields in a synthetic case as interpretable feature groups: species, breed, age, weight, sex, location, presenting signs, vitals, microscopy and culture/sensitivity. It supports one-field occlusion and an optional KernelSHAP mode over those field groups. Both methods measure effects on the teacher-forced score of the gold diagnosis line.

The generated outputs and attributions are synthetic auxiliary diagnostics. They are not clinical validation and do not establish factual treatment safety.

For KernelSHAP:

```bash
python scripts/attribute_t2_occlusion.py \
  --input /path/to/data/secondary/instruction_tuning_synthetic/val.jsonl \
  --adapter /path/to/medgemma-t1-qlora \
  --output results/xai/t2_shap.jsonl \
  --max-examples 4 \
  --method shap \
  --shap-samples 32
```

## Reporting

Attribution results should report the method, target score, number of records, perturbation protocol, mean attribution summaries and any alignment or referral metrics. A heatmap or token ranking alone is not evidence of faithful reasoning. The T2 KernelSHAP output is an attribution result, but it still requires perturbation and stability checks before being described as faithful. T1 remains an occlusion fallback rather than Grad-CAM.

The ready-to-run Colab workflow is `colab_xai_t2.ipynb`, and its input archive is `colab-xai-t2-bundle.tar.gz`.
