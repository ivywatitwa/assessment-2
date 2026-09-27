# T1 Validation and Test Results

## Run provenance

The validation and test evaluations used the same saved adapter. Raw prediction archives and model weights are local files and are excluded from version control. This report records the metrics and their interpretation.

The base model was:

- Model: `google/medgemma-4b-it`
- Revision: `290cda5eeccbee130f987c4ad74a59ae6f196408`
- Task: public T1 microscopy classification

The training log recorded 2,231 public T1 training records, one epoch, 279 optimizer updates, gradient accumulation of 8, learning rate `2e-4` and FP32 computation. The adapter file is `adapter_model.safetensors`.

## Held-Out Validation

The validation archive contains 488 prediction records and no invalid generated labels.

| Metric | Value |
|---|---:|
| Records | 488 |
| Groups | 7 |
| Accuracy | 0.967213 |
| Macro-F1 | 0.491667 |
| Invalid outputs | 0 |

Label order in the confusion matrix is `[negative_control, trypanosome_present]`:

```text
[[0, 16],
 [0, 472]]
```

The model predicted `trypanosome_present` for all 488 validation images:

- `negative_control`: 0 of 16 correctly identified.
- `trypanosome_present`: 472 of 472 correctly identified.
- Negative-control recall/specificity: 0.0.
- Balanced accuracy: 0.5.

The validation labels are highly imbalanced: 472 positive and 16 negative records. The training split contains 2,166 positive and 65 negative records. Therefore, the 0.967213 accuracy is approximately the positive-majority baseline and must not be presented as evidence of useful balanced diagnostic classification. Macro-F1 and the confusion matrix are the more informative results.

## Held-Out Test

The test archive contains 477 prediction records and no invalid generated labels. The evaluation-only notebook used the saved adapter without retraining. The adapter payload has the same SHA-256 hash in the validation and test archives (`cf5d5c28f4af16be3e823ba6e469121e8bec7a17e83e53f00ea640ee2cc66026`).

| Metric | Value |
|---|---:|
| Records | 477 |
| Groups | 21 |
| Accuracy | 0.974843 |
| Macro-F1 | 0.493631 |
| Invalid outputs | 0 |

The test confusion matrix, using label order `[negative_control, trypanosome_present]`, is:

```text
[[0, 12],
 [0, 465]]
```

The model predicted `trypanosome_present` for all 477 test images. It correctly identified 465 of 465 positive images and 0 of 12 negative controls, giving negative-control recall/specificity of 0.0 and balanced accuracy of 0.5. The test result confirms the validation finding of majority-class collapse on a separate held-out split; it does not demonstrate useful balanced classification or clinical performance.

## Reporting Boundary

This is a computational result on the public Tryp proxy task. It is not evidence of performance on Kenyan cattle, Cherehani Labs data, field samples or clinical deployment. The result indicates that the current one-epoch adapter collapsed to the majority class and requires class-balanced retraining or a class-sensitive objective before claiming effective T1 classification.

The `training_summary.json` included in each Colab evaluation archive reports the evaluated split as `train_records` and a null final loss because the eval-only run used a compact bundle whose placeholder `train.jsonl` was a copy of the evaluated split. It is not the original training provenance and should not be used for the training-count claim above. The original training provenance remains 2,231 public T1 training records, one epoch and 279 optimizer updates.
