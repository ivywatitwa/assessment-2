# MiniCPM-V 4.6 T1 results

## Run

The class-balanced adapter used `openbmb/MiniCPM-V-4.6` at revision `36f34a661a4bd35d0dc2294cb044d2584646c7d3`. The one-epoch MLX-VLM LoRA training set contained 4,332 rows, formed by cycling the 65 negative training examples to balance the 2,166 positive examples. This balancing adds no new negative-image diversity.

The saved adapter was first evaluated on the 488-record validation split. The notebook then evaluated the same adapter on the unchanged 477-record test split. Both passes used 16 generated tokens and 4x visual token compression. The test run did not train or alter the adapter.

## Results

| Split | Records | Groups | Accuracy | Macro-F1 | Negative-control recall | Balanced accuracy | Invalid outputs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 488 | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| Test | 477 | 21 | 0.9769 | 0.5711 | 0.0833 | 0.5417 | 0 |

Rows in these confusion matrices are true labels. Columns are predictions. Label order is `[negative_control, trypanosome_present]`.

```text
Validation: [[16, 0], [0, 472]]
Test:       [[ 1, 11], [0, 465]]
```

The validation split contained 16 negative controls from one source-video group. The test split contained 12 negative controls. The adapter correctly classified one negative and missed eleven, while correctly classifying all 465 positive images. Its test accuracy is close to the positive-majority baseline. Macro-F1, negative-control recall, balanced accuracy, and the confusion matrix show the class-imbalance failure more clearly.

## Interpretation

The perfect validation result did not predict balanced performance on the test split. The MiniCPM test result is an exploratory evaluation of one saved adapter on a small, grouped public dataset. No confidence intervals, additional training seeds, base-model comparison, external dataset, or clinical evaluation were run. The result cannot establish performance on livestock, veterinary patients, or clinical use.

The notebook `mlx_t1_balanced_train_minicpm_v46.ipynb` contains the evaluation commands and their recorded outputs. Model weights and raw prediction files remain local and are excluded from version control. The reported summary is also recorded in `data/secondary/reports/minicpm_t1_metrics.json`.
