# Remaining empirical runs before a revised submission

The current `Thesis.docx` includes measured local Python ETL and SQLite FTS5 results, but **does not contain a new T1 base/balanced comparison, Spark run or held-out Chroma score**. Run these commands in an environment with working network/package installation; save the resulting JSONL/JSON artifacts and update the thesis from the actual metrics before claiming them.

## 1. T1: one frozen validation comparison (CUDA GPU)

Use a Colab CUDA GPU runtime with working package downloads, sufficient runtime/disk space, and Hugging Face access to the gated `google/medgemma-4b-it` model (revision pinned in the trainer). Upload or mount the 2.7-GB `medgemma-secondary-kaggle.tar`, the current `scripts/train_t1_qlora.py`, `scripts/compare_t1_runs.py`, `scripts/evaluate_t1.py`, and the original `t1-validation-results.zip`. Unpack the archive to `/content/medgemma-secondary` as in `colab_t1_balanced_train.ipynb`, then overwrite the archived scripts with the three current scripts. Use `getpass`/Colab secrets for `HF_TOKEN`; do not paste it into a document or chat. The base run needs no adapter. Do not use `--max-train` for the full balanced run. Run the notebook's four-record smoke pass before the full pass; archive the final training and validation output directory before resetting the runtime.

```bash
python /content/medgemma-secondary/scripts/train_t1_qlora.py --root /content/medgemma-secondary --output /content/medgemma-t1-base-validation --base-only --split val
python /content/medgemma-secondary/scripts/train_t1_qlora.py --root /content/medgemma-secondary --output /content/medgemma-t1-qlora-balanced --class-balanced --split val
python /content/medgemma-secondary/scripts/compare_t1_runs.py --original /content/t1-validation-results.zip --base /content/medgemma-t1-base-validation/t1_predictions.jsonl --balanced /content/medgemma-t1-qlora-balanced/t1_predictions.jsonl --out /content/t1-validation-comparison.json
```

The comparator refuses mismatched record IDs, gold labels and groups. Its matrices carry the label order and include an invalid-output column if needed. Macro-F1 and balanced accuracy use the two true classes. Inspect the validation comparison before deciding whether to run the unchanged grouped test split *once* with `colab_t1_balanced_test.ipynb`, per `docs/T1_BALANCED_RUN_PROTOCOL.md`. Archive the base/balanced raw predictions and metrics. Do not fill in missing results by extrapolating from the old adapter.

### Separate ResNet baseline (`baseline_t1.py`)

This is a **different model** (frozen ImageNet ResNet-18 plus balanced logistic regression), not the MedGemma base model. It needs `torch`, `torchvision`, `scikit-learn`, `numpy`, `Pillow` and an internet-accessible download or local cache of the pretrained ResNet-18 weights. Copy `baseline_t1.py` into `/content`. The archive contains the prepared PNGs and split manifests but omits the split manifest's original JPG paths; select `--image-source prepared`:

```bash
python /content/baseline_t1.py --splits-dir /content/medgemma-secondary/data/secondary/splits/images --root /content/medgemma-secondary --image-source prepared --check-inputs
python /content/baseline_t1.py --splits-dir /content/medgemma-secondary/data/secondary/splits/images --root /content/medgemma-secondary --image-source prepared --validation-only --out /content/resnet-t1-validation.json
```

The preflight already passed locally (2,231 training, 488 validation, 477 test image paths and no group overlap). To avoid using test results in model selection, run the same command without `--validation-only`, writing `/content/resnet-t1-final.json`, **only after** the MedGemma validation decision has been frozen. Download both JSON files. A strong ResNet result would show the *task* is learnable under this protocol; it would not measure MedGemma's before/after gain.

## 2. Local PySpark metadata ETL

With Java 17 and a compatible PySpark installation (`python -m pip install pyspark==3.5.8`):

```bash
python scripts/spark_etl_secondary.py --metadata-only --master 'local[2]' --out-dir data/secondary/spark_processed
```

Inspect `data/secondary/spark_processed/spark_etl_report.json` for the actual Spark version, count, rejection reasons and elapsed time. Compare counts with `data/secondary/processed_full/etl_report.json` (3,200 accepted locally in Python). The Python 0.063-second time is not a Spark time or fair speed comparison; repeat both under controlled conditions for performance claims.

## 3. Same retriever, new retrieval questions

Once the existing Chroma/SentenceTransformer environment is available, keep the existing index and filters unchanged:

```bash
python scripts/evaluate_retrieval.py --input data/secondary/text/retrieval_heldout.jsonl -k 5 --out data/secondary/reports/retrieval_heldout_chroma.json
```

Keep this result distinct from `retrieval_heldout_lexical.json`, which is an actual **SQLite FTS5** result over the same KB, not a Chroma output. The ten additional disease groups span nine unique reference pages. Both sets use in-corpus, source-derived labels; new independent case/source judgments would be needed for clinical retrieval conclusions.
