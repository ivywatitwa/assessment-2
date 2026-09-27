# Veterinary microscopy model study

This repository contains the code, data records, experiment notes, and dissertation for a secondary-data study of veterinary microscopy classification and literature retrieval. The image task uses the public Tryp mouse-model dataset. It does not evaluate veterinary clinical cases or establish performance on Kenyan cattle.

## Start here

For a quick review, use this order:

1. [Study scope and results](docs/REVIEW_GUIDE.md)
2. [Dissertation draft](docs/THESIS_DRAFT.md)
3. [Formatted dissertation](Thesis.docx)
4. [MedGemma results](docs/T1_VALIDATION_RESULTS.md)
5. [MiniCPM-V results](docs/T1_MINICPM_RESULTS.md)
6. [Implementation and evidence map](docs/IMPLEMENTATION_GUIDE.md)

The thesis source is `docs/THESIS_DRAFT.md`. Regenerate the formatted document with:

```bash
python scripts/build_thesis_docx.py --input docs/THESIS_DRAFT.md --output Thesis.docx
```

## Main findings

The 3,196 eligible Tryp images were split by source video into 2,231 training, 488 validation, and 477 test records. The original MedGemma adapter predicted every image as positive. A separately trained, class-balanced MiniCPM-V 4.6 adapter scored perfectly on validation, but its test result was much weaker:

| Model and split | Records | Accuracy | Macro-F1 | Negative-control recall | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| MedGemma, validation | 488 | 0.9672 | 0.4917 | 0.0000 | 0.5000 |
| MedGemma, test | 477 | 0.9748 | 0.4936 | 0.0000 | 0.5000 |
| MiniCPM-V 4.6, validation | 488 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| MiniCPM-V 4.6, test | 477 | 0.9769 | 0.5711 | 0.0833 | 0.5417 |

The MiniCPM test split contained only 12 negative controls. It correctly classified one and missed 11. These small, source-grouped results do not establish clinical usefulness or generalisation beyond the public mouse-model dataset. The repository also contains development-scale retrieval checks, synthetic auxiliary records, and diagnostic attribution runs. They are not clinical evidence.

## Repository map

| Path | Contents |
|---|---|
| `scripts/` | Data validation, preprocessing, training, evaluation, and document-generation programs |
| `data/secondary/` | Source register, public-data manifests and splits, derived records, and machine-readable reports |
| `data/clinical_text/` | Synthetic auxiliary T2 records, clearly marked as synthetic and not clinical data |
| `data/rag_knowledge_base/` | Cited veterinary knowledge-base source material; see its README and source register for provenance |
| `docs/` | Dissertation source, review guide, methods, experiment notes, and figures |
| `configs/` | Study configuration |
| `colab_*.ipynb` | Cloud training and evaluation workflows |
| `mlx_*.ipynb` | Apple Silicon MLX training and evaluation workflow |
| `Thesis.docx` | Formatted dissertation generated from the Markdown source |

Raw microscope images, model checkpoints, adapters, local vector indexes, credentials, and result bundles are kept outside version control. Download source datasets from their publishers and check the source register and each dataset's licence before use or redistribution.

## Reproduce basic checks

Use Python 3.12 or later. The project environment is defined in `pyproject.toml` and `uv.lock`. Run the provenance validator from the repository root:

```bash
python scripts/validate_secondary_sources.py
```

The MLX training notebook requires Apple Silicon, MLX-VLM, the converted base model, the prepared image files, and access to the model revision listed in the notebook. The Colab notebooks require their documented GPU runtime and source data. Training is not required to review the saved dissertation and result summaries.

## Data and use limits

- `data/clinical_text/` and `data/secondary/instruction_tuning_synthetic/` contain generated examples. They are auxiliary development/training material, not patient records or secondary clinical evidence.
- The saved source audit reports 37 unverified local-intake records and zero records included in T1. Their raw manifest and images are kept locally and are not part of this repository.
- Tryp contains microscopy images from a mouse model. Its labels do not establish a diagnosis in livestock or companion animals.
- Retrieval scores are small development or passage-identification checks. They do not measure treatment-answer factuality or clinical safety.
- The models and example outputs are for research review. They are not veterinary advice or a deployed diagnostic tool.
- Consult the source register and applicable publisher terms before redistributing third-party material.
