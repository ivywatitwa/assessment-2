# Review guide

## Suggested reading order

1. `README.md` for the project overview and key results.
2. `docs/THESIS_DRAFT.md` for the dissertation source.
3. `Thesis.docx` for the formatted dissertation.
4. `docs/T1_VALIDATION_RESULTS.md` for the MedGemma metrics and interpretation.
5. `docs/T1_MINICPM_RESULTS.md` for the balanced MiniCPM-V validation and test results.
6. `docs/IMPLEMENTATION_GUIDE.md` for the code and pipeline details.

## Completed evidence

- Public Tryp image data were parsed into 3,196 eligible records and split by source video.
- The MedGemma adapter was evaluated on validation and test. It predicted every record as positive, so its high accuracy reflects class imbalance rather than balanced classification.
- The class-balanced MiniCPM-V adapter scored 1.000 on validation and 0.5711 macro-F1 on test. It identified only 1 of 12 test negative controls. See `docs/T1_MINICPM_RESULTS.md` for confusion matrices, metrics, and limits.
- The saved source audit records 37 unverified local-intake entries, none used in T1. Their raw manifest and images remain local and are excluded from the repository.
- Local Python metadata processing accepted 3,200 rows in one 0.063-second run. This was not a Spark run.
- A four-query Chroma development check and a separate ten-query SQLite FTS5 check were completed. Neither measures generated-answer factuality.
- Synthetic T2 and attribution outputs are diagnostics, not clinical or validated explanation results.

## Review limits

The Tryp images come from a mouse model and support only a bounded image-classification task. The results do not establish veterinary diagnostic performance, performance on Kenyan cattle, treatment safety, clinician trust, or readiness for deployment. Synthetic records are labelled auxiliary material and are not clinical observations. The source register and cited source terms govern any data reuse.

## Rebuild the formatted dissertation

From the repository root, run:

```bash
python scripts/build_thesis_docx.py --input docs/THESIS_DRAFT.md --output Thesis.docx
```
