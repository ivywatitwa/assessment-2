# Secondary-data study workspace

This directory is the input boundary for the revised dissertation. Its core evidence
is publicly sourced or source-derived. It may also contain a separately marked
synthetic auxiliary corpus for T2 engineering; that corpus is never treated as
secondary evidence or clinical evidence. It must not contain:

- first-hand laboratory, animal, patient or participant data;
- unverified local-intake images;
- the repository's synthetic clinical cases mixed into the public-evidence results; or
- generated claims presented as observations from a published case.

The source register is the authority for eligibility. Raw downloads remain outside
version control when their licence or size requires re-acquisition from the publisher.
The current repository does not contain the Tryp archive or a MedGemma checkpoint.

## Files

- `sources/source_register.json`: source provenance and eligibility register.
- `text/retrieval_eval.jsonl`: four source-derived retrieval development items. This
  is a smoke benchmark for the existing knowledge base, not a sufficient clinical
  evaluation set.
- `instruction_tuning_synthetic/`: explicitly labelled synthetic T2 records copied
  from `data/clinical_text/` for auxiliary training and pipeline stress tests only.
  Every record remains `clinical_use=false` and requires expert validation. These
  records are not secondary evidence.
- `manifests/`: generated image/text manifests and frozen group splits.
- `instruction_tuning/`: generated mixed-task records once eligible public image/text
  data have been prepared.
- `reports/`: validation, split and retrieval reports.

Run the audit without downloading data:

```bash
python3 scripts/validate_secondary_sources.py
```

Run the retrieval smoke benchmark after the local Chroma index exists:

```bash
python3 scripts/evaluate_retrieval.py --input data/secondary/text/retrieval_eval.jsonl
```
