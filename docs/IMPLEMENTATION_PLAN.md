# Minimal Implementation Plan

This is the smallest honest implementation of the proposal using the data currently in
the repository. It completes the text/RAG workflow now and leaves microscopy behind an
explicit data gate.

## What Is Usable

- `data/clinical_text/`: 1,000 synthetic, pre-validation cases split into train/validation/test.
- `data/clinical_text/hallucination_probe.jsonl`: 60 referral-only cases for over-confidence testing.
- `data/rag_knowledge_base/kb_chunks.jsonl`: 71 chunks with real citations and URLs.
- `scripts/build_chroma_index.py`: validates/builds the persistent vector index.
- `scripts/build_rag_prompt.py`: retrieves species-filtered treatment references and builds a grounded prompt.
- `scripts/build_instruction_set.py`: emits MedGemma-compatible T2 instruction records.
- `data/microscopy/`: acquisition and ETL plan, but no real images are currently present.

## What Must Not Be Claimed Yet

- Do not report T1 microscopy accuracy or Grad-CAM localisation: there are no image files.
- Do not call the synthetic treatment records clinical data. They require Cherehani Labs clinical sign-off.
- Do not train a mixed T1/T2 model until real normalised microscopy images exist.
- Public microscopy proxies are not equivalent to labelled East African veterinary slides.

## Run The Available Pipeline

From the repository root:

```bash
python3 scripts/validate_clinical_cases.py --dir data/clinical_text
python3 scripts/build_chroma_index.py --dry-run
python3 scripts/build_instruction_set.py --t2-only --exclude-proxy
```

The last command writes `data/instruction_tuning/{train,val,test}.jsonl`, plus the
evaluation-only `hallucination_probe.jsonl` and `stats.json`. These are suitable for
testing formatting and preprocessing, not for clinical deployment.

Install the RAG dependencies and build the local index:

```bash
python3 -m pip install chromadb sentence-transformers
python3 scripts/build_chroma_index.py
python3 scripts/build_rag_prompt.py CHL-SYN-000859 --split test
python3 scripts/build_rag_prompt.py CHL-SYN-P00001 --split probe
```

The final command prints the exact prompt to pass to a text-capable MedGemma inference
wrapper. It deliberately requires references, prevents unsupported dosing claims, and
allows referral when evidence is inadequate. `build_rag_prompt.py` does not download a
model and does not perform inference.

## Add Microscopy Later

1. Obtain labelled real slides and record their source, licence, host, specimen, and slide/patient group.
2. Put each source under `data/microscopy/raw/<dataset-id>/` using the folder names mapped in
   `data/microscopy/dataset_manifest.json`.
3. Run `python3 scripts/prepare_microscopy.py --local`.
4. Confirm that `data/microscopy/images/{train,val,test}/` contains real PNGs and that the
   prepared manifest has no failed rows.
5. Build the mixed set with `python3 scripts/build_instruction_set.py --exclude-proxy`.
6. Only then adapt the official Google MedGemma fine-tuning notebook for QLoRA. Use the exact
   model revision and current processor/collator from that notebook; do not hand-roll image
   token handling.
7. Evaluate base and fine-tuned models on the same held-out split. Keep T1 proxy classes out of
   the veterinary test report.

## Recommended Dissertation Scope

Make T2 RAG-grounded recommendation generation the primary executable experiment now. Treat
T1 and multimodal QLoRA as conditional on receiving Cherehani microscopy slides. If those slides
do not arrive, report the microscopy pipeline as implemented infrastructure and the missing
classes as a data-availability limitation rather than manufacturing images or labels.
