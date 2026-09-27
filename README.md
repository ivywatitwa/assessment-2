# MedGemma Veterinary Diagnostics — Data & Pipeline

Supporting code and datasets for the MSc Big Data Technologies dissertation:

> **Fine-Tuning MedGemma for Multimodal Explainable Veterinary Diagnostics:
> Microscopy Image Analysis and Treatment Recommendation via a Big Data Pipeline
> at Cherehani Labs, Kenya**
> University of East London / UNICAF — UEL-CN-7000

---

## ⚠️ Read before using any data here

**The clinical text records in `data/clinical_text/` are SYNTHETIC.** They were
generated programmatically for model training and evaluation. They are:

- **not real patient records**
- **not veterinary advice**
- **not validated** — every record carries `requires_expert_validation: true`

They must be reviewed and signed off by Cherehani Labs clinical staff before use
in any experiment whose results are reported. Do not use any treatment protocol
in this repository to treat an animal.

The RAG knowledge base in `data/rag_knowledge_base/` is the opposite: it is
**curated from real, verified literature**. Every chunk carries a real citation
and a URL that was actually fetched. Nothing in it is generated. That is
deliberate — it is the grounding corpus whose entire purpose is to reduce
hallucination, and the dissertation's hallucination-rate metric depends on it
being real.

---

## Layout

```
data/
  clinical_text/         synthetic clinical cases (JSONL) — 1,060 records
  rag_knowledge_base/    71 chunks from 20 verified sources, for ChromaDB
  microscopy/            taxonomy, dataset manifest, ETL outputs
  instruction_tuning/    merged T1+T2 instruction set (built by scripts/)
scripts/                 generators, validators, Spark ETL, RAG index/query
docs/
  PROJECT_SPEC.md        canonical schemas + taxonomy (the contract)
  IMPLEMENTATION_GUIDE.md  step-by-step runbook for all 5 phases
  METHODOLOGICAL_RISKS.md  candid risks — seeds the Limitations chapter
```

## Quick start

```bash
# 1. Regenerate the synthetic clinical cases (deterministic, seeded)
python3 scripts/generate_clinical_cases.py --n 1000 --seed 42
python3 scripts/validate_clinical_cases.py        # 16 checks, must exit 0

# 2. Inspect the microscopy plan without downloading anything
python3 scripts/prepare_microscopy.py --dry-run

# 3. Build the RAG index (needs: pip install chromadb sentence-transformers)
python3 scripts/build_chroma_index.py --dry-run   # validates chunks, no deps
python3 scripts/build_chroma_index.py

# 4. Merge both tasks into the single instruction-tuning set
python3 scripts/build_instruction_set.py --exclude-proxy
```

For the current repository state, use `--t2-only` because no microscopy image files are
present yet. The complete, honest runbook is in [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

For the revised dissertation that prohibits first-hand data collection, use the
[secondary-data implementation plan](docs/SECONDARY_DATA_IMPLEMENTATION_PLAN.md), the
[step-by-step guide](docs/SECONDARY_DATA_STEP_BY_STEP_GUIDE.md), and the generated
[thesis draft](thesis.docx).

## Known hard constraints

These are documented at length in `docs/METHODOLOGICAL_RISKS.md`. The three that
change what is possible:

1. **East Coast Fever has zero public microscopy data.** So do *Anaplasma*,
   *Eimeria* and *Haemonchus*. Four of the seven veterinary target classes depend
   entirely on Cherehani Labs supplying labelled slides. Augmentation cannot
   manufacture a class with no examples. Confirm slide access before Phase 1.

2. **The model must be `google/medgemma-4b-it`.** The 27B variant is text-only
   and cannot ingest an image, so it cannot perform Task 1. The 4B weights are
   licence-gated behind Health AI Developer Foundations terms — accept them on
   Hugging Face before you need them.

3. **Grad-CAM does not apply cleanly.** MedGemma's vision tower is a SigLIP ViT,
   not a CNN, and the model is generative, so there is no softmax class score to
   differentiate. See `docs/IMPLEMENTATION_GUIDE.md` §5 for the three ranked
   workarounds and disclose which you used.

## Licence / provenance

Third-party dataset licences are recorded per-dataset in
`data/microscopy/dataset_manifest.json` (they differ — BBBC041 is CC BY-NC-SA,
Chula-ParasiteEgg is gated). Source citations for the RAG corpus are in
`data/rag_knowledge_base/sources.json`, which doubles as the dissertation's
data-provenance appendix.
