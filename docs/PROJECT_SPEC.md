# Shared Project Spec — MedGemma Veterinary Diagnostics

> Canonical reference for all dataset generation. Every generated artefact MUST conform
> to the taxonomy, schemas, and labelling rules below so the components interlock.

Derived from: *Fine-Tuning MedGemma for Multimodal Explainable Veterinary Diagnostics:
Microscopy Image Analysis and Treatment Recommendation via a Big Data Pipeline at
Cherehani Labs, Kenya* (Assessment 1, Research Proposal, UEL-CN-7000).

---

## 0. Non-negotiable labelling rule

Every synthetic record MUST carry:

```json
{ "provenance": "synthetic", "clinical_use": false, "requires_expert_validation": true }
```

Synthetic clinical content is **research training data only**. It is not veterinary advice
and must never be presented as authoritative clinical guidance. The proposal requires that
synthetic cases be "validated by Cherehani Labs clinical staff" before use — the datasets
generated here are therefore **pre-validation drafts** pending that sign-off.

---

## 1. Two model tasks (per proposal §3.2, Phase 3)

| Task | Input | Output | XAI method |
|---|---|---|---|
| **T1 — Microscopy classification** | Slide image | Pathogen class | Grad-CAM |
| **T2 — Treatment recommendation** | Clinical notes + C&S report (+RAG context) | Treatment plan | SHAP token attribution |

Both are mixed into **one instruction-tuning set** (proposal: "mixed in a single training
set via instruction tuning").

---

## 2. Microscopy pathogen taxonomy (T1)

East-Africa-relevant haemoparasites and enteric parasites. Class IDs are stable.

| id | class | specimen | host | disease |
|---|---|---|---|---|
| 0 | `uninfected` | blood smear | any | — (negative control) |
| 1 | `trypanosoma_spp` | blood smear | cattle, dog | Trypanosomiasis (Nagana) |
| 2 | `theileria_parva` | blood/lymph smear | cattle | East Coast Fever |
| 3 | `babesia_spp` | blood smear | cattle, dog | Babesiosis |
| 4 | `anaplasma_spp` | blood smear | cattle | Anaplasmosis |
| 5 | `eimeria_spp` | faecal float | cattle, goat, sheep | Coccidiosis |
| 6 | `haemonchus_contortus` | faecal float | goat, sheep | Haemonchosis |
| 7 | `plasmodium_falciparum` | blood smear | human (proxy) | Malaria — **transfer-learning proxy only** |

`plasmodium_falciparum` exists solely because the proposal specifies the NIH Malaria Cell
Images Dataset for **microscopy pre-training** (proposal §1.5, §3.2 Phase 1). It is a
pre-training class, excluded from the veterinary evaluation test set.

---

## 3. Disease scope for treatment recommendation (T2)

| Species | Diseases |
|---|---|
| Cattle | East Coast Fever, Trypanosomiasis, Anaplasmosis, Babesiosis, Bovine Mastitis, Lumpy Skin Disease |
| Goat / Sheep | Peste des Petits Ruminants (PPR), Contagious Caprine Pleuropneumonia, Coccidiosis, Haemonchosis |
| Dog | Canine Babesiosis, Canine Trypanosomiasis, Ehrlichiosis, Canine Parvovirus |
| Cat | Feline Upper Respiratory Complex, Haemoplasmosis |

The four diseases named in the proposal abstract/§1.1 — **East Coast Fever,
Trypanosomiasis, Peste des Petits Ruminants, Mastitis** — are the priority evaluation set
and must be over-represented relative to the long tail.

Mastitis is the **only** disease for which culture & sensitivity (C&S) reports are
routinely available; C&S fields are populated for mastitis and other bacterial cases, and
explicitly `null` elsewhere. The proposal says recommendations use "lab technician notes
and culture and sensitivity reports **when available**" — the `null` case is therefore a
first-class scenario the model must handle, not a data defect.

---

## 4. Canonical schemas

### 4.1 Clinical case record (`data/clinical_text/*.jsonl`)

```json
{
  "case_id": "CHL-SYN-000001",
  "provenance": "synthetic",
  "clinical_use": false,
  "requires_expert_validation": true,
  "species": "cattle",
  "breed": "Boran",
  "age_months": 18,
  "weight_kg": 180.0,
  "sex": "female",
  "county": "Narok",
  "presenting_signs": ["pyrexia", "lymphadenopathy", "dyspnoea"],
  "vitals": { "temp_c": 40.8, "hr_bpm": 88, "rr_bpm": 40 },
  "microscopy_finding": { "class": "theileria_parva", "parasitaemia_pct": 3.2 },
  "culture_sensitivity": null,
  "diagnosis": "East Coast Fever",
  "differentials": ["Trypanosomiasis", "Anaplasmosis"],
  "treatment": {
    "primary_drug": "Buparvaquone",
    "dose_mg_per_kg": 2.5,
    "route": "IM",
    "frequency": "single dose",
    "duration_days": 1,
    "supportive_care": ["NSAID for pyrexia", "fluid therapy"]
  },
  "follow_up_days": 7,
  "withdrawal_period_days": 42,
  "clinician_rationale": "…"
}
```

### 4.2 Instruction-tuning record (`data/instruction_tuning/*.jsonl`)

Chat format, MedGemma / HF PEFT compatible. `image` is `null` for T2.

```json
{
  "task": "T2_treatment",
  "image": null,
  "messages": [
    { "role": "user",   "content": "…clinical notes + C&S…" },
    { "role": "assistant", "content": "…grounded treatment plan…" }
  ],
  "meta": { "case_id": "CHL-SYN-000001", "diagnosis": "East Coast Fever", "provenance": "synthetic" }
}
```

For T1: `"task": "T1_microscopy"`, `"image": "data/microscopy/images/<split>/<class>/<file>.png"`.

### 4.3 RAG document (`data/rag_knowledge_base/*.json`)

Chunked for ChromaDB ingestion. Every chunk carries a citable source.

```json
{
  "doc_id": "vet-kb-ecf-001",
  "title": "East Coast Fever — Aetiology, Diagnosis and Treatment",
  "disease": "East Coast Fever",
  "species": ["cattle"],
  "section": "treatment",
  "text": "…",
  "source": { "citation": "…APA…", "type": "peer_reviewed | manual | guideline", "url": "…" },
  "provenance": "curated_from_literature"
}
```

RAG corpus is **curated from real literature**, NOT synthetic — it is the grounding
substrate whose whole purpose is to reduce hallucination. Fabricated sources would defeat
the RAG hypothesis (proposal §2.3) and invalidate the hallucination-rate metric (§3.3).

---

## 5. Splits

Stratified by class, grouped so no `case_id` leaks across splits.

| split | fraction |
|---|---|
| train | 0.70 |
| val | 0.15 |
| test | 0.15 |

The **test set must contain zero synthetic-only diseases absent from train**, and the
priority four (ECF, Trypanosomiasis, PPR, Mastitis) must appear in all three splits.

---

## 6. Evaluation targets (proposal §3.3)

- T1: accuracy, macro-F1, Grad-CAM localisation accuracy
- T2: ROUGE-L, factual-consistency score, hallucination rate
- Both: base MedGemma vs QLoRA fine-tuned MedGemma

A held-out **hallucination probe set** is required: cases whose correct answer is
"insufficient evidence / refer to clinician", used to measure over-confident fabrication.

---

## 7. Directory contract

```
data/
  clinical_text/         synthetic clinical cases (JSONL) + generator
  rag_knowledge_base/    curated veterinary literature chunks for ChromaDB
  microscopy/            manifests, class taxonomy, download + prep scripts
  instruction_tuning/    merged T1+T2 instruction set, split into train/val/test
scripts/                 generators, ETL, download helpers
docs/                    data documentation, datasheet, implementation runbook
```
