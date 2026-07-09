# Microscopy Image Dataset Layer

Data-acquisition and preparation infrastructure for **Task T1 — Microscopy classification**
(fine-tuning MedGemma for veterinary diagnostics, Cherehani Labs). Conforms to the class
taxonomy in [`../../docs/PROJECT_SPEC.md`](../../docs/PROJECT_SPEC.md) §2 (stable class IDs 0–7).

> **Honesty statement.** No synthetic microscopy images are generated here. This layer only
> (a) acquires *real, verified, public* datasets and (b) documents, without euphemism, which
> taxonomy classes have real public data and which do not. Classes with no public source are
> declared as dissertation data-availability limitations (proposal §3.5), not papered over.

## Contents

| File | Purpose |
|---|---|
| `class_taxonomy.json` | §2 taxonomy as machine-readable data (id, class, specimen, host, disease, is_pretraining_proxy). |
| `dataset_manifest.json` | Every real dataset: URL, licence, APA-7 citation, n_images, native classes, and explicit mapping onto §2. |
| `../../scripts/download_microscopy.py` | Idempotent/resumable downloader (streaming, checksum recording, gated-source detection). |
| `../../scripts/prepare_microscopy.py` | Spark ETL: ingest → normalise 896×896 → grouped-stratified split → parquet/JSONL manifest. |
| `raw/<dataset>/` | Downloaded source data (git-ignored). |
| `images/<split>/<class>/` | Normalised 896×896 PNGs (real runs). |
| `prepared/` | Split manifest (`split_manifest.jsonl`), `summary.json`, `manifest.parquet`. |

## Real datasets located (all verified July 2026)

| Dataset | n images | Licence | Academic OK | Covers (class) |
|---|---|---|---|---|
| **NIH/NLM Malaria Cell Images** (Rajaraman 2018) | 27,558 | US-Gov / NLM public (effectively public domain) | Yes | 7 (P. falciparum), 0 (uninfected) |
| **BBBC041v1** — P. vivax blood smears | 1,364 | CC BY-**NC-SA** 3.0 | Yes (non-commercial) | 7 (genus proxy), 0 |
| **Tryp** thick-smear trypanosome (Anzaku 2023) | 3,178 | CC BY 4.0 | Yes | 1 (Trypanosoma, proxy), 0 |
| **Mendeley** Microscopic Images of Parasites (Li & Zhang 2020) | 34,298 | CC BY 4.0 | Yes | 3 (**only** Babesia source), 1, 7, 0 |
| **Chula-ParasiteEgg-11** (ICIP 2022) | 13,750 | IEEE DataPort terms (confirm) | Conditional / **gated** | none directly — faecal-egg *morphology* proxy only |

Exact URLs, DOIs and full APA-7 citations are in `dataset_manifest.json`.

## Data Availability & Limitations (READ THIS — proposal §3.5)

Coverage of taxonomy §2, class by class:

| id | class | Public data? | Source / status |
|---|---|---|---|
| 0 | `uninfected` | **Yes** | Well covered (NIH uninfected cells, Mendeley RBC, Tryp negatives). |
| 1 | `trypanosoma_spp` | **Proxy only** | Tryp (T. b. brucei, *unstained thick* smear, mouse) + Mendeley (Giemsa). Both domain-shifted from field Giemsa-stained **cattle thin** smears. Supplement with Cherehani slides. |
| 2 | `theileria_parva` | **NO public data** | **GAP.** No downloadable dataset exists. Depends entirely on Cherehani Labs proprietary slides. **Declared limitation.** |
| 3 | `babesia_spp` | **Proxy only** | Single source: Mendeley Babesia (1,173 imgs, species/host unspecified — likely not cattle *B. bovis/bigemina*). Supplement with Cherehani slides. |
| 4 | `anaplasma_spp` | **NO public data** | **GAP.** No dataset located. Cherehani proprietary slides only. **Declared limitation.** |
| 5 | `eimeria_spp` | **NO public data** | **GAP for labels.** Published Eimeria oocyst studies keep data private. Chula-ParasiteEgg usable only for faecal-float *domain* pretraining (no Eimeria class). Cherehani slides needed. |
| 6 | `haemonchus_contortus` | **NO public data** | **GAP for labels.** Tools (OvaCyte/Parasight) are commercial. Chula *hookworm* eggs are a loose morphological analogue of strongyle-type eggs, **not** H. contortus. Cherehani slides needed. |
| 7 | `plasmodium_falciparum` | **Yes** | Well covered (NIH primary; BBBC041 is *vivax*, genus proxy; Mendeley). Pre-training proxy class — excluded from the veterinary evaluation test set (§2). |

### Blunt summary of the gaps

- **Fully public, usable now:** classes **0** and **7** (both malaria-derived / negative). These
  drive the microscopy **pre-training** stage the proposal specifies (§1.5, §3.2 Phase 1).
- **Public *proxy* only (domain-shifted, use with care):** classes **1** (Trypanosoma) and
  **3** (Babesia). Real images exist but staining/host/specimen differ from the East-African
  cattle target domain; they weakly seed the classes but cannot be the evaluation ground truth.
- **No public data at all:** classes **2** (Theileria parva / ECF), **4** (Anaplasma),
  **5** (Eimeria), **6** (Haemonchus). These are **hard dependencies on Cherehani Labs'
  proprietary slides.** Until those slides are digitised and labelled, the model cannot be
  *evaluated* on these four veterinary target classes. This is the single biggest data risk of
  the dissertation and must be stated as such.

The two priority veterinary diseases with a microscopy component — **East Coast Fever
(Theileria)** and **Trypanosomiasis** — are respectively a **hard gap** and a **proxy-only**
class. Plan the Cherehani slide-collection effort accordingly.

## Resolution choice — 896 × 896, RGB, PNG

MedGemma (4B/27B multimodal) inherits **Gemma 3's SigLIP-400m vision encoder**, whose native
input is **896 × 896**. Normalising every crop to 896 × 896 removes an internal rescale step
and keeps the per-image visual token count stable. Because source crops vary wildly in size
(NIH cells ≈ 130 × 130; BBBC041 fields ≈ 1600 × 1200; Tryp up to 1920 × 1080), each image is
**aspect-preserving letterbox-padded** (black padding) then resized so no aspect distortion is
introduced. Padding is recorded in the summary so it is never mistaken for specimen background.
(Small crops are genuinely upscaled — this is inherent to matching the encoder's fixed grid and
is documented rather than hidden.)

## Leakage prevention — grouped splitting (a real methodological trap)

Cells/crops from the **same physical slide (or patient)** share staining, optics and background.
If crops from one slide appear in **both** train and test, the classifier can memorise
slide-specific artefacts and the test accuracy is **optimistically biased** — a classic and easy
mistake in cell-image ML.

Mitigation implemented in `prepare_microscopy.py`:

- Split is by **group (slide/patient), never by individual image** — an entire group goes to
  exactly one of train/val/test.
- Group keys are derived per dataset: NIH from the patient token in the filename
  (`C100P61ThinF…` → patient `C100`); BBBC041/Tryp from the source field/frame; **Mendeley has
  no slide metadata**, so each species folder is treated as **one opaque group** and lands wholly
  in a single split. That deliberately sacrifices cross-slide class balance (visible in the
  dry-run: Babesia falls entirely in `train`) to **avoid silent leakage** — a documented,
  honest trade-off, not a bug.
- Stratification is per class over groups; splits target 0.70 / 0.15 / 0.15 (§5) by image count.
- Deterministic seed (`20260709`) for reproducibility.

## Usage

```bash
# 1. See what exists and how each is accessed
python scripts/download_microscopy.py --list

# 2. Fetch the anonymously-downloadable sets (NIH, Tryp, BBBC041, Mendeley)
python scripts/download_microscopy.py --dataset nih_malaria_cell
python scripts/download_microscopy.py --all            # gated sets report MANUAL steps

# 3. Plan/validate the split WITHOUT pyspark or downloaded data
python scripts/prepare_microscopy.py --dry-run

# 4. Real ETL on local Spark (needs pyspark + Pillow + downloaded raw data)
python scripts/prepare_microscopy.py --local
```

Gated sources (Kaggle mirror, IEEE DataPort/Chula) are **detected** and print exact,
copy-pasteable instructions rather than failing cryptically. A failed download **never** exits
zero and is **never** silently skipped.

## Licence compliance notes

- **BBBC041 is CC BY-NC-SA 3.0** (non-commercial + share-alike). Fine for an academic
  dissertation, but keep BBBC041-derived crops in a clearly-flagged partition so the NC/SA
  constraint can be honoured if any derived artefact is later redistributed.
- **Chula-ParasiteEgg-11** licence is governed by IEEE DataPort terms and must be confirmed on
  the landing page before any redistribution; access requires registration.
- NIH/NLM, Tryp and Mendeley sets are cleanly compatible with academic research (public-domain /
  CC BY 4.0). All require **citation** — see `dataset_manifest.json`.
