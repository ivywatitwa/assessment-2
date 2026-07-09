# Veterinary RAG Knowledge Base

Grounding corpus for the MedGemma treatment-recommendation task (T2) at Cherehani
Labs. This is the retrieval substrate whose purpose is to **reduce hallucination**
and which underpins the dissertation's hallucination-rate metric (PROJECT_SPEC
§4.3, §6). Unlike the synthetic clinical cases, this corpus is **curated from real,
verified literature** — every chunk carries a citable, fetched source.

## Contents

| File | Description |
|---|---|
| `kb_chunks.jsonl` | 71 chunks conforming to PROJECT_SPEC §4.3, one JSON object per line. |
| `sources.json` | Provenance manifest: 20 sources, APA-7 citations, URLs, access date, verification status, and the `doc_id`s derived from each. Doubles as the dissertation data-provenance appendix. |
| `chroma_db/` | Persistent ChromaDB store (created by `scripts/build_chroma_index.py`; not committed). |

Both `kb_chunks.jsonl` and `sources.json` are produced by
`scripts/build_kb_chunks.py`, which is the reproducible, single-source-of-truth
generator (re-run it to regenerate both artefacts).

## Corpus scope and coverage

All 16 diseases in PROJECT_SPEC §3 are represented. The priority four
(East Coast Fever, Trypanosomiasis, PPR, Bovine Mastitis) are deliberately
over-represented: each has all five core sections **plus** a dedicated
East-Africa epidemiology chunk backed by a Kenyan peer-reviewed study.

| Disease | Species | Sections covered | EA-epi source |
|---|---|---|---|
| East Coast Fever | cattle | aetiology, clinical_signs, diagnosis, treatment, prevention, epi×2 | Yes (Patel 2019 ILRI; CFSPH) |
| Trypanosomiasis | cattle, dog | aetiology, clinical_signs, diagnosis, treatment, prevention, epi | Yes (Okello 2022, Lambwe) |
| Peste des Petits Ruminants | goat, sheep | aetiology, clinical_signs, diagnosis, treatment, prevention, epi | Yes (Kihu 2015, Turkana) |
| Bovine Mastitis | cattle | aetiology, clinical_signs, diagnosis, treatment, prevention, epi | Yes (Mbindyo 2020, Embu/Kajiado) |
| Anaplasmosis | cattle | aetiology, clinical_signs, diagnosis, treatment, prevention | — |
| Babesiosis (bovine) | cattle | aetiology, clinical_signs, diagnosis, treatment, prevention | — |
| Lumpy Skin Disease | cattle | aetiology, clinical_signs, diagnosis, treatment, prevention | partial (in aetiology) |
| Contagious Caprine Pleuropneumonia | goat, sheep | aetiology, clinical_signs, diagnosis, treatment | partial (in aetiology) |
| Coccidiosis | cattle, goat, sheep | aetiology, clinical_signs, diagnosis*, treatment, prevention | — |
| Haemonchosis | goat, sheep | aetiology, diagnosis, treatment, prevention | partial (in aetiology) |
| Canine Trypanosomiasis | dog | aetiology (combined) | — |
| Canine Babesiosis | dog | aetiology, clinical_signs, treatment | partial (in aetiology) |
| Ehrlichiosis | dog | aetiology, clinical_signs, diagnosis, treatment, prevention | — |
| Canine Parvovirus | dog | aetiology, clinical_signs, diagnosis, treatment, prevention | — |
| Feline Upper Respiratory Complex | cat | aetiology, clinical_signs, diagnosis, treatment, prevention | — |
| Feline Haemoplasmosis | cat | aetiology, clinical_signs, diagnosis, treatment | — |

\* Where a source presented two sections together (e.g. Coccidiosis
clinical-signs-and-diagnosis, CCPP diagnosis-and-treatment), the material is kept
in the chunk whose `section` label best matches its dominant content rather than
split artificially.

## Sources and verification methodology

**The single overriding rule of this corpus: no invented citation, DOI, or URL.**
A fabricated source would silently invalidate the hallucination-rate metric, so
each source was located and confirmed before any chunk was written.

Source families used (all open-access):

- **The Merck Veterinary Manual** (15 disease pages) — `type: manual`. The
  professional reference standard; each page was fetched and its aetiology,
  clinical-signs, diagnosis, treatment and prevention content extracted.
- **CFSPH (Iowa State University) factsheet** — `type: guideline`. Used for the
  East Coast Fever geographic-distribution / vector-biology chunk.
- **Peer-reviewed open-access East-African studies** (4) — `type: peer_reviewed`:
  Patel et al. 2019 (*BMC Vet Res*, ILRI, ECF ITM vaccine), Okello et al. 2022
  (*J Parasitol Res*, trypanosomiasis in Lambwe, Kenya), Kihu et al. 2015
  (*BMC Vet Res*, PPR sero-epidemiology in Turkana), and Mbindyo et al. 2020
  (*Vet Med Int*, mastitis in Embu & Kajiado). These provide the Kenya-specific
  epidemiology that generic manuals cannot.

**Verification procedure**
1. Locate each source via web search, preferring FAO/WOAH/ILRI/Merck/PMC/PLOS/BMC.
2. Fetch the page/PDF and read the actual returned text.
3. Write each chunk as a faithful close paraphrase or quotation of that fetched
   text — preserving drug names, doses, vectors, incubation periods and figures.
4. Record the source in `sources.json` with `verification: verified_fetched`
   (full text retrieved and read). No source in this release is
   `verified_metadata_only`; anything that could not be fetched was **excluded**.
5. A subset of agent-fetched Merck URLs were independently re-fetched by the
   curator as a spot check; all resolved and matched.

Every source in `sources.json` lists the exact `doc_id`s derived from it, so any
chunk can be traced back to its origin.

## Chunking strategy (and why)

- **Chunk on section boundaries, not fixed character windows.** Each chunk is one
  clinically coherent unit (a single disease × a single section: aetiology,
  clinical_signs, diagnosis, treatment, prevention, or epidemiology_east_africa).
  This keeps retrieval semantically clean — a query about *treatment* pulls
  treatment text, not a window straddling clinical signs and prevention — and
  makes citations exact: one chunk maps to one source and one topic.
- **Chunk length ≈ 80–140 words (mean ~108).** This is intentionally shorter than
  the spec's 200–500-word *target*, for a concrete embedding reason: the default
  embedding model `all-MiniLM-L6-v2` has a **256-token (~180-word) context** and
  silently truncates anything longer. Sizing each chunk to fit inside that window
  means the entire chunk is embedded — no information is lost to truncation — which
  gives more faithful retrieval than larger chunks that would be clipped. Section
  granularity also matches how a clinician reasons (aetiology vs treatment) and how
  the T2 prompt will request grounding.
- **Species stored for filtering.** Each chunk records its applicable `species`
  list. The retriever can therefore constrain grounding to the patient's species
  — e.g. a canine case is not grounded in cattle-only dosing.

## Embedding-model choice

`all-MiniLM-L6-v2` (sentence-transformers) is the documented default constant in
`scripts/build_chroma_index.py` (`EMBEDDING_MODEL`). Rationale:

- Strong retrieval quality for its size, and the de-facto baseline for RAG.
- Small (384-dim, ~80 MB) and CPU-friendly — important for a Kenya-deployable
  pipeline that may not have a GPU at inference time.
- Cosine similarity (`hnsw:space: cosine`) is configured on the collection.

The constant is centralised so it can be swapped (e.g. for a biomedical embedder
such as PubMedBERT/BioLORD) without touching the ingestion or query logic; the
chunk sizing rationale above should be revisited if a longer-context model is
adopted.

## Gaps and honest limitations

- **East-Africa epidemiology exists as a dedicated, peer-reviewed chunk only for
  the priority four.** The 12 long-tail diseases carry East-Africa context inside
  their aetiology chunks where the source stated it (e.g. LSD's origin in eastern
  & southern Africa; *Rhipicephalus sanguineus* / *B. vogeli* in Africa; *Haemonchus*
  in tropical summer-rainfall regions; CCPP in Africa) but do **not** yet have a
  Kenya-specific quantitative epidemiology chunk. This is a candidate for a future
  curation pass (ILRI / KALRO / FAO sources).
- **Lumpy Skin Disease numeric incubation period is unverified** — neither the
  Merck page nor the (unparseable) CFSPH PDF stated it in days, so it was omitted
  rather than guessed.
- **Canine Babesiosis dosing is generic.** The only available Merck page is the
  multi-species "Babesiosis in Animals" article; its diminazene/imidocarb doses
  are not canine-species-specific and the chunk says so explicitly (small-form
  *B. gibsoni* in particular responds poorly to imidocarb). Treat as directional,
  pending a dog-specific source.
- **Canine Trypanosomiasis** is a single combined chunk (drawn from the shared
  Merck trypanosomiasis page) rather than five sections, reflecting the limited
  dog-specific content on that page.
- **Sources rejected as unverifiable (excluded, not fabricated):** the CFSPH
  Iowa State PDF factsheets for **Lumpy Skin Disease** and **Contagious Caprine
  Pleuropneumonia** were fetched but returned unparseable binary/compressed
  streams, so no chunk relies on them. (The CFSPH *theileriosis* PDF parsed
  successfully via local `pdftotext` and is used for ECF.)

## Regenerating and using the corpus

```bash
# 1. Regenerate kb_chunks.jsonl + sources.json from the curated generator
python scripts/build_kb_chunks.py

# 2. Validate the corpus WITHOUT needing chromadb installed
python scripts/build_chroma_index.py --dry-run

# 3. Build the persistent ChromaDB index (requires: pip install chromadb sentence-transformers)
python scripts/build_chroma_index.py

# 4. Query it
python scripts/query_rag.py "buparvaquone dose for east coast fever" --species cattle -k 3
```

All chunks are pre-validation research grounding material derived from published
veterinary literature; they are not veterinary advice and require expert
validation before any clinical use.
