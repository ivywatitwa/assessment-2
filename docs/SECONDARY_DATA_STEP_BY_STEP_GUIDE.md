# Step-by-step guide — public secondary data with a synthetic auxiliary T2 corpus

**Read first:** [Revised research and assessment plan](SECONDARY_DATA_IMPLEMENTATION_PLAN.md).

**Basis:** the supplied June 2026 thesis proposal, Assessment 2 brief, and prohibition on first-hand collection.

**Status:** an implementation runbook. Future scripts and experiments below are work to perform, not completed functionality or obtained results.

## How to use this guide

Work through the numbered steps in order. Each step identifies actions, outputs and a completion check. Steps 1–4 establish what can be measured; steps 5–12 build the system; steps 13–16 produce research evidence; steps 17–18 turn it into the dissertation.

Use the proposed `data/secondary/` namespace for the revised study so source eligibility and experiment versions are explicit. Existing scripts need configuration changes to consume it. Commands under **Available now** use current interfaces. Commands under **Planned interface** become runnable only after the named implementation work is complete.

## Step 1 — Freeze the revised research scope

**When:** week 1.

1. Adapt the title, aim, objectives and research-design paragraph in the companion plan.
2. Define T1 initially as public Tryp image classification: trypanosome-present versus negative control, with existing boxes for XAI.
3. Define T2 as published-evidence veterinary recommendation generation. Record whether it includes published cases, knowledge questions, or both.
4. Keep the four priority T2 diseases as the starting coverage target: East Coast Fever, trypanosomiasis, PPR and bovine mastitis.
5. Replace the staff trust/usability objective with explanation faithfulness, stability and robustness.
6. Document the secondary-data research route under applicable university requirements. Record the change from the original participant-based design in the methodology.
7. State that new surveys, clinician labelling, synthetic-case sign-off, lab sampling and prospective case collection are absent from the protocol.

**Create:** `docs/research_protocol.md`, including RQs, scope, metric definitions, planned exclusions and a dated change log.

**Done when:** each objective has an experiment and every experiment can be completed with existing public evidence and computational analysis.

## Step 2 — Build the literature and source audit

**When:** weeks 1–3; update the review before submission.

1. Search the university library and appropriate scholarly databases for medical/veterinary multimodal models, parameter-efficient adaptation, RAG, explanation evaluation and microscopy leakage.
2. Record search date, database, exact query, inclusion/exclusion criteria and screening decisions. Example searches:
   - `MedGemma AND (veterinary OR microscopy OR fine-tuning)`
   - `(veterinary OR animal) AND retrieval augmented generation`
   - `(Grad-CAM OR SHAP) AND (faithfulness OR stability OR medical)`
   - `trypanosome microscopy dataset video split`
3. Build a literature matrix: citation, question, data, model, comparison, metric, finding, limitation and relevance to an RQ.
4. Verify the proposal's citations and novelty statements. Correct the ChromaDB/FAISS citation mismatch and Casella date inconsistency identified in the companion plan.
5. Create a separate data-source register. Reading a source for the literature review does not automatically make it eligible for training or redistribution.

**Source-register fields:**

```text
source_id, title, authors, publication_date, version, url_or_doi,
accessed_at, licence_or_terms, allowed_use, source_type,
host_species, specimen, native_labels, existing_annotations,
group_identifiers, source_family_id, intended_role,
download_checksum, eligibility_status, exclusion_reason
```

`intended_role` distinguishes image data, training text, held-out evaluation evidence and retrieval documents. `source_family_id` links versions, mirrors and substantially duplicated publications.

**Create:** `docs/literature_matrix.csv`, `docs/search_log.md`, `data/secondary/sources/source_register.csv`.

**Done when:** the core data sources have attributable origins, usable terms and a clear role, and the research gap follows from the review rather than an assumed absence of prior work.

## Step 3 — Audit the current repository and set up environments

### Current assets and necessary changes

| Asset inspected during planning | How to use it |
|---|---|
| `data/clinical_text/` | Existing documentation lists 1,000 synthetic cases plus 60 synthetic probes. Import them only into the separate synthetic auxiliary T2 corpus; never mix them into public-secondary results. |
| `data/instruction_tuning/` | Existing outputs derive from the old inputs; rebuild the revised dataset from eligible source material. |
| `data/microscopy/local_intake_manifest.jsonl` | Contains 37 intake entries; all have `included_in_t1: false`. Local files exist, but public origin and diagnostic labels are not established. Keep them outside the study inputs. |
| `data/microscopy/prepared/*.dryrun.*` | Planning records can contain `PLAN-ONLY://` paths. They are not observations or processed images. |
| `data/microscopy/dataset_manifest.json` | Useful source shortlist. Correct label mappings and grouping assumptions before reuse. |
| `scripts/prepare_microscopy.py` | Contains Spark image processing. Extend ingestion for native annotations and stronger grouping; export successful processed paths consistently. |
| `scripts/build_instruction_set.py` | Useful chat-record builder. Currently hardcodes old inputs and drops important image grouping/domain metadata; requires revision. |
| `scripts/build_chroma_index.py`, `scripts/query_rag.py` | Existing ChromaDB indexing/retrieval components. Add source-version configuration and use only audited KB content. |
| `scripts/build_rag_prompt.py` | Builds prompts but performs no MedGemma inference; currently loads synthetic cases from hardcoded paths. |
| `data/rag_knowledge_base/` | Existing corpus and provenance scaffolding: 71 chunks and a source register. Recheck source fidelity, permissions and evidence detail before freezing the research KB. |
| `pyproject.toml`, `uv.lock` | Current project dependencies cover ChromaDB and sentence-transformers; Spark, training, evaluation and XAI environments still need to be established. |

### Environment actions

1. Use the current lockfile for the existing local tooling. Record the Python and package versions actually used.
2. Create an ETL dependency group/environment with a mutually compatible Python, Java, PySpark, Pillow and dataframe stack. Do not assume any arbitrary Java/PySpark combination works.
3. Use a separate NVIDIA GPU environment for Transformers, PEFT, TRL, bitsandbytes and Accelerate, based on the official MedGemma fine-tuning example.
4. Add evaluation dependencies such as scikit-learn and `rouge-score`, and XAI dependencies such as `grad-cam` and SHAP.
5. Pin the successfully tested environments. Record OS, CPU, RAM, GPU, VRAM, CUDA and numerical precision. Package minimum versions are not a reproducibility lock.
6. Obtain access to `google/medgemma-4b-it` under its model terms and store access tokens in environment/platform secrets. Pin an exact model and processor revision.
7. Run a small image inference, text inference and gradient probe early. Measure actual GPU availability and memory rather than relying on historical Kaggle quotas or guaranteed 16-GB training claims.

**Available now, from the repository root:**

```bash
uv sync --locked
uv run python scripts/download_microscopy.py --list
uv run python scripts/build_chroma_index.py --dry-run
```

The final command checks the existing KB's structure, not clinical correctness, source fidelity or secondary-data eligibility.

The downloader's current size estimates come from the old manifest. Use the publisher's current metadata for storage planning; for example, the manifest understates the Tryp archive size.

**Create:** environment lock files and `docs/environment.md`.

**Done when:** local validation runs, the GPU environment can load the pinned model and process both input types, and a small backward pass reaches the image pathway.

## Step 4 — Acquire existing public images and prepare their labels

**When:** begin in week 1; finalise eligible scope by week 3.

1. Open the [Tryp paper](https://doi.org/10.1038/s41597-023-02608-y) and its [versioned dataset](https://doi.org/10.6084/m9.figshare.22825787.v1). Record the version and terms.
2. Download the published archive, verify its supplied checksum and record a local SHA-256. Public metadata checked during planning lists a roughly 4.65-GB archive; budget additional extraction space.
3. Preserve the downloaded structure under `data/secondary/raw/microscopy/tryp/`. Parse the existing COCO/YOLO annotations; do not request new annotations.
4. Create one image-manifest row per original image. Derive `trypanosome_present` from existing visible-parasite annotations. Derive `negative_control` from the published non-infected control partition. Log inconsistent or ambiguous rows for exclusion.
5. Store the publisher's organism identity (`Trypanosoma brucei brucei`), host (`mouse`) and preparation (`unstained thick smear`) in metadata.
6. Extract the source video ID from filenames. For example, `positive_video_005_00000073.jpg` maps to `positive_video_005`, not to the frame number or the enclosing `images` folder.
7. Preserve original boxes and original image dimensions. Record whether stronger animal/slide identifiers are available.
8. Scan duplicate hashes and near-duplicate images across folders. Use matching files to merge related groups or exclude duplicates before splitting.

**Image-manifest schema:**

```text
record_id, source_id, source_version, original_image_path, sha256,
native_label, benchmark_label, organism, host_species, specimen,
image_width, image_height, video_id, slide_id, animal_id,
group_id, boxes_xyxy, annotation_source, domain,
eligibility_status, exclusion_reason
```

### Corrections required in the old pipeline

- Its Tryp grouping uses the parent folder name. Replace this with the documented video identifier; retain stronger IDs if released.
- Its folder-based label discovery does not parse Tryp's original `images/labels` layout. Implement an annotation-aware adapter.
- Its image split code assumes one class per group. Replace that assumption with group-aware splitting that supports several labels within an animal/slide group.
- Its proxy filter uses taxonomy class flags. Human negative images may share `uninfected` with animal data, so class filtering alone cannot establish domain eligibility.
- BBBC041 *P. vivax* and genus-only Plasmodium labels must not be mapped to *P. falciparum*.
- Use source-ID/hash-based processed filenames to prevent collisions between datasets with identical basenames.

**Create:** `data/secondary/manifests/images_raw.jsonl`, plus a count/exclusion report.

**Done when:** every eligible image has a defensible source-derived label and group key, and spatial annotations can be plotted in their correct original coordinates without relabelling the data.

**Current run status:** the supplied archive passed MD5 verification. The extracted image partitions contained 3,208 files; 12 were excluded because they were duplicate bytes or positive-partition images without boxes. The parser produced 3,196 eligible records across 95 source-video groups.

## Step 5 — Construct source-traceable text items

**When:** weeks 2–5.

Start with the following reference pages already listed in `data/rag_knowledge_base/sources.json`, rechecking their current content and reuse terms:

| Starting condition | Reference page |
|---|---|
| East Coast Fever | [Theileriosis in animals](https://www.merckvetmanual.com/circulatory-system/blood-parasites/theileriosis-in-animals) |
| Trypanosomiasis | [Trypanosomiasis in animals](https://www.merckvetmanual.com/circulatory-system/blood-parasites/trypanosomiasis-in-animals) |
| PPR | [Peste des petits ruminants](https://www.merckvetmanual.com/generalized-conditions/peste-des-petits-ruminants/peste-des-petits-ruminants) |
| Bovine mastitis | [Mastitis in cattle](https://www.merckvetmanual.com/reproductive-system/mastitis-in-large-animals/mastitis-in-cattle) |

These are seed references, not a sufficient independent train/test corpus. Expand with reusable published cases, WOAH/FAO guidance and relevant peer-reviewed literature. Search PubMed/PMC and the university library using each disease plus `case report`, `treatment` or `culture sensitivity`. Select original evidence by the source-register criteria; do not assume that public reading access grants AI-training or redistribution rights.

1. Search for accessible published veterinary cases and authoritative treatment references covering the four starting diseases.
2. Record which source supports each fact. Use exact page, section, table or paragraph locators.
3. For a **published case**, extract only recorded signs, species and laboratory findings into the input. Hold the reported answer separately. Missing weight, age, location or C&S stays missing.
4. For a **knowledge item**, use a fixed question template about an explicitly stated disease and host, then extract the answer from the source. Do not invent a patient history to make it look clinical.
5. Use the repository's existing synthetic cases as an **auxiliary T2 development/training corpus** while the public corpus is expanded. Run `scripts/import_synthetic_t2_auxiliary.py`; preserve `provenance=synthetic`, `clinical_use=false`, and `requires_expert_validation=true`.
6. Keep the synthetic corpus in `data/secondary/instruction_tuning_synthetic/`, separate from `data/secondary/instruction_tuning/`. In results, label it synthetic pre-validation data, not secondary evidence or clinical data.
7. Define a bounded answer schema: assessment scope, management statements, medication details where explicitly supported, missing information and evidence IDs. Use null for unsupported numerical fields.
8. Extract alternative accepted answers only where the evidence supports them. Preserve formulation, route, species and units. Meat and milk withdrawal values are separate fields and may be unknown.
9. Keep historical reported management distinct from a current-guideline answer. Record which type each item tests.
10. Give all derivatives of a case, document section or question family the same source/group identity. This includes paraphrases and different questions about the same case.
11. Source-check all numerical medication fields and a documented sample of other fields by returning to the publication. This is researcher extraction checking, not clinician validation. Record corrections and unresolved fields.

**Text-item schema:**

```text
item_id, task, item_type, source_id, source_family_id,
case_id_if_published, group_id, species, disease,
input_facts, question, culture_sensitivity_or_null,
reference_answer, reference_claims, required_claims,
accepted_alternatives, source_locators, answerability,
provenance, extraction_method, evidence_status, clinical_use
```

Recommended provenance values for the public corpus are `published_case_extraction` and `literature_derived_item`. Synthetic auxiliary records retain `provenance: synthetic`, `clinical_use: false`, `requires_expert_validation: true`, and `evaluation_role: synthetic_auxiliary_only`. Do not relabel synthetic records as published cases merely by adding a bibliography.

If a diagnosis is supplied in a knowledge question, score recommendation quality rather than diagnostic accuracy. If microscopy results are supplied in a text case, document that input condition; it is not an independent demonstration of image interpretation.

**Create:** `data/secondary/text/items.jsonl`, `data/secondary/text/evidence_claims.jsonl`, and a source-coverage table.

**Auxiliary command:**

```bash
python3 scripts/import_synthetic_t2_auxiliary.py
```

**Done when:** every public answer claim has a source locator, synthetic records are isolated and labelled, all inputs preserve what the source actually supplies, and the available text strata are explicitly named.

## Step 6 — Freeze leakage-resistant splits

**When:** weeks 4–5, before augmentation and fine-tuning.

1. Start with the old repository's 70/15/15 target ratio, but allocate whole groups. Actual counts may differ substantially.
2. For Tryp, split source videos, retaining both classes in train/validation/test where feasible. With only 11 published negative-source videos, report group counts and resulting uncertainty explicitly. If animal/slide IDs connect videos, group at that higher level.
3. Prefer an existing entire-video protocol when it fits the revised experiment. If constructing a new binary split with negative controls, publish it and do not claim direct comparability with the original detection benchmark.
4. For text, group by source document/family and case, including all paraphrases, mirrors and derivatives. Avoid placing different pages of one substantially duplicated guideline into separate groups merely to increase counts.
5. Stratify disease/species where feasible without breaking group boundaries. If there are too few source groups, reduce scope or use explicitly exploratory grouped evaluation rather than claiming strong held-out evidence.
6. Any image/text records from the same actual published case share a split. Unrelated image and text sources remain separate tasks and are never invented into matched patients.
7. Keep training augmentations inside the train split. Retain original evaluation examples and attach perturbations to their parent IDs.
8. Assert no overlap in group IDs, exact content hashes or documented near-duplicate families. Treat uncertain grouping as a recorded limitation, not as proof of independence.
9. Freeze split manifests, seed and checksums. Select all model and explanation settings on training/validation data.

### Separate the KB from held-out answers

Maintain three roles: training material, retrieval reference material and held-out evaluation cases/items.

- Exclude held-out case reports, their answer summaries and near-duplicates from the training set and operational KB.
- It is legitimate for general guidelines needed to answer a held-out case to be in the KB: this is the intended open-book RAG setting.
- If knowledge questions are derived from a guideline that is also in the KB, name that evaluation **open-book source-grounded QA**. It measures finding and using available evidence, not unseen-knowledge generalisation.
- Retain source-group-disjoint evaluation material where feasible and report these strata separately. Record KB overlap for every test item.

**Create:** split manifests, `data/secondary/manifests/kb_membership.jsonl`, and `docs/split_audit.md`.

**Done when:** no prohibited overlap exists, original groups are traceable and the meaning of each held-out benchmark is clear.

## Step 7 — Implement the revised Spark ETL

**When:** weeks 4–5.

1. Add explicit input/output paths and an experiment configuration to the existing preprocessing code.
2. Read image metadata and text items with explicit Spark schemas.
3. Join source eligibility metadata; reject ineligible rows, missing evidence, `PLAN-ONLY://` paths and unverified local sources.
4. Normalise labels, source IDs, units and null values with documented transformations. Preserve original values for auditing.
5. Process image pixels in deterministic partition jobs with identical dependencies on workers. Record resize scale, padding and the image-to-processor transformation. Retain native originals.
6. Transform existing boxes using the same geometry. Keep coordinates needed to return an attribution map to the original field.
7. Write clean, partitioned Parquet metadata and export JSONL records with **actual successful processed image paths**. The current script writes pre-normalisation JSONL and post-normalisation Parquet separately; resolve this mismatch before instruction building.
8. Fail the build or explicitly resolve failed rows; do not silently produce a smaller dataset.
9. Record input/output counts, unique groups, per-class/per-disease counts, exclusions, runtime and partition configuration.
10. Demonstrate idempotence: identical inputs/configuration produce the same IDs, split assignments and content, with no accumulating duplicates.

**Create or extend:** `scripts/etl_secondary.py`, `scripts/prepare_microscopy.py`.

**Planned interface — implement before running:**

```bash
spark-submit scripts/etl_secondary.py --config configs/secondary.yaml
```

**Outputs:** `data/secondary/processed/`, `data/secondary/splits/`, ETL quality report.

**Done when:** real image and text records pass the eligibility, provenance, count and grouping checks and can be loaded directly by the next stage.

## Step 8 — Audit and build the RAG knowledge base

**When:** weeks 6–7.

1. Revisit the existing chunk corpus and source list. Confirm that each chunk's claims are supported by the cited source; a working URL alone is insufficient.
2. Store permitted source snapshots or reproducible references/checksums, extraction method, source version and exact locators. Identify paraphrases as paraphrases.
3. Remove or correct unsupported or cross-species statements. Include regional epidemiology as context, not as a substitute for a treatment source.
4. Chunk at coherent section boundaries. Check token lengths against the chosen embedding model's actual input limit; do not silently truncate key evidence.
5. Add metadata for species, disease, source ID/version, section, date and licence. Preserve units and clinically relevant qualifications.
6. Configure `build_chroma_index.py` for the audited corpus and a versioned persistence path. Record embedding-model revision, distance metric and chunking configuration.
7. Build the index. Use a LangChain retrieval chain connected to the existing ChromaDB/query components to preserve the proposal's architecture.
8. Apply species filtering where input species is known. Do not retrieve by a hidden gold diagnosis. A diagnosis filter is permissible only where the input task explicitly supplies the diagnosis.
9. Review the current hardcoded `section=treatment` filter in `build_rag_prompt.py`. Needed diagnostic, contraindication or C&S evidence may be in other sections; choose the policy on validation data.
10. Tune a small top-k range, such as 3 and 5, against a validation query set with source-derived relevant chunk IDs. Freeze the chosen setting.
11. Handle empty or irrelevant retrieval explicitly in the prompt. Cache IDs, text, scores and filters for each evaluation query.

**Create or extend:** configurable index/query scripts, a LangChain wrapper, `scripts/evaluate_retrieval.py`.

**Outputs:** frozen KB/index, validation retrieval metrics, source coverage and token-length report.

**Done when:** queries return traceable evidence, species filtering behaves as intended and retrieval performance is measured rather than inferred from a few convincing examples.

## Step 9 — Build one mixed image/text instruction dataset

**When:** week 7.

1. Extend `build_instruction_set.py` to take revised image/text split paths and a new output path.
2. Replace the fixed historical taxonomy with the eligible benchmark taxonomy.
3. Preserve `source_id`, `group_id`, domain, host, provenance, split and evidence IDs in every record. The current T1 builder does not preserve all of these.
4. T1 user input contains the image and a neutral classification instruction. The assistant answer is one canonical label. Never include the label in user-visible filenames or metadata rendered into the prompt.
5. T2 user input contains only the published facts/question and optionally retrieved evidence. The assistant answer is the source-derived structured recommendation.
6. Use the same answer schema across RAG and non-RAG conditions. Allow nullable fields and an explicit insufficient-evidence status.
7. If training with and without evidence contexts, derive both versions from training items only, keep them in the same group, record the mixing ratio and use one resulting adapter for C and D.
8. Interleave image and text examples with a stated sampling policy. Start with roughly equal task sampling if data imbalance would otherwise suppress T2; select any changes on validation performance.
9. Keep all held-out probes out of training. Balanced answerability training examples must come from the training material, not the test probes.
10. Inspect image paths, source fields, target leakage, rendered token lengths and both input types through the actual processor.

**Code corrections:** the current `--exclude-proxy` is not a complete domain filter, and `--t2-only` currently changes missing-source checks without unconditionally bypassing available image rows. Make both behaviours match their intended semantics before relying on them.

**Outputs:** `data/secondary/instruction_tuning/{train,val,test}.jsonl` and a task/source/group count summary.

**Done when:** both tasks appear in the intended proportions and every training target traces back to eligible secondary evidence.

## Step 10 — Implement the inference and evaluation harness first

**When:** weeks 6–8, before major training.

1. Implement `scripts/run_inference.py` around the official image-text model/processor API.
2. Support configuration A, B, C or D, a data split, optional adapter path and a fixed output path.
3. Use a consistent deterministic decoding policy for the primary comparison. Record all generation settings.
4. Keep evaluation targets outside the model input. Inference must not read the assistant reference when rendering a prompt.
5. Save raw model output before parsing. Include item/group ID, configuration, model/adapter revisions, prompt hash, retrieved chunk IDs, latency and parse status.
6. Implement strict T1 label parsing with a predefined synonym map; avoid open-ended fuzzy matching that turns incorrect output into a correct label.
7. Implement T2 structured claim parsing and evidence matching. Compare drug/route/species/unit combinations to evidence-backed accepted values. A citation counts as supporting a claim only if the cited passage supports that claim.
8. Score source-supported, contradicted, unsupported and unscorable claims distinctly. Report parsing failures and unscorable-output rates; do not silently drop them from the results.
9. Use the same reference-evidence pool for factual scoring of A–D. Score in-prompt citation correctness separately for B and D.
10. Test evaluators with a small set of hand-specified software fixtures: correct versus wrong units, wrong host, invalid citation, empty answer, unknown label and incomplete output. These are software tests, not fabricated clinical evaluation cases.
11. Run base-model A/B pilot inference on validation items and a T1 majority-class baseline. Use the pilot to fix implementation issues before the held-out evaluation.

**Create:** `scripts/run_inference.py`, `scripts/evaluate_t1.py`, `scripts/evaluate_t2.py`.

**Outputs:** development predictions, metric fixtures and a machine-readable results schema.

**Done when:** the harness correctly detects known mistakes and produces reproducible per-item scores and aggregate counts.

## Step 11 — Fine-tune the pinned multimodal model with QLoRA

**When:** weeks 8–10.

1. Start from the [official MedGemma fine-tuning notebook](https://github.com/Google-Health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb), recording its commit and the environment that actually works.
2. Use the same `google/medgemma-4b-it` revision and processor as the baseline. Keep quantisation/precision consistent between the primary base/adapted inference comparisons.
3. Use the official multimodal collator pattern, adapted to the revised schemas. Verify that images reach the vision encoder and text-only records remain valid.
4. Explicitly verify loss masking: prompt, padding and special image tokens must not become unintended prediction targets; assistant targets must remain trainable.
5. Begin with a conservative QLoRA configuration: 4-bit NF4, batch size 1, gradient accumulation 8, rank 8 or 16, gradient checkpointing and a small validation-driven learning-rate choice. These are pilot settings, not guaranteed optima.
6. Select a supported compute dtype using the actual GPU; P100/T4 hardware should not be assumed to support native BF16. Test the model/quantisation combination on the assigned device.
7. Decide which linear layers receive LoRA. If freezing the vision tower, restrict target modules accordingly and verify trainable parameter names. `all-linear` does not automatically implement that restriction.
8. Set sequence length from measured prompt lengths, including images and retrieved text. Ensure clinically relevant source content and assistant labels are not truncated.
9. Overfit a tiny **training-only** mixed batch as an engineering sanity test. Inspect loss, predicted labels, image-dependent behaviour and gradient flow.
10. Run the planned training experiment, selecting checkpoints on a predeclared validation criterion that accounts for both tasks. Keep per-task validation results visible.
11. Save adapters, trainer/optimizer state, config, seed, data hashes and logs. Test checkpoint resume early. Export checkpoints to persistent platform storage before session termination; input dataset mounts may be read-only.
12. Repeat the final training recipe across three seeds if the measured budget permits. Otherwise state the actual number of independent training runs.

**Create:** `scripts/train_qlora.py` or a reproducible training notebook, plus `configs/secondary.yaml` and an experiment log.

**Outputs:** adapter checkpoint(s), learning curves, validation results, peak memory, elapsed GPU-hours and resume-test record.

**Done when:** a saved adapter reloads, produces both task outputs through the common harness, and is selected without looking at test results.

## Step 12 — Implement Grad-CAM and SHAP with explicit targets

**When:** feasibility probe in weeks 1–3; main work in weeks 11–12.

### 12A. Grad-CAM for the generative image model

1. Use a differentiable score for a **complete canonical class-label sequence**, such as length-normalised conditional log-probability. Use the same definition for both model versions. First-token scores are ambiguous when class labels share a token prefix.
2. Locate the actual SigLIP layer and inspect its activation shape. Derive the spatial grid from the loaded model's patch configuration and tensors; do not copy a generic 14×14 grid.
3. Apply the needed token-to-grid reshape and confirm non-zero gradients. Frozen parameters do not prevent attribution if the relevant input/activation graph is retained; inference-only/no-grad wrappers do.
4. Verify the installed quantisation/PEFT stack supports the required gradients. If XAI needs a dequantised copy, measure prediction/score agreement and report that numerical difference rather than silently presenting it as the identical inference model.
5. Map heatmaps through the recorded resize/padding geometry to the original images. Exclude padding from spatial scoring.
6. Compare the heatmap peak and mass with the **union of existing parasite boxes**. Report correctly classified positives and all annotated positives separately, stating whether the explained target is the predicted or reference class.
7. Measure map mass in boxes as `sum(attribution inside box union) / sum(nonnegative attribution)`. Handle zero maps explicitly. Do not label this quantity IoU.
8. Include random/centre-map or area-based references: large or dense box regions can inflate localisation scores even for uninformative maps.
9. Test high-attribution patch deletion/insertion against matched random patch orders, using fixed perturbation baselines and score definitions.

If Grad-CAM is technically infeasible, record the failure and implement a named alternative such as occlusion attribution. Report the method change and apply the same computational tests; do not rename attention maps or another method as Grad-CAM.

### 12B. SHAP for text recommendation outputs

1. Select a fixed meaningful output span, for example a source-supported management statement. Explain its teacher-forced conditional score, not arbitrary text that changes on every perturbation.
2. For a paired model comparison, use the same reference output span and same input-token grouping. Explanations of each model's own generated answer may be supplementary but have different targets.
3. Attribute clinical/question tokens while holding retrieved context fixed. A separate evidence-attribution experiment can vary the retrieved passages while holding the question fixed.
4. Freeze masking strategy, background, sample budget and seed. Time one example before scaling up.
5. Remove high-attribution token groups and compare score change with length-matched random removal. Report that masking may generate unnatural inputs.
6. Compare attribution stability under a predefined controlled transformation using matched surviving tokens. Do not treat failed alignment as a zero difference.

**Subset plan:** for example, 50 held-out annotated images and 20–40 held-out text items, or all eligible items if fewer. These are compute-budget targets. Select IDs by a fixed stratified rule before examining their model explanations; include error examples in a separately identified analysis.

**Create:** `scripts/explain_images.py`, `scripts/explain_text.py`, `scripts/evaluate_xai.py`.

Save development explanations under validation-specific paths. Generate the fixed held-out explanation set only after settings are frozen in step 13.

**Done when:** both methods have reproducible scalar targets, verified geometry/masking, raw outputs and at least one faithfulness test beyond visual inspection.

## Step 13 — Run the final controlled comparison

**When:** weeks 13–14.

1. Freeze the data, KB, model/adapter revisions, inference settings, parsers, metrics and explanation settings. Record all versions.
2. Run T1 base and adapted models on the same test images. Report majority baseline, accuracy, macro-F1, class support, recall, specificity and confusion matrix.
3. Run all four T2 configurations on identical test items:
   - A: base, no RAG.
   - B: base, RAG.
   - C: adapted, no RAG.
   - D: adapted, RAG.
4. Cache identical retrieved evidence for B and D. A/C receive the same question/facts and answer schema with no retrieved context; avoid contradictory instructions requiring absent citations.
5. Calculate source-supported precision, required-information coverage, unsupported/contradicted rates, ROUGE-L, answerability and parse failures. Add citation metrics for B/D.
6. Report results separately for published cases and knowledge items, with KB-source overlap indicated. Break down by disease/species where counts support interpretation.
7. Compute paired differences for C−A, D−B, B−A and D−C. Confidence intervals should resample groups jointly across configurations, e.g. 1,000–2,000 group-bootstrap replicates.
8. Do not treat a handful of negative videos or source documents as a large sample because they produce many frames/items. Report group counts, instability and undefined metrics transparently.
9. If multiple training seeds ran, show their spread separately from test-sampling confidence intervals. Repeating deterministic inference is not a new training replicate.
10. If a test-time software bug requires a rerun, document it and rerun affected configurations consistently. Test results must not become hyperparameter-search feedback.

**Outputs:** `results/main/` with raw predictions, per-item scores, aggregate tables and confidence intervals.

**Done when:** every main claim traces to a saved prediction table and the comparisons isolate adaptation and retrieval effects.

## Step 14 — Replace staff evaluation with robustness and error analysis

**When:** weeks 15–16.

1. Build a fixed computational probe protocol from held-out public material:
   - Answerable item with relevant evidence.
   - Same item with a key evidence passage removed.
   - Same item with irrelevant retrieved evidence.
   - Existing image with predefined mild blur/compression or illumination change.
2. Label these as derived perturbations and keep their parent group IDs. They are not newly observed clinical cases.
3. Define answerability **per requested field**. Removing a dose reference may make the dose unsupported while leaving a general management answer supportable; do not require blanket refusal of the whole answer.
4. Measure evidence adherence, unsupported specificity and unnecessary abstention. Use the common external evidence ledger for factual support and the supplied context for context-adherence analysis; those are different measures.
5. Run XAI deletion/insertion and stability tests for both model versions on the fixed subsets.
6. Create a reproducible error taxonomy: wrong image label, majority bias, background focus, wrong species, contradicted claim, unsupported numeric detail, invalid citation, missing required information and malformed output.
7. Link each reported error to its public input, raw output and source evidence. Researcher interpretation must be labelled as such rather than presented as an independent clinician judgement.
8. Discuss the results against published explanation/trust research, keeping technical fidelity separate from perceived trust and usability.

**Outputs:** `results/robustness/`, `results/xai/`, `docs/error_analysis.md`.

**Done when:** the former survey period yields quantitative evidence and a documented error analysis without participant input.

## Step 15 — Measure the big data pipeline and compute costs

**When:** record throughout; consolidate by week 16.

1. Measure ETL input/output size, records/s, elapsed time, partitions and peak resource use on the actual dataset.
2. Compare a small number of relevant Spark configurations under the same hardware. Record caching and warm-up conditions; avoid claiming linear scaling from one run.
3. If using replicated manifests to test larger ETL volumes, label them as workload replicas and keep them completely outside model datasets and statistical sample counts.
4. Measure retrieval and inference latency on a fixed query set, separating cold start/model load from warm per-item latency. State whether timings are CPU or GPU.
5. Record embedding/index size, adapter size, training time, XAI time and peak GPU memory.
6. Explain what the measurements imply for feasibility in resource-constrained settings, without claiming deployment performance at the lab.

**Outputs:** `results/performance/`, configuration table, measured resource budget.

**Done when:** the Spark/vector-database work is evidenced by implemented transformations and measured behaviour, not just an architecture diagram.

## Step 16 — Assemble the reproducible research artifact

**When:** consolidate by week 16, then update alongside writing.

**Proposed layout — create as each step is implemented:**

```text
configs/secondary.yaml
data/secondary/
  sources/                    source register and evidence locators
  raw/                        permitted original downloads
  manifests/                  image, source and KB-membership metadata
  text/                       source-derived items and reference claims
  processed/                  clean Parquet / transformed images
  splits/                     frozen group-disjoint manifests
  knowledge_base/              audited chunks and versioned index
  instruction_tuning/         mixed-task JSONL
models/                       adapters and checkpoint metadata
results/
  main/                       A–D and T1 comparisons
  retrieval/                  relevance/latency results
  xai/                        attribution arrays, overlays and metrics
  robustness/                 derived-probe results
  performance/                Spark and compute measurements
docs/                         protocol, audits, literature and reproduction guide
```

1. Record data/model/KB hashes and every run's configuration, seed and code version.
2. Document dependency installation, eligible-data acquisition, preprocessing, training and evaluation in order.
3. Verify loading a saved adapter and reproducing a small result batch from a clean environment.
4. Provide an offline demonstration using public evaluation examples: image classification plus explanation, and a text recommendation plus traceable sources. The dissertation artifact is a research demonstration, not a field trial; no user study is needed.
5. Include only redistributable material in the submission artifact. Supply manifests/acquisition instructions where source terms prevent redistribution. Exclude credentials and large temporary caches.

**Planned command sequence — these interfaces must be implemented first:**

```bash
python scripts/etl_secondary.py --config configs/secondary.json --metadata-only
python scripts/build_chroma_index.py --dry-run
python scripts/evaluate_retrieval.py --input data/secondary/text/retrieval_eval.jsonl -k 5
python scripts/build_secondary_splits.py --input data/secondary/text/items.jsonl --out-dir data/secondary/splits/text --seed 42
python scripts/build_secondary_instructions.py --text-split-dir data/secondary/splits/text --out-dir data/secondary/instruction_tuning
python scripts/merge_mixed_instruction_set.py --public-dir data/secondary/instruction_tuning --synthetic-dir data/secondary/instruction_tuning_synthetic --out-dir data/secondary/instruction_tuning_mixed
python scripts/train_qlora.py --config configs/secondary.json
python scripts/run_inference.py --input data/secondary/instruction_tuning/test.jsonl --output results/test.jsonl --experiment A
python scripts/run_inference.py --input data/secondary/instruction_tuning/test.jsonl --output results/test.jsonl --experiment B
python scripts/run_inference.py --input data/secondary/instruction_tuning/test.jsonl --output results/test.jsonl --experiment C
python scripts/run_inference.py --input data/secondary/instruction_tuning/test.jsonl --output results/test.jsonl --experiment D
python scripts/evaluate_t1.py --input results/t1_predictions.jsonl --out results/t1_metrics.json
python scripts/evaluate_t2.py --input results/t2_predictions.jsonl --out results/t2_metrics.json
python scripts/explain_images.py --config configs/secondary.json --split test
python scripts/explain_text.py --config configs/secondary.json --split test
python scripts/evaluate_xai.py --config configs/secondary.json
```

`build_secondary_instructions.py` is a compatibility wrapper around the extended builder. The current `build_chroma_index.py` is validated with its existing `--dry-run` interface; add a configuration argument only when the KB versioning implementation is ready. Configure inference to evaluate both T1 and T2 for A/C, and T2 for B/D; configure explanation scripts to run both model versions. Save the exact tested commands for the robustness and performance protocols from steps 14–15 in the reproduction guide when implemented.

**Done when:** the artifact connects eligible source data to reported figures and tables without undocumented manual steps.

## Step 17 — Write the dissertation from the evidence

**When:** start early; full draft in weeks 17–20.

1. Use the structure and word budget in the companion plan, staying below 15,000 words excluding references and appendices.
2. **Introduction:** describe the veterinary motivation, revised scope and explicit RQs. State the secondary-data constraint and contribution.
3. **Literature review:** synthesise competing approaches and limitations. Explain why MedGemma, QLoRA, RAG and the chosen XAI tests fit the questions.
4. **Methodology:** describe source eligibility, extraction, grouping, model revisions, controlled A–D design, evidence-based scoring and resource budget. Include deviations from the original proposal.
5. **Findings:** report actual dataset flow/counts first, followed by retrieval, T1, T2, XAI, robustness and performance results. Include uncertainty and failures.
6. **Discussion:** answer every RQ, explain surprising/negative findings, compare with prior work and separate internal validity from transfer to animals/specimens/settings not measured.
7. **Conclusion:** summarise only measured contributions. Position clinical validation and clinician acceptance as future research, not completed components.
8. Cite original dataset papers and software/model documentation appropriately in APA 7. Verify bibliographic metadata rather than copying inaccuracies from the proposal.

**Minimum figures/tables to prepare:**

- Pipeline architecture and data-flow diagram.
- Source eligibility and exclusion flow with counts.
- T1/T2 dataset and independent-group distribution.
- Leakage/KB-overlap design diagram.
- Training and validation curves.
- T1 confusion matrix and baseline comparison.
- T2 A–D results with effect sizes/confidence intervals.
- Retrieval performance and evidence coverage.
- XAI examples accompanied by quantitative localisation/faithfulness results.
- Robustness, error taxonomy and runtime/resource tables.

**Done when:** the findings answer the revised questions and every table/figure has a saved artifact and explanatory discussion.

## Step 18 — Complete the assessment checks and submit

**When:** weeks 21–24; use the actual VLE deadline.

1. Check the revised protocol, abstract, methodology, results and conclusion for consistent secondary-data scope.
2. Ensure no sentence claims clinician trust, lab validation, Kenyan field accuracy, clinical efficacy or real-patient evaluation unless that specific claim is supported by the study's permitted evidence.
3. Review against the six Assessment 2 criteria: structure/presentation 20%, content 40%, critical analysis 10%, language 10%, competency 10%, referencing 10%.
4. Check the required title page, contents and lists of figures/tables; APA 7 in-text citations and references; figure captions and numbering.
5. Check the 15,000-word maximum and module guidance on what counts. Keep the important analysis in the main text.
6. Ensure the work is presented as an individual dissertation and document tools/assistance in accordance with applicable academic rules.
7. Proofread the final exported file and check that tables, equations, links and images remain readable.
8. Submit electronically through Turnitin before the VLE due date, **23:59 UTC**, and retain the receipt.

**Done when:** the submitted dissertation and research artifact consistently support the revised study's conclusions.

## Start here: the first five working days

| Day | Action | Tangible result |
|---|---|---|
| 1 | Write revised protocol; inventory existing data; initialise source and literature registers | Explicit scope and exclusion rules |
| 2 | Inspect Tryp paper/archive metadata; begin published-text sourcing | Evidence-backed T1 choice and T2 source shortlist |
| 3 | Download/audit a public image sample; verify video IDs and existing boxes | Working raw-image manifest sample |
| 4 | Extract 10–20 source-traceable T2 development items and map their evidence | First benchmark records without invented cases |
| 5 | Run model text/image/gradient feasibility probes; assess source-group coverage | Week-1 feasibility report and prioritised engineering backlog |

The immediate milestone is a defensible public-data benchmark and a verified model path. Full training follows only after the sources, groups, prompts and metrics are coherent.
