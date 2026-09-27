# Revised thesis implementation plan — public secondary data plus synthetic auxiliary T2

**Prepared:** 20 September 2026

**Programme:** MSc Big Data Technologies, UEL / UNICAF, UEL-CN-7000

**Companion:** [Step-by-step implementation guide](SECONDARY_DATA_STEP_BY_STEP_GUIDE.md)

## 1. The recommended project

Implement an **offline, quantitative experimental study using existing public data plus a separately labelled synthetic auxiliary T2 corpus**. Preserve the proposal's technical core: Apache Spark, a multimodal MedGemma model, mixed-task QLoRA fine-tuning, ChromaDB/LangChain retrieval, Grad-CAM, SHAP, and controlled evaluation. The synthetic corpus is a development/training aid, not secondary evidence and not clinical validation.

Replace the clinician study with computational explanation-quality and robustness experiments. Replace private laboratory data with already-published images, annotations, case material and veterinary reference documents. Use the repository's synthetic T2 cases only as an explicitly labelled auxiliary development/training corpus while the public text corpus is expanded; never combine synthetic and public results into one clinical-validity claim.

The supplied thesis PDF is the June 2026 proposal and still includes participant research. This document sets out the revisions needed to implement that proposal under the stated prohibition on first-hand collection; it does not imply that a separately approved revised proposal was supplied.

**Document precedence:** for this public-secondary-data study with a synthetic auxiliary T2 corpus, use this plan and its companion guide in preference to conflicting instructions in `IMPLEMENTATION_PLAN.md`, `IMPLEMENTATION_GUIDE.md`, `PROJECT_SPEC.md`, and `METHODOLOGICAL_RISKS.md`. Those older documents include clinician recruitment, local-slide acquisition and different synthetic-data assumptions. The schema changes described here are implementation work still to be completed.

### Suggested revised title

> **Fine-Tuning MedGemma for Explainable Veterinary Decision Support Using Public Microscopy Data and Published Veterinary Knowledge: A Secondary-Data Big Data Pipeline Study**

If the core microscopy experiment uses Tryp, specify its mouse-model provenance in the abstract and methods. Kenya and East Africa remain the motivating application context and a focus of the literature corpus. The title should not imply that the system was tested at Cherehani Labs.

### Revised aim

> To design, implement, and evaluate a reproducible big data pipeline that adapts MedGemma to public animal-parasite microscopy classification and literature-grounded veterinary treatment recommendation, and measures the effects of QLoRA, retrieval augmentation, and explanation methods using secondary data and computational evaluation.

### Revised objectives

1. Establish a documented, reusable collection of public microscopy images and existing annotations, published veterinary case material where available, and authoritative veterinary literature.
2. Implement Apache Spark ingestion, validation, provenance tracking, grouped splitting, and reproducible preparation of image and text instruction datasets.
3. Implement a ChromaDB and LangChain retrieval pipeline over traceable veterinary knowledge.
4. Fine-tune one pinned multimodal MedGemma checkpoint with QLoRA on both image and text tasks.
5. Compare base and fine-tuned models, with and without retrieval for the text task, using fixed held-out data and predefined metrics.
6. Evaluate Grad-CAM and SHAP through existing spatial annotations, perturbation tests, stability tests, and computational cost.
7. Critically assess data limitations, source-domain transfer, reproducibility, and relevance to resource-constrained veterinary settings.

### Research questions and evidence

| Question | Experiment | Main evidence |
|---|---|---|
| RQ1. How does mixed-task QLoRA adaptation affect classification and literature-grounded recommendation performance? | Base versus fine-tuned, with retrieval status held constant | T1 macro-F1; T2 source-supported claim precision and required-information coverage |
| RQ2. How does RAG affect unsupported claims and evidence use? | RAG off versus on, separately for base and fine-tuned models | Unsupported/contradicted claim rates, citation support, answerability and retrieval metrics |
| RQ3. How faithful, stable, and spatially aligned are the explanations? | Grad-CAM and SHAP for both model versions on a predefined subset | Pointing-game accuracy where boxes exist; deletion/insertion tests; attribution stability; runtime |
| RQ4. What are the performance and resource characteristics of the pipeline? | Controlled ETL, retrieval, training, and inference measurements | Records/s, processing time, retrieval latency, peak memory and GPU-hours |

Improvement is a hypothesis, not a promised outcome. Null or negative results can answer these questions when the experiment is sound.

## 2. Exact changes to the original proposal

Page references below refer to the supplied **Thesis Proposal - IVY WATITWA.pdf**.

| Original component | Revised implementation |
|---|---|
| Abstract, p. 2; §§1.1–1.2, p. 4: work at Cherehani Labs | Public-data research prototype motivated by the described workflow. Treat statements about the lab as proposal context rather than independently established study findings. |
| Objective 1, p. 5: lab images, records and validated synthetic cases | Existing public datasets, released annotations and traceable published text, with the existing synthetic cases retained only as a labelled auxiliary T2 development/training corpus. |
| Objective 6, p. 5: staff trust/usability evaluation | Computational explanation faithfulness, stability, localisation and resource evaluation. |
| §1.4, p. 5: Technology Acceptance Model | Use human-centred AI as a design motivation. Discuss TAM in related work if relevant; do not claim to test acceptance, perceived usefulness or ease of use. |
| §1.5, p. 6: VetCompass and staff-validated synthetic cases | Use accessible reusable released material; VetCompass access is not assumed. Existing synthetic records may support auxiliary T2 training/stress tests, but remain pre-validation and are not secondary evidence. |
| §3.1, p. 9: mixed methods with qualitative clinician research | Quantitative computational experimentation and critical synthesis of published literature. Researcher error analysis is analysis of outputs, not participant research. |
| Phase 1, p. 9: new lab images and notes | Download existing public images/labels and extract existing published evidence. |
| Phases 2–4, p. 9: RAG, QLoRA, XAI | Retain; add source separation, explicit ablations and reproducible explanation targets. |
| Phase 5, p. 9: trust survey and thematic analysis | Robustness, failure analysis and explanation-quality experiments using fixed public evidence. |
| §3.3, p. 10: Likert ratings and thematic codes | Remove. Add claim support, citation entailment, abstention/coverage and perturbation metrics. |
| §3.3, p. 10: Grad-CAM localisation | Compute against already-published boxes/masks only. Use faithfulness/stability tests where spatial ground truth is absent. |
| §3.5, p. 10: participant ethics | Record the university's applicable secondary-data research requirements, data licences and provenance. No recruitment or new animal work is part of this protocol. |
| Timetable, p. 11: weeks 15–16 survey | Robustness experiments, attribution analysis, error taxonomy and result consolidation. |

### Copy-ready replacement research-design paragraph

> This study adopts a quantitative experimental design based exclusively on secondary data. Publicly released microscopy images and their existing labels and spatial annotations are used for the image task. Published veterinary case material and source-traceable veterinary reference documents support the text task and retrieval corpus. An Apache Spark pipeline prepares the data, and a fixed multimodal MedGemma model is adapted using QLoRA. Base and fine-tuned configurations are evaluated with and without retrieval augmentation for the text task. Explanation quality is assessed computationally through existing annotations, perturbation-based faithfulness measures and stability tests. No interviews, questionnaires, clinician ratings, new laboratory measurements, new patient records or prospective animal data are collected. The study evaluates technical performance within the available data domains; clinician trust, clinical effectiveness and local deployment outcomes are outside its empirical claims.

## 3. Data policy: what enters this study

### Include

- Already-published image datasets with documented reuse terms and original labels.
- Already-published bounding boxes/masks for localisation evaluation.
- Accessible published veterinary case reports or released text benchmarks with usable evidence and terms.
- Published veterinary manuals, guidelines and research articles for reference extraction and retrieval, subject to their reuse terms.
- Reproducible transformations of those materials: resizing, format conversion, source-based question construction, and controlled perturbations. Preserve links to the original source.
- Newly computed predictions, execution logs and metric results. These are experimental outputs of secondary-data analysis, not new observations from participants or animals.
- The repository's existing synthetic T2 cases may be used as a separately labelled auxiliary training/development corpus. They must retain `provenance:"synthetic"`, `clinical_use:false` and `requires_expert_validation:true`, and their results must never be reported as secondary-data or clinical performance.

### Exclude from the main study

- Interviews, surveys, focus groups, staff usability sessions and clinician scoring of outputs.
- New slides, microscope photographs, clinical observations, laboratory tests and case records collected for this dissertation.
- Requests for clinicians to supply new diagnostic labels, bounding boxes, treatment references or validation judgements.
- Local images without established public secondary-data provenance and reusable labels.
- The repository's programmatically generated clinical cases and referral probes as public secondary evidence. They may be used only in a separate synthetic auxiliary T2 experiment, with `clinical_use:false` and `requires_expert_validation:true` preserved.

Extracting information already present in a publication is secondary analysis. Inventing a patient, a diagnosis, a sensitivity report or a dose and calling it a published case is not. The core plan therefore does not depend on generating synthetic clinical cases.

## 4. Recommended dataset strategy

### T1: a feasible, bounded image benchmark

**The public Tryp dataset is now the implemented T1 source.** The original paper documents 3,085 annotated positive images and 93 negative images, derived from mouse blood-smear videos. The supplied versioned archive was MD5-verified, contained 3,208 image files, and produced 3,196 eligible records after excluding 12 duplicate or unannotated records. The images contain *Trypanosoma brucei brucei*. Existing boxes support image-level label derivation and XAI evaluation without recruiting annotators.

Recommended core task: **trypanosome-present versus negative-control image classification**, using the publisher's existing annotation/sample status and clearly defining the image-label derivation. Preserve the original species identity in metadata. A negative-control field does not establish that an animal is free of every pathogen.

- Repartition by **source video**, or by animal/slide if stronger identifiers are available. The paper explicitly distinguishes frame-level and entire-video splits; its default folder split must not be assumed leakage-safe.
- Keep all frames and derivatives from a video in one split. Original filenames expose the video ID, e.g. `positive_video_005_00000073.jpg`.
- Handle the strong positive/negative imbalance in training only. Report test-class support, specificity and a majority-class baseline.
- Use full fields for primary classification/localisation. A tight ground-truth crop can make localisation artificially easy and removes the original field-level task.
- Treat missing animal/slide IDs as a residual dependence limitation even when video groups are disjoint.

**Secondary candidates, not prerequisites:**

| Source | Potential role | Boundary |
|---|---|---|
| [Tryp paper](https://doi.org/10.1038/s41597-023-02608-y) and [dataset](https://doi.org/10.6084/m9.figshare.22825787.v1) | Core animal-model image benchmark and existing XAI boxes | Mouse, unstained thick smear; not Kenyan livestock field validation |
| [Microscopic Images of Parasites Species v2](https://data.mendeley.com/datasets/38jtn4nzs6/2) | Additional parasite morphology training or a separate, carefully scoped benchmark | Landing page confirms Babesia, Trypanosome and host-cell classes; host/patient/slide provenance is not established there. Avoid random crop splitting and clinical species claims. |
| [NIH malaria data](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html) | Optional, separately identified transfer-learning experiment | Human proxy; both infected and uninfected human images remain human-domain data |
| [BBBC041](https://bbbc.broadinstitute.org/BBBC041) | Optional human-microscopy XAI/method check | *P. vivax*, not *P. falciparum*; human-cell boxes, not animal-parasite localisation evidence |

Tryp's public metadata and the Mendeley and BBBC041 landing pages were checked during planning. The supplied Tryp archive was downloaded and its MD5 `c8aec0e86dda12ade5d122c302e9074d` matched the publisher's Figshare metadata. Only the image and label partitions were extracted; the archive's videos were not extracted. The processed records and exclusions are recorded in the source register and manifests.

The existing repository taxonomy has more veterinary classes than its verified data support. **Do not promise all seven veterinary classes.** Publish a revised taxonomy based on usable labels. If suitable public data for East Coast Fever or other proposed classes are not found, retain those diseases in T2 where reference evidence exists and state the T1 coverage gap. An unsuccessful source search does not prove that no public dataset exists anywhere.

### T2: published-evidence recommendation benchmark

Use two explicitly named strata if data are available:

1. **Published-case stratum:** inputs contain only patient findings actually reported in a case; treatment targets and acceptable alternatives are traceable to its evidence and relevant references. Historical treatment is labelled as reported management, not automatically current best practice.
2. **Literature-grounded knowledge stratum:** questions request treatment/management information for a stated condition and host, drawn from published sources. These are knowledge/recommendation items, not real patient encounters or diagnostic-accuracy cases.

Start with the proposal's four priority conditions: **East Coast Fever, trypanosomiasis, PPR and bovine mastitis**. Expand to other diseases/species only after adequate evidence and independent source groups are available.

Every answer field needs a source locator. Preserve missing values. Include actual culture-and-sensitivity information only where a published source supplies it. If no suitable individual C&S reports are available, document that part of the original workflow as unevaluated; published aggregate resistance percentages are not individual patient reports.

A planning target is **200–400 source-traceable text items**, with group-based train/validation/test splits. This is a workload estimate, not a claim that such a dataset exists or a statistical minimum. Count independent source documents and cases as well as item counts. Many questions from one manual page do not create many independent observations. Use a smaller transparent benchmark if that is the evidence available.

If only a small public knowledge corpus is available, use the synthetic records to exercise T2 training and stress testing, but report the results as **synthetic pre-validation performance**. The public literature retrieval results and synthetic-model results must be presented in separate tables. Neither supports claims about real clinical notes or treatment outcomes.

## 5. Experimental design

### Controlled configurations

| ID | Weights | Retrieval for T2 | Role |
|---|---|---|---|
| A | Base MedGemma | Off | Unadapted baseline |
| B | Base MedGemma | On | RAG contribution without adaptation |
| C | Mixed-task QLoRA adapter + same base | Off | Adaptation contribution without RAG |
| D | Same adapter as C + same base | On | Combined system |

For T1, compare base versus adapted weights; retrieval is not part of image classification. For T2, run A–D on identical items. Hold model revision, precision, output schema, decoding policy and evaluation code fixed. Cache the same retrieved contexts for B and D. Use a common scoring evidence set for A–D; a model without retrieval must not be penalised merely for lacking an in-prompt citation.

Interpret contrasts separately: C−A and D−B estimate adaptation effects; B−A and D−C estimate retrieval effects. A versus D alone cannot disentangle these changes.

The proposal describes two tasks using one multimodal model. Separate image and text records mixed during training satisfy that design. They do **not** constitute paired image-plus-notes clinical cases or prove joint multimodal reasoning on a single patient.

### Metric definitions

| Area | Required measures and interpretation |
|---|---|
| T1 classification | Accuracy, macro-F1, per-class precision/recall, specificity for negative controls, confusion matrix, class counts and majority baseline. Invalid generated labels count as errors. |
| T2 factual consistency | Source-supported claim precision = supported eligible claims / all eligible claims. Also measure required-information coverage so blanket abstention cannot appear optimal. |
| T2 unsupported output | Unsupported or contradicted eligible claims / all eligible claims. Report unsupported and contradicted separately. This is a bounded **source-grounded hallucination proxy**, not proof that every unverified statement is clinically false. |
| T2 text similarity | ROUGE-L against the source-derived reference; supporting metric only. |
| T2 evidence | Valid citation-ID rate and citation support/entailment for B and D. Citation presence alone does not establish support. |
| T2 answerability | Correct abstention on predefined insufficient-evidence items, unnecessary abstention on answerable items, output completeness and unsupported specificity. |
| Retrieval | Recall@k and MRR against source-derived relevant passages, species mismatch rate, empty-retrieval rate and latency. |
| Image explanation | Pointing-game hit rate and attribution mass in the union of existing boxes; deletion/insertion score curves against random masks; stability under controlled transforms. |
| Text explanation | Change in a fixed output-span score after removal of high-attribution input tokens versus random tokens; attribution stability and runtime. |
| Engineering | ETL time/throughput, retrieval/inference latency, peak memory, adapter size and GPU-hours with hardware/configuration recorded. |

Restrict factual scoring to a predefined structured claim schema, with unit normalisation and evidence-backed acceptable alternatives. Track unscorable claims, malformed output and empty answers separately. Zero-claim outputs have no claim-precision denominator; report that as undefined alongside abstention and coverage rather than awarding a perfect score.

Use paired comparisons on the same held-out examples, and confidence intervals that resample independent groups: videos/animals for images and source documents/cases for text. Bootstrapping individual correlated frames or paraphrases exaggerates precision. A target of three training seeds is useful if compute permits; one completed seed must be reported as one, with no estimate of training-seed variability.

## 6. Replacing the clinician phase

Allocate the original weeks 15–16 to these experiments:

1. **Evidence availability:** remove a relevant retrieved passage; check whether unsupported specificity increases and whether the model appropriately limits its answer.
2. **Retrieval quality:** compare relevant context with irrelevant context under a fixed protocol, using existing public passages.
3. **Image robustness:** predefined mild blur/compression and illumination changes; report these as simulated perturbations, not measured field conditions.
4. **XAI faithfulness:** high-attribution deletion versus matched random deletion for images and text.
5. **XAI stability:** compare aligned image maps and matched text-token attributions after controlled changes.
6. **Error analysis:** classify observed failures such as source confusion, wrong host, unsupported numeric detail, majority-class bias and background-sensitive image predictions.

Published studies of clinician trust can inform the discussion. They cannot supply measured trust scores for this prototype. Explanation fidelity, overlap with boxes and low hallucination rates are not direct measures of trust or usability.

## 7. Revised 24-week delivery schedule

These are relative project weeks from the proposal; map them to the actual VLE deadline. Do not assume a new 24-week extension.

| Weeks | Work | Completion evidence |
|---|---|---|
| 1–3 | Revise scope/RQs; literature review; public-data audit; model access; GPU and XAI feasibility probes | Revised protocol, literature matrix, source register, feasible T1/T2 scope |
| 4–5 | Acquire eligible public sources; implement Spark ETL; derive labels; freeze source-group splits | Clean versioned datasets, source counts, leakage report, ETL logs |
| 6–7 | Audit KB; build ChromaDB and LangChain RAG; source-based scoring rubric and inference harness | Frozen KB, retrieval results, working baseline pipeline |
| 8–10 | Mixed-task QLoRA; validation-based selection; checkpoint/resume verification | Saved adapter(s), training curves, resource log |
| 11–12 | Grad-CAM and SHAP on predefined subsets; pilot faithfulness/stability tests | Reproducible explanations and verified scoring targets |
| 13–14 | Freeze settings; held-out A–D evaluation; paired uncertainty estimates | Predictions, main tables, per-task results and confidence intervals |
| 15–16 | Robustness, XAI comparisons, error analysis and pipeline benchmarks | Replacement for survey phase: quantitative robustness/XAI report |
| 17–20 | Complete dissertation and reproducibility package | Full draft and executable artifact instructions |
| 21–24 | Revise argument, audit references/figures, proofread and submit | Final dissertation and submission receipt |

Write the literature review and methods throughout development. Build the evaluator before substantial training so there is a reliable route from checkpoint to dissertation evidence.

### Scope decisions at the end of week 3

- **T1 ready:** public labelled images, group identifiers and usable spatial annotations now exist. The bounded Tryp task has been prepared; add classes only where evidence supports them.
- **T2 ready:** source-traceable questions/answers and enough independent groups for honest evaluation exist. Choose and name the supported strata.
- **Missing spatial labels:** retain computational attribution tests; omit unsupported localisation scores rather than request new labels.
- **Image branch genuinely infeasible:** explicitly narrow the aim/title to the text study and document this as a deviation. A text-only result is not completion of the original two-task objective.
- **Insufficient data for adaptation:** report the limitation and revised achievable design. Do not manufacture cases, class labels or successful outcomes to preserve the original schedule.

## 8. Assessment 2 alignment

The supplied **Assessment 2 - UEL-CN-7000.pdf**, pp. 3–6, requires an individual dissertation with demonstrated research design, literature work, evaluation, reflection and ethical practice. It does not require a survey, private clinical dataset, deployed application or a predetermined model improvement. The no-first-hand-data constraint is supplied by the user and governs this revision.

| Marking criterion, p. 5 | Weight | Evidence to produce |
|---|---:|---|
| Structure / presentation | 20% | Logical chapters, readable pipeline diagram, numbered tables/figures, consistent formatting and linked appendices |
| Content | 40% | Justified problem, critical literature review, explicit RQs, reproducible methodology, implemented pipeline, actual results and analysis |
| Critical analysis and evaluation | 10% | Controlled ablations, uncertainty, leakage analysis, data-domain limitations, failed hypotheses and comparison with literature |
| Language, grammar and punctuation | 10% | Clear academic prose, consistent terminology and proofreading |
| Level of competency | 10% | Explainable technical decisions, coherent presentation of implementation and ability to interpret results |
| Referencing | 10% | Scholarly journals/textbooks plus dataset/tool references; accurate APA 7 citations and source locators |

Spark is justified by the proposal and degree context; the assessment rubric does not independently require a production cluster. Demonstrate partitioned transformations and measure performance honestly. A local Spark run is local parallel processing, not evidence of a multi-node deployment. Replicated records used to benchmark ETL volume must never enter model evaluation as independent data.

### Dissertation structure and working word budget

The maximum is **15,000 words**, excluding references and appendices (assessment p. 6). The following is a suggested budget, not a prescribed chapter allocation:

| Section | Suggested words | What it should establish |
|---|---:|---|
| Abstract | 250 | Actual scope, methods, principal measured findings and limitations |
| Introduction | 1,400 | Motivation, revised aim, RQs, objectives and contribution |
| Literature review | 2,800 | Medical/veterinary adaptation, QLoRA, RAG, XAI, evaluation and defensible research gap |
| Methodology, including implementation | 3,800 | Secondary-data design, provenance, splits, pipeline, models, controls, scoring and compute |
| Findings | 2,200 | Dataset flow/counts, retrieval, A–D comparisons, XAI and performance results |
| Discussion | 2,300 | RQ answers, comparison with prior work, errors, threats to validity and implications |
| Conclusion | 600 | Contribution within measured scope and grounded future work |
| **Total** | **13,350** | Buffer for captions and other material included in local word-count rules |

Include the required title page, table of contents, and lists of tables/figures (assessment pp. 3–4). Follow module guidance for acknowledgements, abbreviations and any additional preliminary pages. Appendices can contain source manifests, schemas, configurations, supplementary results and reproduction instructions; keep essential reasoning in the main body.

### Citation corrections to make during the literature review

- Johnson, Douze and Jégou's *Billion-scale similarity search with GPUs* supports FAISS-related similarity search, not a claim that it documents ChromaDB. Cite ChromaDB documentation for that tool.
- The proposal cites Casella et al. as 2021 in the text but lists a 2023 reference. Verify the publication and align the dates.
- Recheck claims that no prior veterinary multimodal/XAI study exists using a dated, documented search. Avoid absolute novelty claims unsupported by the review.
- Verify that cited papers actually demonstrate the claimed effect, especially clinician-trust claims attributed to interpretability reviews.
- Distinguish open model weights and HAI-DEF model terms from open-source code licences.

## 9. Definition of done

- [ ] Revised title, objectives and methods consistently describe secondary-data research.
- [ ] Every input has documented origin and reuse status; main experiments contain no unverified local intake or generated clinical cases.
- [ ] Public image labels, source text evidence and train/validation/test groups are auditable.
- [ ] Spark ETL and ChromaDB/LangChain retrieval run reproducibly.
- [ ] One pinned multimodal MedGemma model has been evaluated before and after mixed-task QLoRA.
- [ ] T2 A–D results isolate retrieval and adaptation effects.
- [ ] XAI results use existing annotations and explicit computational tests.
- [ ] Raw predictions, error counts, confidence intervals and resource logs support the reported findings.
- [ ] Claims are limited to the actual species, specimens, text strata and settings measured.
- [ ] APA 7 references, figures, word count and individual authorship requirements have been checked.
- [ ] Dissertation submitted through Turnitin by the VLE due date, **23:59 UTC**, with a saved receipt (assessment p. 6).

For the exact engineering sequence, continue with the [step-by-step guide](SECONDARY_DATA_STEP_BY_STEP_GUIDE.md).
