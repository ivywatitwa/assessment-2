# Fine-Tuning MedGemma for Explainable Veterinary Decision Support Using Public Microscopy Data and Published Veterinary Knowledge

## A Provenance-Aware Secondary-Data Pipeline Study

**Student:** Ivy Watitwa

**Programme:** MSc Big Data Technologies

**Module:** UEL-CN-7000 Mental Wealth; Professional Life (Dissertation)

**Institution:** University of East London / UNICAF

**Submission:** Assessment 2 dissertation

**Date:** 26 September 2026

## Declaration of scope

This dissertation is written for a public-secondary-data study with a separately labelled synthetic auxiliary T2 corpus. It does not report new laboratory observations, animal procedures, patient records, participant responses, clinician ratings, staff validation, or a field deployment. Public source metadata, existing published annotations, curated veterinary literature, deterministic software outputs and computationally derived perturbations are the evidence base. The repository's synthetic clinical cases are used only for explicitly labelled T2 development/training and stress testing; they are not secondary evidence or evidence of clinical performance. Unverified local microscopy intake remains excluded.

## Acknowledgements

This dissertation acknowledges the public dataset authors, veterinary researchers and open-source software communities whose published work and tools make reproducible secondary analysis possible. Their sources remain subject to the individual licences and attribution requirements recorded in the project source register. No person or laboratory supplied first-hand data, clinical validation or participant responses for this study.

## Abstract

Multimodal medical models may support veterinary microscopy and evidence retrieval, but high apparent accuracy on a narrow public dataset cannot establish clinical usefulness. This dissertation investigates a provenance-controlled, secondary-data pipeline for MedGemma and MiniCPM-V veterinary decision-support research. A public mouse-model trypanosome microscopy dataset provides the image-classification task (T1); published veterinary literature supplies a small retrieval development set. Synthetic treatment cases (T2) are kept separate from public evidence and used only for auxiliary diagnostics. No clinical samples or participant data were collected.

The Tryp archive yielded 3,196 eligible images after 12 duplicate or unannotated files were excluded. Images were split by source video, not individual frame, into 2,231 training, 488 validation and 477 test records. The initial one-epoch MedGemma 4B QLoRA adapter predicted every validation and test image as trypanosome-present (accuracy 0.9672 and 0.9748; negative-control recall 0). A separate one-epoch, class-balanced MiniCPM-V 4.6 MLX-LoRA run scored 1.000 on its 488-record validation split, then scored 0.9769 accuracy, 0.5711 macro-F1, 0.0833 negative-control recall and 0.5417 balanced accuracy on the 477-record test split. The test contained only 12 negative controls, of which the adapter identified one. The perfect validation result did not carry over to balanced test performance. A local, non-Spark metadata ETL processed 3,200 records in 0.063 seconds with no rejections. A four-query Chroma development check achieved Recall@5 of 1.00 and MRR of 0.8125 after post-hoc filtering. On ten other disease groups spanning nine source pages, a separately implemented SQLite FTS5 retriever achieved Recall@5 and MRR of 1.00; this does not measure held-out Chroma performance. Small synthetic T2 generation, image-occlusion and field-level SHAP runs produced diagnostic artifacts, not validated recommendations or faithful clinical explanations.

The contribution is an auditable implementation, a measured local metadata pass, a limited source-disjoint lexical retrieval check, and a class-sensitive comparison of a collapsed MedGemma run with a separately trained balanced MiniCPM-V model. The study does not establish a controlled causal model comparison, reliable performance across independent groups, Spark execution, performance on Kenyan cattle, treatment safety, clinician trust or deployment readiness.

**Keywords:** MedGemma; veterinary artificial intelligence; secondary data; retrieval-augmented generation; ChromaDB; QLoRA; explainable AI; microscopy; reproducibility.

# 1. Introduction

## 1.1 Background

Animal health is closely connected with food security, household income and public health. In East Africa, diseases affecting cattle, goats, sheep and companion animals can reduce productivity and create direct treatment costs. The proposal that initiated this dissertation focused on East Coast Fever, trypanosomiasis, Peste des Petits Ruminants and mastitis because these conditions are relevant to veterinary practice in the region and because their diagnosis and management require the combination of observations, laboratory evidence and domain knowledge.

Artificial intelligence has been used for diagnostic image classification, clinical prediction and question answering, but a veterinary system must be evaluated in relation to its data domain. A model trained on human microscopy or general web text cannot automatically be treated as a reliable livestock diagnostic tool. This is particularly important when the model generates fluent treatment language: a plausible sentence can contain an unsupported species, dose, route, withdrawal period or diagnosis. Retrieval-augmented generation (RAG) offers a way to expose source material to the model at inference time, but the retrieval corpus, evidence links and evaluation protocol must be made explicit.

MedGemma includes a multimodal 4B model, a text-only 27B model and a multimodal 27B model (Sellergren et al., 2025). The 4B image-capable checkpoint was chosen for this bounded experiment because adapting it was more feasible with the available GPU resources than adapting a 27B model; it was not chosen because every 27B variant lacks vision. Medical pretraining makes it a plausible transfer-learning starting point, but it cannot replace veterinary-domain evaluation. ChromaDB supplies source retrieval; the local metadata preparation in this study is not a measured distributed Spark deployment.

## 1.2 Problem statement

The original project proposed to fine-tune MedGemma on laboratory microscopy and clinical text from Cherehani Labs and to ask clinicians whether explanations improved trust. That design is not implementable under the constraint that first-hand data collection is not allowed. It also contained methodological risks that would affect the validity of the dissertation if they were not revised: public human microscopy proxies could be mistaken for veterinary ground truth, frames from the same video could leak between splits, synthetic treatment cases could be confused with clinical evidence, and clinician trust could be claimed without participant data.

The revised problem is therefore:

> How can a reproducible big data pipeline adapt and evaluate a multimodal medical model for bounded veterinary decision-support tasks using only public, source-traceable secondary data, while measuring the contribution of retrieval and computational explanations without overclaiming clinical validity?

The executed problem is reproducible multimodal data engineering and model evaluation on a *modest-sized* secondary dataset. The Big Data Technologies contribution is therefore framed around variety, veracity, provenance, duplication control and reproducibility rather than an unsupported claim of distributed scale. Distributed scalability is not among the measured findings.

## 1.3 Aim

The aim is to design a reproducible secondary-data pipeline for public animal-parasite microscopy classification and evidence-grounded veterinary recommendation research, implement and evaluate the achievable public T1 and retrieval components, and define bounded tests for adaptation, retrieval and computational explanations without treating planned comparisons as completed findings.

## 1.4 Objectives

1. Establish a documented register of public image datasets, existing annotations and source-traceable veterinary literature.
2. Implement provenance checks that exclude unverified local intake and plan-only paths, while isolating synthetic records in a separately labelled auxiliary T2 corpus.
3. Implement deterministic grouped data preparation and split auditing suitable for Spark and downstream model training.
4. Implement a ChromaDB retrieval pipeline over an audited veterinary knowledge corpus and measure retrieval quality on source-derived development queries.
5. Prepare a mixed-task instruction format for public microscopy classification and literature-grounded treatment recommendation.
6. Evaluate the archived MedGemma T1 adapter against a separately trained, class-balanced MiniCPM-V adapter with class-sensitive metrics, and preserve the model and split boundaries in the comparison.
7. Evaluate Chroma retrieval on source-derived development queries and a separately labelled lexical retriever on source-disjoint queries, without conflating their scores.
8. Run bounded image-occlusion and field-level SHAP diagnostics with explicit targets, then report defined, undefined and unvalidated explanation outcomes separately.
9. Report implementation evidence, blocked experiments, limitations and requirements for a complete future empirical run without fabricating results.

## 1.5 Research questions

**RQ1.** What validation and test performance does a class-balanced MiniCPM-V 4.6 adapter achieve on grouped public trypanosome microscopy classification, and how does it compare descriptively with the earlier MedGemma adapter?

**RQ2.** What retrieval performance can be established for the audited veterinary knowledge base on source-derived development and source-disjoint lexical queries, and what evidence is still required before claiming a RAG answer benefit?

**RQ3.** What do the bounded image-occlusion and field-level SHAP diagnostics establish about attribution signals, and which alignment, faithfulness and stability claims remain untested?

**RQ4.** Which provenance and computational stages were completed reproducibly, and which resource or distributed-processing claims remain untested?

This study answers RQ1 with MiniCPM-V validation and test results and reports the earlier MedGemma validation/test results as a separate model run, not as a controlled head-to-head comparison. It answers a bounded part of RQ2 on four Chroma development items and ten source-disjoint SQLite FTS5 items, and answers RQ3 as a diagnostic-boundary question rather than a claim of explanation quality. It does not resolve performance across independent datasets or seeds, recommendation factuality, comparative RAG effects, explanation faithfulness or complete resource performance across all conditions. RQ4 has measured local metadata processing but no executed Spark stage.

## 1.6 Scope and contribution

The image benchmark is the published Tryp dataset, which contains *Trypanosoma brucei brucei* microscopy images from a mouse model, existing parasite boxes and negative controls (Anzaku et al., 2023). The implemented task is a bounded trypanosome-present versus negative-control image classification problem. Results from this dataset would not establish performance on Kenyan cattle, thin Giemsa smears or East Coast Fever.

The intended public text benchmark is composed of published-case extractions and/or clearly labelled literature-derived questions. No patient facts are invented to create realistic-looking public cases. While that corpus is expanded, the repository's existing 1,000 synthetic cases and 60 synthetic probes are used as a separate auxiliary T2 development/training corpus. The implementation uses four source-derived Chroma development items and ten new disease-group-disjoint literature questions tested with a lexical retriever; the latter were selected without revising the development filter and span nine unique source pages. These are in-corpus passage-identification checks rather than clinical outcome evidence.

The contribution has four parts. First, it converts a participant- and laboratory-dependent proposal into a secondary-data protocol. Second, it provides an auditable source boundary and data contracts for heterogeneous image, annotation, literature and synthetic inputs. Third, it implements grouped public-image evaluation and a retrieval smoke check. Fourth, it documents both a MedGemma majority-class collapse and a balanced MiniCPM-V validation result, demonstrating why model-specific, class-sensitive metrics matter. The pipeline addresses big-data engineering concerns of variety, veracity, provenance, duplicate control and reproducibility; it does not claim that 3,196 images constitute a distributed-scale workload.

# 2. Literature Review

## 2.1 Medical foundation models and MedGemma

Large language models can encode substantial medical knowledge, but knowledge stored in model parameters does not guarantee correct or current use in a particular clinical context. Singhal et al. (2023) showed that language models can encode clinical knowledge, while Tu et al. (2024) described the development of generalist biomedical AI across modalities. These findings motivate domain adaptation but also reinforce the importance of task, population and data provenance.

MedGemma adapts the Gemma family for medical text and image tasks. The technical report describes the 4B multimodal model and both a text-only and a multimodal 27B variant; the multimodal models use a medically tuned SigLIP image encoder (Sellergren et al., 2025). Either multimodal size could, in principle, accept microscopy images. The 4B variant made the experiment computationally feasible, whereas the 27B multimodal variant would require substantially more memory and time; neither has a demonstrated veterinary advantage on these data. For an image-only classifier without generative output, the authors' standalone MedSigLIP encoder is another architectural alternative, not a model evaluated here. These published medical benchmarks do not demonstrate cattle-disease performance.

The transfer from medical pretraining to veterinary microscopy is uncertain. Histopathology may be closer to microscopy than natural images, but domain similarity is not equivalence. Differences in host species, stain, specimen preparation, parasite morphology, magnification and image-capture device can produce distribution shift. The public Tryp data are useful because they are animal-parasite microscopy, but the mouse model and unstained thick smear must remain visible in every result table.

## 2.2 Artificial intelligence in veterinary diagnostics

Veterinary AI spans parasite detection, clinical-record prediction and decision support, but these are distinct evaluation problems. The Tryp authors benchmarked Faster R-CNN, RetinaNet and YOLOv7 for *object detection* on unstained mouse blood-smear frames, reporting test F1 values of 0.71, 0.66 and 0.72, respectively, at an intersection-over-union threshold of 0.5 (Anzaku et al., 2023, Table 4). Their five-fold **whole-video** evaluation of Faster R-CNN instead yielded mean F1 of 0.71 with a standard deviation of 0.07 (Anzaku et al., 2023, Table 5). These are detection scores for locating parasite boxes under the authors' protocols, not image-level binary macro-F1 on this project's new grouped splits: comparing 0.72 directly with this study's 0.49 would be invalid. Their own attention to whole-video partitions and negative-image false detections supports both the grouped design and the explicit reporting of false positives in this dissertation.

Related microscopy research exposes transfer limits. Morais et al. (2022) studied automatic detection of *T. cruzi* in mobile-phone blood-smear images; a different species and acquisition pathway rule out treating that paper as a Tryp or Kenyan livestock baseline. Ward et al. (2022) demonstrated an affordable digital-pathology proof of concept for helminth and *Schistosoma* eggs in stool, a different specimen and target. Torres et al. (2018) compared automated malaria microscopy under field conditions in Peru; it motivates attention to acquisition and deployment context, not a claim about animal trypanosomiasis. For actual animal disease, Desquesnes et al. (2022) review the diagnostic trade-offs among parasitological, serological and molecular methods. Together these works suggest that microscopy classifiers must be judged against their own species, specimen, reference test and grouping protocol rather than an attractive number borrowed from another task. Evaluation under imbalance also calls for class-sensitive reporting rather than accuracy alone (Saito & Rehmsmeier, 2015).

The original proposal identified East Coast Fever, trypanosomiasis, PPR and mastitis as important regional conditions. Okello et al. (2022) found 15/454 cattle (3.30%) positive by buffy-coat microscopy and 71/454 (15.63%) by PCR in Lambwe, Kenya, highlighting how a reference method changes measured prevalence. They reported no *T. brucei* infections in that survey; the mouse-model *T. b. brucei* Tryp proxy therefore should not be confused with the dominant cattle infections in that study. Kihu et al. (2015) studied PPR sero-epidemiology in Turkana, and Mbindyo et al. (2020) studied mastitis in Kenyan dairy cattle. These papers supply context, not patient-level recommendation benchmarks.

Microscopy datasets also vary in task definition. A dataset may contain cropped cells, full fields, object-detection boxes or image-level labels. Converting a box dataset to image classification is possible, but the conversion must be described because it changes the task. Negative controls indicate absence of the labelled target under the publisher's protocol; they do not establish absence of all pathogens.

## 2.3 Retrieval-augmented generation

RAG combines a generative model with a retrieval system that supplies relevant documents at inference time. Lewis et al. (2020) established the basic retrieval-augmented generation formulation for knowledge-intensive tasks, and Gao et al. (2023) reviewed RAG architectures and evaluation issues. In veterinary recommendation generation, RAG is attractive because drug, species, route, antimicrobial resistance and withdrawal information should be connected to identifiable evidence.

RAG does not guarantee factuality. Retrieval can return a passage from the wrong species, a diagnostic section instead of a treatment section, or a historical recommendation that is not appropriate for the input. A model can ignore relevant retrieved evidence or generate additional unsupported details. Consequently, the revised design separates retrieval metrics from generation metrics and reports citation support rather than treating citation presence as proof.

The repository's KB contains 71 section-level chunks from 20 listed sources and covers 16 diseases, five species and six sections. Its development design uses ChromaDB and a sentence-transformer embedding model. The corpus is a grounding substrate, not a held-out clinical dataset. The source register and source locators are therefore as important as the vector index.

## 2.4 Explainable AI for image and text models

Grad-CAM was introduced by Selvaraju et al. (2017) as a gradient-based method for localising image regions relevant to a prediction. Applying the method to a vision transformer requires conversion from a token sequence to a spatial grid. Applying it to a generative model also requires a scalar target: generated class text is not automatically a softmax probability. A defensible implementation must specify whether the score is a class-token logit or a length-normalised conditional sequence score, identify the target layer and map any heatmap back through resize and padding operations.

SHAP provides a framework for attributing a prediction to input features (Lundberg & Lee, 2017). For generated recommendations, token attribution is computationally expensive and sensitive to the output target, token masking, background distribution and prompt length. A statement that displays coloured tokens is not by itself evidence that the explanation is faithful. The proposed full protocol would compare removal of highly attributed features with matched random removal and measure stability under controlled transformations. The completed auxiliary run instead computed field-level values for only four synthetic items.

Hoffman et al. (2018) discuss metrics for explainable AI, and human-centred AI emphasises reliability, safety and appropriate human involvement (Shneiderman, 2020). These works inform the discussion, but they do not supply clinician trust data for this dissertation. Without a participant study, only measured computational properties—not perceived usefulness or acceptance—could be claimed; the present small-sample diagnostics do not yet establish general explanation faithfulness or stability.

## 2.5 Parameter-efficient adaptation and reproducibility

QLoRA is a practical strategy for adapting large quantised models using low-rank trainable adapters (Dettmers et al., 2023). Its feasibility depends on model architecture, processor behaviour, sequence length, image tokens, optimiser state and GPU memory. A successful loss curve does not prove that the model received the image; an overfit-small-batch test and image-dependent prediction check would strengthen evidence of image use.

Reproducibility requires more than a random seed. It includes source versions, hashes, licence terms, grouping rules, model commit, processor commit, environment, hardware, decoding settings, retrieval index and evaluation code. In this project, a grouped split is necessary because adjacent video frames and paraphrases from one document are correlated. Counting them as independent observations would produce misleading confidence intervals.

## 2.6 Research gap and conceptual framework

The original proposal's claim of a wholly unexplored veterinary multimodal space is not supported: Tryp itself contains detection baselines, and related parasite-microscopy systems exist. The narrower contribution here is a transparent *case study* of evaluating a MedGemma adapter and a separately trained, balanced MiniCPM-V adapter on the published Tryp proxy, with source-video splitting, provenance-aware retrieval checks and class-sensitive reporting. The four-condition RAG design and broader explanation study remain specifications; they are not claimed as completed novelty.

The conceptual framework combines three ideas. Human-centred AI motivates transparency and safety. RAG theory motivates source grounding. Big-data engineering provides the data provenance, partitioned processing and reproducibility infrastructure needed to connect heterogeneous sources to controlled experiments. Technology Acceptance Model concepts can be discussed as future user-evaluation theory, but they are not tested here.

# 3. Methodology

## 3.1 Research design

This study uses a quantitative experimental design based exclusively on secondary data and computationally derived transformations. It is not a mixed-methods participant study. The main unit of analysis is a source group: a video or animal/slide group for microscopy, and a published case or source-document family for text. Individual frames and paraphrases remain useful for model training but are not treated as independent evidence in uncertainty estimation.

The research design has three layers. Data engineering covers source registration, eligibility checking, parsing, normalisation and grouped splitting. Model experimentation includes an initial MedGemma QLoRA run, a separate class-balanced MiniCPM-V 4.6 MLX-LoRA run, and the planned RAG conditions. Evaluation covers image classification, Chroma development retrieval, a distinct source-disjoint SQLite FTS5 lexical check and bounded synthetic/XAI diagnostics. The planned Spark extension follows Zaharia et al. (2016), but the measured preprocessing stage here is local, non-Spark metadata ETL.

:::figure Figure 1. Implemented pipeline and evidence boundaries (author's synthesis).
:::image figures/figure1.svg
:::endfigure

Figure 1 distinguishes the executed public image-model and retrieval checks from the proposed, but unexecuted, RAG answer comparison. Processing heterogeneous provenance and model inputs is a data-engineering problem here; 3,196 images do not by themselves establish large-scale distributed processing.

## 3.2 Secondary-data policy

Eligible inputs are public datasets with documented terms, existing labels and existing spatial annotations; published veterinary cases or released text benchmarks; and published veterinary manuals, guidelines and research articles used for retrieval or evidence extraction. Allowed transformations include format conversion, resizing, source-based question construction and controlled perturbations. Each transformation retains a link to its parent source.

Excluded inputs are surveys, interviews, clinician ratings, new diagnostic labels, new bounding boxes, laboratory measurements, patient records collected for this project, unverified local images, and programmatically generated clinical cases used as if they were observations. The repository's synthetic records are retained as a separate auxiliary T2 development/training corpus with `clinical_use=false` and `requires_expert_validation=true`; they are not secondary evidence and are not combined with public-data results as clinical validity evidence.

## 3.3 Data sources

The public Tryp dataset is the primary image source. Its paper reports 3,085 annotated positive images and 93 negative images and documents source videos and bounding-box annotations (Anzaku et al., 2023). The versioned archive supplied for this implementation contained 3,208 image files. The annotation-aware parser excluded four positive-partition images with no boxes and eight duplicate-byte images, producing 3,196 eligible records: 3,103 trypanosome-positive records and 93 negative controls. The operational archive count is reported separately from the paper's summary rather than silently forcing the archive to match the publication. The parser reads the publisher's YOLO labels and derives a binary image-level label. It stores the native organism and mouse host rather than mapping the data to cattle.

The Mendeley parasite dataset is an optional morphology or transfer source. Its public landing page lists Babesia, Trypanosome, Plasmodium and host-cell classes and states a CC BY 4.0 licence (Li & Zhang, 2020). It does not establish patient or slide-level grouping sufficiently for a naïve random split, so it is not required for the core experiment. NIH malaria and BBBC041 are human microscopy proxies. BBBC041 is specifically *P. vivax* data under CC BY-NC-SA 3.0 (Ljosa et al., 2012); it must not be relabelled as *P. falciparum* or veterinary data.

For text retrieval, the repository's 71 chunks were used in the implementation smoke test. The register records their disease, species, section, citation, URL and source type. The four priority diseases are represented in the corpus. The corpus includes Merck Veterinary Manual material and East African peer-reviewed sources such as Patel et al. (2019), Okello et al. (2022), Kihu et al. (2015) and Mbindyo et al. (2020). Source-specific reuse terms must be checked before redistribution or release.

## 3.4 Data contract and provenance

The revised image manifest requires a record ID, source ID and version, original path and hash, native and benchmark labels, organism, host, specimen, source-video group, box coordinates, annotation source, provenance and eligibility status. The revised text item requires an item ID, source and source-family IDs, group ID, species, disease where explicitly stated, question or published facts, reference answer, structured claims, source locators, answerability, provenance and evidence status.

The `source_register.json` file contains seven entries: four public dataset sources, the curated veterinary KB, unverified local intake and synthetic generated cases. The local intake is excluded, while the synthetic source is permitted only as a labelled auxiliary corpus. The validation script rejects missing identifiers, invalid URLs, unapproved synthetic use, local unverified eligible sources and inconsistent statuses.

## 3.5 Metadata ETL and grouped splitting

The implemented ETL validates manifests and source eligibility with a dependency-free metadata processor, writing clean JSONL and a rejection report. A full-manifest local run processed 3,196 image metadata rows and four source-derived text rows: 3,200 input, 3,200 clean, zero rejected, 0.063 seconds elapsed on this machine (metadata only; no image decoding or pixel normalization in the timed stage). A separate PySpark eligibility-ETL runner is supplied but was not executed because PySpark could not be installed in this environment. Image bytes are not stored in metadata rows; a separate deterministic stage writes 896 by 896 RGB PNGs and transforms publisher boxes using the same resize and padding geometry. Neither Spark execution nor a throughput advantage is claimed.

The programme fit is based on the data-engineering problem rather than dataset size alone. **Variety** is represented by image bytes, YOLO annotations, manifests, veterinary text, retrieval metadata and synthetic records with different evidence statuses. **Veracity** is addressed through source licences, hashes, annotation origin, eligibility checks and explicit rejection reports. **Volume and duplication** are controlled by distinguishing 3,208 archive files from 3,196 eligible records and 95 source groups, rather than treating every frame as an independent animal. **Reproducibility** is addressed through deterministic normalisation, grouped splitting, pinned model revision and machine-readable outputs. These are implemented big-data pipeline controls; they are not evidence of cluster-scale throughput or linear scaling.

The current secondary split utility is dependency-free and deterministic. It requires a group ID and a label field, sorts groups using a stable hash and assigns groups toward 70% train, 15% validation and 15% test targets. The completed Tryp split contains 2,231 training images from 67 groups, 488 validation images from seven groups and 477 test images from 21 groups. No source-video group occurs in more than one split (Figure 2). The separate four-item retrieval smoke set remains development-only.

:::figure Figure 2. Public Tryp image flow and group-separated split (author's analysis).
:::image figures/figure2.svg
:::endfigure

## 3.6 RAG implementation

The veterinary KB is indexed in ChromaDB using a sentence-transformer embedding function. Query filters include disease, species and section. The development evaluator records query text, filters and returned document IDs and measures Recall@k and mean reciprocal rank against source-derived relevant IDs. For a further, source-disjoint passage-identification check, the same 71 chunks were indexed in SQLite FTS5 (Porter tokenization, BM25 title weight 2 and body weight 1); ten prewritten questions from ten families excluded from the four development families were evaluated. The FTS5 scores cannot be attributed to Chroma, and labelled passages remain inside the indexed corpus. Passing retrieved context to the model and checking faithful citation belong to the unexecuted generation comparison.

The production T2 experiment will separate four conditions. Conditions B and D receive the same cached retrieved contexts. Conditions A and C do not receive retrieval. Factual scoring for all four conditions uses a common external evidence ledger, while citation correctness is reported separately for the retrieval conditions. This prevents the no-RAG baseline from being penalised solely for not having an in-prompt citation.

## 3.7 Model conditions and training

The initial T1 run used multimodal `google/medgemma-4b-it` at revision `290cda5eeccbee130f987c4ad74a59ae6f196408`. Its archived training log records one epoch, 2,231 public T1 training records, 279 optimiser updates, gradient accumulation of eight, learning rate `2e-4` and FP32 computation. The adapter was evaluated on validation and test, but predicted the positive class for every record. A separate follow-up used `openbmb/MiniCPM-V-4.6` at revision `36f34a661a4bd35d0dc2294cb044d2584646c7d3` with MLX-VLM LoRA, 4x visual token compression, one epoch and training-only class balancing. The effective training set contained 4,332 records (2,166 per class, with minority records cycled from the existing training split), at learning rate `2e-4`, rank 8 and 1,083 optimiser updates. MiniCPM was evaluated on validation and, subsequently, on the unchanged test split using the same adapter. These distinct architectures and training pipelines do not support a controlled causal comparison, and no base-model comparator or alternative seed was run.

The four T2 configurations are:

| Configuration | Weights | Retrieval |
|---|---|---|
| A | Base | Off |
| B | Base | On |
| C | QLoRA adapter | Off |
| D | Same QLoRA adapter | On |

The planned image comparison is base versus adapted model, but no base-model result was archived. The MedGemma and MiniCPM adapters were evaluated on the grouped validation and test splits; neither model was trained on validation or test records. No seed-variability estimate is available, and no T2-specific adapter is included in the reported findings.

The completed mixed training set combines public microscopy records with the synthetic auxiliary T2 training records. It contains 2,231 public T1 records and 700 synthetic T2 records in training, 488 public T1 records and 151 synthetic T2 records in validation, and 477 public T1 records and 149 synthetic T2 records in test. The mixed set preserves `evaluation_role=synthetic_auxiliary_only` so T2 synthetic results can be separated from public retrieval and image results. The synthetic validation/test partitions are reported as pre-validation synthetic performance, never as secondary-data or clinical performance.

## 3.8 Evaluation measures

The completed T1 evaluations report accuracy, macro-F1, class support, confusion matrices, invalid generated labels, per-class recall and balanced accuracy. Group-based confidence intervals were specified but not computed, and no uncertainty interval is asserted for these findings.

For T2, the planned primary measure is source-supported claim precision: supported eligible claims divided by all eligible claims. Unsupported and contradicted claims are reported separately. Required-information coverage prevents blanket refusal from appearing optimal. ROUGE-L is included as a surface-overlap measure, not as a clinical correctness measure. Valid citation rate and citation support are separate from factual consistency.

For retrieval, Recall@k and MRR are computed against source-derived relevant passages. A future independent experiment would also measure species mismatch, empty retrieval and latency. The small completed image-occlusion diagnostic measures attribution mass inside existing boxes, while the synthetic T2 diagnostic scores ten masked fields; deletion/insertion curves, stability and token-attribution faithfulness remain proposed evaluations rather than findings.

## 3.9 Reproducibility and audit controls

The audit contract calls for source versions and hashes, split seed and group IDs, model and processor revisions, environment and hardware, retrieval index, prompts, decoding settings and raw outputs. The saved material establishes source grouping, pinned MedGemma and MiniCPM revisions, training metadata and per-image predictions, but does not supply a complete hardware, latency or prompt-hash inventory. Source counts are distinct from record counts: many frames from one video are not independent animals.

The implemented stages validate provenance, convert published labels, normalise images, assign groups, prepare instructions and evaluate retrieved passages without silently mixing unverified or synthetic records into the public-image benchmark. Appendix A maps these stages to their scripts and saved artifacts. The general-purpose inference interface is not the source of the archived T1 predictions and must not be read as evidence that the full four-condition comparison ran.

## 3.10 Ethics and responsible use

No participant recruitment or animal work is part of the revised protocol. Data licensing and source attribution remain ethical requirements. The model must be described as research-only decision support and not as veterinary advice or a deployed diagnostic. The HAI-DEF model terms, source-specific data licences and any restrictions on redistribution must be recorded before release.

The absence of a clinician survey is a limitation, not an ethical shortcut for making trust claims. The dissertation can report computational explanation properties and published literature about human-centred AI, but cannot infer clinician acceptance or usability.

# 4. Implementation

## 4.1 Repository audit

The source audit separated three kinds of material already in the project: published microscopy and veterinary knowledge, generated synthetic treatment cases, and local images without verified annotations. Only the first category could support public-data model evaluation; the others required either auxiliary status or exclusion.

The audit found 37 local-intake manifest records. They were classified as duplicate-removed, blood-smear records needing expert validation, faecal/debris records needing validation, bacterial microscopy, culture plates or unknown microscopy. No record was marked `included_in_t1=true`. The revised validator therefore reports the local-intake boundary as passing while retaining the records outside the study.

The original instruction format had no source-family boundary. The revised construction carries group and provenance fields into each training record, while preserving explicit synthetic flags. This made it possible to train on public T1 data alone despite the presence of synthetic T2 records in the broader repository.

## 4.2 Source register implementation

The new register records the Tryp, Mendeley, NIH and BBBC041 public datasets; the curated veterinary KB; the excluded local intake; and the synthetic auxiliary corpus. The validator checks the `secondary_data_only_with_synthetic_auxiliary` policy, required fields, duplicate source IDs, URL structure, role/status consistency and prohibited source types. It also scans the existing local-intake manifest for accidental inclusion.

The register records the Tryp archive MD5 checksum, processed record count, excluded-record count and source-video group count. This distinguishes the downloaded and processed public source from the paper-only candidates. The register also marks the KB as eligible for local research indexing pending source-term review, rather than presenting the entire corpus as universally redistributable.

## 4.3 Public image parser

The Tryp parser searched the supplied archive for images, derived source-video IDs from publisher filenames and read the publisher's YOLO boxes. It recorded hashes, binary benchmark labels, organism, host, specimen and normalised annotations, while excluding duplicate bytes and positive-partition images without boxes. It did not invent animal, breed or location metadata.

This is an important correction to the previous microscopy preparation code, which used the parent folder as a Tryp group and assumed folder-based class labels. The new contract keeps all frames from a source video together and preserves the native annotation origin.

## 4.4 Source-derived retrieval benchmark

Four development queries targeted existing veterinary KB passages on East Coast Fever treatment, trypanosomiasis diagnosis and treatment, PPR treatment and control, and bovine mastitis culture-and-sensitivity treatment. Each query had species, disease, relevant passage IDs and source locators. They were constructed from the literature corpus and used for development; they were neither unseen queries nor synthetic animal cases.

The retrieval evaluator applies disease, species and optional section filters before calculating Recall@5 and reciprocal rank. In the corrected East Coast Fever query, the designated passage ranks first. This tests query filtering on a small development set; the saved IDs and scores are mapped to artifacts in Appendix A.

In operational terms, the retrieval check takes a question and its species/disease metadata, applies eligible-source filters, ranks matching KB passages, and compares the returned document IDs with the predeclared relevant IDs. The returned passage includes a citation and source locator; a *future* RAG model run would place that evidence beside the user question before generation. This dissertation measured the retrieval step, not whether MedGemma used the passage faithfully in a treatment answer.

## 4.5 Evaluation interfaces

The T1 evaluator requires `record_id`, `group_id`, `gold_label` and `predicted_label`. It reports accuracy, macro-F1, class counts, confusion matrix and a classification report. The T2 evaluator defines structured claims with explicit `supported` and optional `contradicted` flags; no held-out source-grounded T2 claim-scoring result is reported. These interfaces prevent the completed T1 analysis from relying on unstructured visual judgement of model prose.

Separate guarded interfaces validate experiment configurations without manufacturing predictions when model prerequisites are absent. The empirical T1 claims here come from the archived adapter evaluations, not from a configuration dry run.

The bounded explanation runners evaluate the *saved T1 adapter*, not a new explainable model. For T1, the script measures the difference between the fixed positive-class and negative-class text scores, masks each of 16 image tiles in turn, and records the change in that margin. For synthetic T2, KernelSHAP masks combinations of ten named case fields and attributes changes in the score of a fixed reference diagnosis line. The image output is a coarse perturbation score, not Grad-CAM; the text output is field-level attribution, not an explanation of generated treatment. Sections 5.5–5.6 give the actual inputs and outputs so the distinction between a mechanism and a validated explanation is visible.

# 5. Findings

## 5.1 Findings boundary

The completed results comprise provenance and dataset audits, a full local metadata ETL pass, a four-query Chroma retrieval development run, a ten-query source-disjoint lexical retrieval check, grouped public T1 evaluations of MedGemma and MiniCPM-V on validation and test, and small synthetic T2/XAI diagnostic runs. The MiniCPM validation result is reported alongside its weaker test result and small negative-class support. The synthetic and explanation artifacts are reported below at their actual sample sizes and are not substituted for clinical evidence.

## 5.2 Source and exclusion audit

| Audit item | Verified result | Interpretation |
|---|---:|---|
| Source-register entries | 7 | Four public datasets, one curated KB, one excluded local source and one synthetic auxiliary source |
| Local-intake manifest records | 37 | Existing repository intake; not study data |
| Local-intake records included in T1 | 0 | Passes the revised exclusion rule |
| Synthetic T2 cases/probes | 1,000 main cases + 60 probes | Auxiliary development/training only; not secondary or clinical evidence |
| Tryp archive image files inspected | 3,208 | MD5 verified; archive version used for this implementation |
| Duplicate/unannotated Tryp records excluded | 12 | Eight duplicate-byte records and four positive-partition records without boxes |
| Eligible processed T1 images | 3,196 | 95 source-video groups; no group leakage across splits |
| Source-derived T2 development items | 4 | Literature-grounded smoke items; insufficient for final model evaluation |
| MedGemma access/checkpoint | Downloaded and adapted | Revision `290cda5eeccbee130f987c4ad74a59ae6f196408`; T1 validation and test outputs archived |
| MiniCPM-V 4.6 follow-up | Trained and validation/test-evaluated | Revision `36f34a661a4bd35d0dc2294cb044d2584646c7d3`; balanced MLX-LoRA adapter; 488 validation and 477 test predictions |
| Full local metadata ETL | 3,200 input; 3,200 accepted; 0 rejected | 0.063 s, metadata-only Python pass; not a Spark timing |
| Source-disjoint lexical queries | 10 disease/source groups | FTS5 retrieval only; Chroma held-out run not measured |

The source audit passed. This is a meaningful result because the revised protocol distinguishes public secondary evidence, synthetic auxiliary data and local unverified intake instead of allowing them to flow into one undifferentiated instruction set. The audit also demonstrates that the revised protocol can be applied without deleting or relabelling the original records.

## 5.3 Knowledge-base validation

The existing knowledge-base dry run validated 71 chunks. The validated corpus covers 16 diseases, five species (cat, cattle, dog, goat and sheep) and six sections (aetiology, clinical signs, diagnosis, East African epidemiology, prevention and treatment). ChromaDB indexing and query infrastructure were present locally, and a cached sentence-transformer embedding model was available for the smoke retrieval run.

The dry-run validation verifies schema and source metadata. It does not independently re-read all source pages during this run and does not establish that every chunk is clinically current. The KB is consequently treated here as curated research grounding material rather than authoritative treatment advice.

## 5.4 Retrieval smoke result

| Item | Query topic | Relevant documents | Retrieved relevant at k=5 | Reciprocal rank |
|---|---|---|---:|---:|
| RET-001 | East Coast Fever treatment, treatment-section filter | 1 | 1.00 | 1.000 |
| RET-002 | Trypanosomiasis diagnosis/treatment | 2 | 1.00 | 0.250 |
| RET-003 | PPR treatment/control | 2 | 1.00 | 1.000 |
| RET-004 | Bovine mastitis C&S treatment | 2 | 1.00 | 1.000 |
| **Mean** | **Four development items; corrected treatment-aware retrieval** | 7 relevant references | **1.000** | **0.8125** |

Filtering by disease and species alone missed the target treatment passage for the East Coast Fever query. Adding `section=treatment` fixed that development query, but this is a change made *after inspecting the miss*. Its four-query increase is therefore a debugging result, not an independent estimate of generalisation. Unseen questions and source families are needed to test whether that filter helps systematically rather than overfitting the example.

**Worked retrieval example (RET-001).** The stored query asks, “What is the evidence-grounded treatment context for East Coast Fever in cattle?” Its expected document is `vet-kb-ecf-004`, a source-labelled treatment chunk from the curated veterinary knowledge base. In the documented initial development run, filtering by disease and species alone failed to return this chunk among the top five. In the saved corrected run, the additional `section=treatment` filter returned `vet-kb-ecf-004` at rank 1 (reciprocal rank 1.0). Across the four development queries, Recall@5 changed from 0.75 to 1.00 and MRR from 0.5625 to 0.8125. This is an improvement to *retrieval selection*, not a before-and-after comparison of generated answers: there is no saved, paired RAG-versus-no-RAG model output for this question. The earlier top-five document IDs are not preserved in the corrected report, so they are not reconstructed here.

The Chroma smoke result has four items, all from the same curated corpus and all used for development. It must not be reported as a statistically generalisable retrieval benchmark.

**Additional development-disjoint retrieval check.** Before running the new lexical evaluator, ten questions were saved in `data/secondary/text/retrieval_heldout.jsonl`, one each for anaplasmosis, bovine babesiosis, lumpy skin disease, contagious caprine pleuropneumonia, coccidiosis, haemonchosis, canine babesiosis, canine ehrlichiosis, canine parvovirus and feline upper respiratory complex. No disease/source-page group overlaps RET-001–RET-004, but canine and bovine babesiosis share a reference page, leaving nine unique source pages across the ten questions. The reference passage ID for each question was read from an existing KB chunk, and the unchanged 71-chunk corpus was indexed with SQLite FTS5 (Porter/BM25, title weight 2, body weight 1). With the predeclared disease/species filters and one prevention-section filter, the lexical retriever returned all ten reference IDs at rank one: Recall@5 = 10/10 = 1.00, MRR = 1.00. An exploratory re-evaluation dropping only the disease filter also returned 10/10 at rank one. These numbers describe the lexical retriever **only**, not Chroma or model-generated answers. Named diseases and near-verbatim source terms make these passage-identification questions relatively easy, and the relevance labels were selected from the indexed KB itself; disease-group separation is an improvement over four development queries, not independent clinical or out-of-corpus validation. The individual returned IDs and ranks are saved in `data/secondary/reports/retrieval_heldout_lexical.json`. The held-out Chroma result remains unmeasured.

## 5.5 Public T1 model result

The split contains 3,103 positive and 93 negative-control images: training has 2,166 positive and 65 negative images; validation has 472 and 16; test has 465 and 12. The 95 source-video groups do not cross split boundaries. The initial one-epoch MedGemma adapter was trained on 2,231 public T1 records and evaluated on the 488-record validation and 477-record test splits. Its saved evaluation artifacts use the same adapter payload. A separate one-epoch MiniCPM-V 4.6 adapter used deterministic training-only balancing: the 65 negative training examples were cycled to form 2,166 effective examples per class (4,332 training rows total). The same adapter was evaluated on validation and then on test. The validation split contains only 16 negative examples, all from one source-video group; the test split contains 12 negatives. Metrics below were recomputed from the saved per-image prediction files and recorded in `data/secondary/reports/minicpm_t1_metrics.json`; group-based confidence intervals are not available.

| Model / split | Records | Groups | Accuracy | Macro-F1 | Negative recall | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| MedGemma / validation | 488 | 7 | 0.9672 | 0.4917 | 0.0000 | 0.5000 |
| MedGemma / test | 477 | 21 | 0.9748 | 0.4936 | 0.0000 | 0.5000 |
| MiniCPM-V 4.6 / validation | 488 | 7 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| MiniCPM-V 4.6 / test | 477 | 21 | 0.9769 | 0.5711 | 0.0833 | 0.5417 |

:::figure Figure 3. MedGemma held-out test confusion matrix (477 images; saved predictions).
:::image figures/figure3.svg
:::endfigure

Neither model produced invalid labels in the reported evaluations. Rows in the confusion matrices are true labels and columns are predictions, ordered `[negative_control, trypanosome_present]`. For MedGemma, validation is `[[0, 16], [0, 472]]` and test is `[[0, 12], [0, 465]]` (Figure 3); it called every record positive. For MiniCPM validation, the matrix is `[[16, 0], [0, 472]]`: all 16 negative controls and all 472 positive images were classified correctly. Its test matrix is `[[1, 11], [0, 465]]`: it identified one of 12 negative controls and all 465 positives. Validation accuracy is 488/488 = 1.000, while test macro-F1 is 0.5711 and balanced accuracy is 0.5417. The validation negatives come from one source-video group; test has only 12 negative records. The perfect validation result therefore does not establish balanced performance on the test split.

**Worked image-output example.** For the earlier MedGemma adapter, validation record `tryp-d65a5bf4a6599e2a` (group `negative_video_008`) has truth `negative_control`, but its raw output is `trypanosome_present`; positive record `tryp-6b06d097f90b7576` receives that same output. In the separate MiniCPM validation file, both records receive their correct labels. The contrast illustrates the observed outcomes of two different models; it is not a controlled before-and-after retraining comparison, because architecture and training pipeline differ and no common base-model comparator was evaluated.

The MedGemma accuracies equal the always-positive baseline: 472 / 488 = 0.9672 and 465 / 477 = 0.9748. Its macro-F1 was approximately 0.49 because negative-class F1 was zero, and the test split confirmed this failure mode. MiniCPM-V's perfect validation score reflects correct classification of both observed classes, including all 16 negative controls, but its test negative-control recall was only 1/12 = 0.0833. All validation negatives come from one source-video group (`negative_video_008`); the 12 MedGemma test negatives come from only two groups (`negative_video_007` and `negative_video_009`). These small counts limit interpretation. No confidence interval or external-domain claim is made.

The evaluation-only bundles contain a `training_summary.json` whose `train_records` field reflects a compact evaluation placeholder, not the original training run. Training provenance (2,231 records) comes from the archived training log and split counts; the saved predictions and confusion matrices supply the evaluation counts. This distinction prevents the evaluation bundle from being mistaken for evidence of a 488-record training run.

## 5.6 Auxiliary synthetic T2 and attribution diagnostics

The T1-trained adapter was also probed on **four synthetic** T2 validation cases and **eight synthetic** hallucination probes. Diagnosis exact match and drug exact match were both 0/4 in the validation subset and 0/8 in the probe subset; none of the eight probes generated a referral response. For example, synthetic case `CHL-SYN-000713` has reference diagnosis “Canine Parvovirus,” but its archived raw generation begins `trypanosome_presentmodel` followed by repeated `model` fragments, rather than a diagnosis and treatment plan. This excerpt is a literal substring of the saved output, not a representative clinical answer. It shows that the *T1 adapter*, on these tiny synthetic samples, was not functioning as a T2 recommendation model; it is not a measured rate of unsafe clinical advice or a test of a T2-trained model.

**Worked image-explanation example.** Four T1 validation images were processed with 4-by-4 tile occlusion: the unmasked image and each image with one tile replaced by its mean colour were scored for the fixed margin `score(trypanosome_present) - score(negative_control)`. For positive image `tryp-6b06d097f90b7576`, the unmasked margin was 56.01, and all 16 recorded baseline-minus-masked margin changes were *negative*: covering each tile increased rather than reduced the positive-class margin. It has eight publisher parasite boxes, but the positive-attribution mass inside them is **undefined** (`null`), because there is no positive tile change to normalise. The second box-annotated positive image has the same undefined result. An earlier aggregate incorrectly treated two negative-control images *without* boxes as valid zero-mass examples. The corrected evaluator excludes them: its two box-bearing examples have null mass, and it reports no valid positive pointing-game result (`data/secondary/reports/xai_corrected_summary.json`). No verified Grad-CAM output exists.

**Worked text-explanation example.** Four *synthetic* T2 cases have ten-field KernelSHAP attributions. For `CHL-SYN-000911` the fixed, teacher-forced target is `Diagnosis: Bovine Mastitis`; the stored score-attribution values include 7.79 for presenting signs, 4.50 for species and 4.07 for culture and sensitivity. These are score contributions, **not probabilities**. The method masks combinations of structured prompt fields and measures how the fixed diagnosis-line score changes. It does **not** explain a successful generated answer: that case's raw generation begins `trymodel`, then repeats `model` fragments. The corrected XAI summary identifies the raw per-item method as `field_shap`, rather than the original report's erroneous `field_occlusion` label. This is field-level SHAP, not token-level SHAP; four synthetic examples do not establish faithfulness.

## 5.7 Local ETL result and execution boundary

The metadata-only ETL read all 3,196 eligible Tryp image manifest rows and four published-source text-item rows, checked source registration, excluded provenance and group identifiers, and wrote 3,200 clean JSONL rows with zero rejections (`data/secondary/processed_full/etl_report.json`). Elapsed wall-clock time inside the Python process was 0.063 seconds on this host. That figure excludes image loading, resizing, model work and Spark startup and is a single unreplicated timing, not a throughput benchmark. `scripts/spark_etl_secondary.py` implements local PySpark manifest loading, broadcast source-register join, eligibility checks, Spark JSON output and a timed count/report, but it could not be run because the PySpark dependency was unavailable and package installation timed out. No Spark timing, count or speed-up is inferred from the Python pass.

## 5.8 Results available and results not established

The dataset audit, full local Python metadata pass, public T1 split, MedGemma evaluation, balanced MiniCPM-V training and validation/test evaluations, four-query Chroma development check, ten-query source-disjoint FTS5 check, and bounded auxiliary diagnostic artifacts are completed. No base-model comparison was run; the four base/adapted × retrieval conditions were designed but not run as a controlled comparison. There is no held-out Chroma query score, Spark run, held-out T2 source-supported factuality or ROUGE-L result, measured RAG reduction in unsupported claims, reliable Grad-CAM or explanation-faithfulness result, or clinician or Kenyan field evaluation. Keeping these boundaries visible is necessary to interpret the completed evidence correctly.

# 6. Discussion

## 6.1 Answer to RQ1

The original MedGemma adapter predicted `trypanosome_present` for all 488 validation records and all 477 test records, with accuracies 0.9672 and 0.9748, macro-F1 0.4917 and 0.4936, and zero negative-control recall. The separately trained, class-balanced MiniCPM-V 4.6 adapter correctly classified all 488 validation records but only one of 12 negative controls in the 477-record test split. Its test accuracy was 0.9769, macro-F1 0.5711 and balanced accuracy 0.5417. The validation result did not carry over to balanced test performance. Neither architecture has a base-model comparison, so the effect of adaptation is unknown. The T2 probes used the earlier T1-trained adapter; their zero exact matches cannot be interpreted as performance of a T2-specific model.

## 6.2 Answer to RQ2

The Chroma smoke test shows that the RAG *retriever* is operational, and the treatment filter can recover the passage missed in development (Section 5.4). Because the filter was added after seeing the miss, its four-query score cannot establish an out-of-sample gain. A separate lexical retriever identified ten source-disjoint, in-corpus passages at rank one, even in an exploratory run without disease filtering. This tests a different retrieval engine and relatively easy source-derived questions; it neither validates the tuned Chroma filter nor measures RAG answer factuality. Whether retrieved evidence reduces unsupported claims remains unanswered: there is no paired held-out generation comparison.

## 6.3 Answer to RQ3

The repaired aggregate confirms that no box-bearing image had a defined positive-attribution mass or valid positive pointing target; the original zero was a reporting bug, not evidence against localisation. The four synthetic field-level SHAP outputs score a reference diagnosis line, not the malformed generated answers. Spatial alignment, faithfulness and stability therefore remain unestablished, and no Grad-CAM result is claimed.

## 6.4 Answer to RQ4

The implementation demonstrated source validation, 896 by 896 image normalisation, source-video grouped splitting, mixed-task instruction construction, knowledge-base validation and retrieval evaluation. The full local Python ETL accepted 3,200/3,200 metadata rows with a single measured elapsed time of 0.063 seconds. Separate MedGemma QLoRA and MiniCPM-V MLX-LoRA runs produced validation and test predictions. For an MSc Big Data Technologies dissertation, the defensible contribution is the reproducible handling of heterogeneous, provenance-sensitive data and correlated records, not a claim that this modest dataset required a distributed cluster. Spark execution and throughput, training-energy use, peak memory, comparative inference latency and comprehensive explanation costs were not measured. RQ4 is answered for local metadata processing and auditability, not distributed performance.

## 6.5 Strengths

The main strength is the agreement between research design and evidence boundary. Explicit source statuses prevent unverified local images from being mistaken for labelled data. Group IDs prevent source-video leakage. MedGemma's validation and test evaluations used the same adapter payload. MiniCPM's paired validation and test results show that a perfect validation score did not generalise to balanced performance on the small test split. The four-condition design would separate adaptation from retrieval if executed; it is a reproducible specification, not an achieved comparative result. The metric design also separates citation presence from factual source support.

A second strength is the use of a public animal-parasite dataset rather than silently treating human malaria as veterinary evidence. A third strength is the recognition that explanation methods require model-specific targets. A fourth is the local retrieval smoke test, which already exposed an East Coast Fever ranking problem before model training could hide it behind fluent output.

## 6.6 Limitations

The principal limitation is incomplete empirical execution beyond the bounded T1 evaluation. Neither model was repeated across seeds and no base-model comparison was saved. Negative examples remain scarce: 65 training, 16 validation and 12 test examples; all 16 validation negatives belong to one group. MiniCPM correctly classified only one of the 12 test negatives. Although source-video groups were separated, groups are not necessarily independent animals or collection sites; adjacent frames within a group remain correlated. The dissertation can report a public-proxy T1 result, but cannot present it as clinical performance or generalise it to Kenyan cattle. The four-query Chroma retrieval set is development-only. The ten-query FTS5 set excludes those four source groups, but tests a different algorithm on in-corpus, source-derived labels rather than answering whether the Chroma filter generalises. The Tryp benchmark is mouse, *T. brucei brucei* and unstained thick smear data, so transfer to Kenyan cattle is uncertain. Mendeley, NIH and BBBC041 sources have additional grouping, species and licence limitations.

The RAG corpus is curated from literature but has source-specific permissions and may contain historical or context-dependent treatment statements. A current veterinary clinician would be required for any real clinical deployment, but recruiting or asking clinicians to validate data is outside this study. Computational source checking is not equivalent to clinical endorsement.

Grad-CAM and SHAP can be technically difficult on quantised, parameter-efficient generative multimodal models. Heatmap overlap with a box does not prove causal reasoning, and token attribution can be unstable under prompt changes. Controlled deletion tests are stronger than visual inspection but still depend on masking choices.

The absence of a participant study means that clinician trust, perceived usefulness, ease of understanding and workflow usability cannot be estimated. Published human-centred AI literature informs the design rationale only. The results cannot be generalised to Cherehani Labs or any other laboratory without prospective external validation.

## 6.7 Implications and future work

Further work should evaluate the MiniCPM adapter on additional independent datasets or source groups and compare it with its base checkpoint. Run the new source-disjoint questions through Chroma before asserting that its section filter generalises; use less lexically overlapping questions and independent relevance judgments for a stronger retrieval test. Public source-derived T2 items should be expanded across independent source families before comparing the four base/adapted × retrieval conditions. Claim-level factuality should be scored against a common evidence ledger rather than citation presence alone.

Future work could add livestock-specific public microscopy sources if their labels, licence and grouping metadata are adequate. It could also examine a dedicated veterinary vision model rather than a general medical transfer model. A participant study involving clinicians could evaluate trust and workflow only after separate ethics approval and only if such first-hand research becomes permitted. That future work must not be written into the present results.

# 7. Conclusion

This dissertation revised the original MedGemma veterinary diagnostics proposal to comply with the prohibition on first-hand data collection. The resulting study combines public microscopy data, published veterinary knowledge, a measured 3,200-row local Python metadata pass, ChromaDB development retrieval, a ten-query source-disjoint SQLite FTS5 check and two distinct T1 adapter runs: MedGemma QLoRA and class-balanced MiniCPM-V MLX-LoRA. Spark remains an unexecuted engineering extension, and the synthetic auxiliary T2 corpus is not clinical evidence. Laboratory data collection, clinician surveys and trust claims are outside the completed study.

The work established a traceable secondary-data source boundary, 3,196 processed public Tryp images in grouped splits, a literature KB and an auditable four-item Chroma retrieval check. The latter exposed a missed treatment passage and motivated a filter correction; because that correction was made on the development examples, the resulting score must not be sold as an independently tested improvement. The separate lexical check returned ten other source-family passages at rank one, but does not test Chroma. A reporting bug in the XAI summary was also corrected from the saved per-item data, with undefined values preserved rather than converted to reassuring zeros.

The initial MedGemma model failed to recognise any of 28 negative controls across its validation and test sets, despite apparently high accuracy. A separate balanced MiniCPM-V 4.6 run classified all 488 validation images correctly, then identified only one of 12 negative controls on test. The validation result therefore did not establish balanced generalisation. Both model runs remain bounded to one public mouse-model dataset, and neither supports claims of veterinary diagnostic readiness.

# References

Anzaku, E. T., Mohammed, M. A., Ozbulak, U., Won, J., Hong, H., Krishnamoorthy, J., Van Hoecke, S., Magez, S., Van Messem, A., & De Neve, W. (2023). Tryp: A dataset of microscopy images of unstained thick blood smears for trypanosome detection. *Scientific Data, 10*, 716. https://doi.org/10.1038/s41597-023-02608-y

Desquesnes, M., Gonzatti, M., Sazmand, A., Thévenon, S., Bossard, G., Boulangé, A., Gimonneau, G., Truc, P., Herder, S., Ravel, S., Sereno, D., Jamonneau, V., Jittapalapong, S., Jacquiet, P., Solano, P., & Berthier, D. (2022). A review on the diagnosis of animal trypanosomoses. *Parasites & Vectors, 15*. https://doi.org/10.1186/s13071-022-05190-1

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *Advances in Neural Information Processing Systems, 36*. https://arxiv.org/abs/2305.14314

Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M., & Wang, H. (2023). Retrieval-augmented generation for large language models: A survey. *arXiv*. https://arxiv.org/abs/2312.10997

Hoffman, R. R., Mueller, S. T., Klein, G., & Litman, J. (2018). Metrics for explainable AI: Challenges and prospects. *arXiv*. https://arxiv.org/abs/1812.04608

Kihu, S. M., Gachohi, J. M., Ndungu, E. K., Gitao, G. C., Bebora, L. C., John, N. M., Wairire, G. G., Maingi, N., Wahome, R. G., & Ireri, R. (2015). Sero-epidemiology of Peste des petits ruminants virus infection in Turkana County, Kenya. *BMC Veterinary Research, 11*, 87. https://doi.org/10.1186/s12917-015-0401-1

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-T., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459–9474. https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html

Li, S., & Zhang, Y. (2020). *Microscopic Images of Parasites Species* (Version 2) [Data set]. Mendeley Data. https://doi.org/10.17632/38jtn4nzs6.2

Ljosa, V., Sokolnicki, K. L., & Carpenter, A. E. (2012). Annotated high-throughput microscopy image sets for validation. *Nature Methods, 9*(7), 637. https://doi.org/10.1038/nmeth.2083

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

Mbindyo, C. M., Gitao, G. C., & Mulei, C. M. (2020). Prevalence, etiology, and risk factors of mastitis in dairy cattle in Embu and Kajiado Counties, Kenya. *Veterinary Medicine International, 2020*, 8831172. https://doi.org/10.1155/2020/8831172

Morais, M. C. C., Silva, D., Milagre, M. M., Oliveira, M. T., Pereira, T., Silva, J. S., Costa, L. F., Minoprio, P., Junior, R. M. C., Gazzinelli, R., de Lana, M., & Nakaya, H. I. (2022). Automatic detection of the parasite *Trypanosoma cruzi* in blood smears using a machine learning approach applied to mobile phone images. *PeerJ, 10*, e13470. https://doi.org/10.7717/peerj.13470

Okello, I., Mafie, E., Eastwood, G., Nzalawahe, J., Mboera, L. E. G., & Onyoyo, S. (2022). Prevalence and associated risk factors of African animal trypanosomiasis in cattle in Lambwe, Kenya. *Journal of Parasitology Research, 2022*, 5984376. https://doi.org/10.1155/2022/5984376

Patel, E., Mwaura, S., Di Giulio, G., Cook, E. A. J., Lynen, G., & Toye, P. (2019). Infection and treatment method vaccine against East Coast fever: Reducing the number of doses per straw for use in smallholder dairy herds. *BMC Veterinary Research, 15*, 46. https://doi.org/10.1186/s12917-019-1787-y

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Sellergren, A., Kazemzadeh, S., Jaroensri, T., Kiraly, A., Traverse, M., Kohlberger, T., Xu, S., Jamil, F., Hughes, C., Lau, C., Chen, J., Mahvar, F., Yatziv, L., Chen, T., Sterling, B., Baby, S. A., Baby, S. M., Lai, J., Schmidgall, S., . . . Yang, L. (2025). *MedGemma technical report* (arXiv:2507.05201). https://doi.org/10.48550/arXiv.2507.05201

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 618–626). https://doi.org/10.1109/ICCV.2017.74

Shneiderman, B. (2020). Human-centered artificial intelligence: Reliable, safe and trustworthy. *International Journal of Human–Computer Interaction, 36*(6), 495–504. https://doi.org/10.1080/10447318.2020.1741118

Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., Scales, N., Tanwani, A., Cole-Lewis, H., Pfohl, S., Papadimitriou, P., Seneviratne, M., Chandrasekaran, P., Connelly, P., Kjellstrom, H., Liu, H., Eslami, S. M. A., Razi, A., & Natarajan, V. (2023). Large language models encode clinical knowledge. *Nature, 620*, 172–180. https://doi.org/10.1038/s41586-023-06291-2

Torres, K., Bachman, C. M., Delahunt, C. B., Alarcon Baldeon, J., Alava, F., Gamboa Vilela, D., Proux, S., Mehanian, C., McGuire, S. K., Thompson, C. M., Ostbye, T., Hu, L., Jaiswal, M. S., Hunt, V. M., & Bell, D. (2018). Automated microscopy for routine malaria diagnosis: A field comparison on Giemsa-stained blood films in Peru. *Malaria Journal, 17*. https://doi.org/10.1186/s12936-018-2493-0

Tu, T., Azizi, S., Driess, D., Mirchandani, S., Natarajan, V., Chang, P.-C., Carroll, A., Lau, C., Tanwani, A., Cole-Lewis, H., Eslami, S. M. A., Pfohl, S., & Singhal, K. (2024). Towards generalist biomedical AI. *NEJM AI, 1*(3). https://doi.org/10.1056/AIoa2300138

Ward, P., Dahlberg, P., Lagatie, O., Larsson, J., Tynong, A., Vlaminck, J., Zumpe, M., Ame, S., Ayana, M., Khieu, V., Mekonnen, Z., Odiere, M., Yohannes, T., Van Hoecke, S., Levecke, B., & Stuyver, L. J. (2022). Affordable artificial intelligence-based digital pathology for neglected tropical diseases: A proof-of-concept for the detection of soil-transmitted helminths and *Schistosoma mansoni* eggs in Kato-Katz stool thick smears. *PLOS Neglected Tropical Diseases, 16*(6), e0010500. https://doi.org/10.1371/journal.pntd.0010500

Zaharia, M., Xin, R. S., Wendell, P., Das, T., Armbrust, M., Dave, A., Meng, X., Rosen, J., Venkataraman, S., Franklin, M. J., Ghodsi, A., Gonzalez, J., Shenker, S., & Stoica, I. (2016). Apache Spark: A unified engine for big data processing. *Communications of the ACM, 59*(11), 56–65. https://doi.org/10.1145/2934664

## Appendix A: Evidence and reproducibility map

The results in Chapter 5 can be traced to locally saved machine-readable artifacts. Where an archive contains both original outputs and evaluation-only metadata, the original run log is the source for training provenance and the evaluation metrics are the source for held-out results.

| Result or procedure | Evidence in the project directory |
|---|---|
| Source eligibility and local exclusions | `data/secondary/reports/source_audit.json`; `data/secondary/sources/source_register.json` |
| Tryp parsing and grouped image split | `data/secondary/manifests/images_raw.jsonl`; `data/secondary/splits/images/`; `data/secondary/manifests/images_prepared.jsonl` |
| Processed public and synthetic instruction sets | `data/secondary/instruction_tuning_mixed/`; synthetic records carry auxiliary provenance flags |
| Four-query retrieval result | `data/secondary/reports/retrieval_eval.json`; `data/secondary/text/retrieval_eval.jsonl` |
| Source-disjoint lexical retrieval check | `data/secondary/text/retrieval_heldout.jsonl`; `data/secondary/reports/retrieval_heldout_lexical.json`; `scripts/evaluate_retrieval_lexical.py` |
| Full local metadata ETL | `data/secondary/processed_full/etl_report.json`; `data/secondary/processed_full/clean_metadata.jsonl`; `scripts/etl_secondary.py` (Python, not Spark) |
| MedGemma T1 validation and test | `docs/T1_VALIDATION_RESULTS.md`; raw model archives are local and excluded from version control |
| MiniCPM-V 4.6 balanced training, validation and test | `data/secondary/reports/minicpm_t1_metrics.json`; `docs/T1_MINICPM_RESULTS.md`; `mlx_t1_balanced_train_minicpm_v46.ipynb` |
| Bounded synthetic T2 and XAI diagnostics | Corrected summary in `data/secondary/reports/xai_corrected_summary.json`; `docs/XAI_T2_RESULTS.md`; raw model archives are local and excluded from version control |

The configuration is recorded in `configs/secondary.json`. The Tryp preparation and split programs are `scripts/parse_tryp_dataset.py`, `scripts/normalize_tryp_images.py` and `scripts/build_secondary_splits.py`. Training and evaluation interfaces include `scripts/train_t1_qlora.py`, `scripts/train_t1_minicpm_mlx.py`, `scripts/evaluate_t1.py` and `scripts/compare_t1_runs.py`; the unexecuted Spark path is `scripts/spark_etl_secondary.py`. The pinned model revisions and run settings are stated in Sections 3.7 and 5.5. MiniCPM's test metrics are retained in the machine-readable summary; raw predictions and weights remain local.

## Appendix B: Interpretation limits

The public Tryp result does not establish performance on Cherehani Labs or Kenyan cattle samples. The Chroma queries were used in development; ten additional groups were reserved from that development set but evaluated only with a different lexical engine against in-corpus labels. Neither set constitutes an independent clinical benchmark. Synthetic T2 cases cannot stand in for patient records. The corrected XAI summary sets box-mass and pointing-game means to null because no positive box-bearing example supports those calculations. None of these observations measures treatment safety or clinician trust.
