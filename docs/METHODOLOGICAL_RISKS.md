# Methodological Risks

> A candid register of where the proposal's plan is technically or statistically shaky. **This is for
> your benefit** — each item is written so you can lift it into your Discussion/Limitations chapter
> (which you write yourself). Examiners reward a candidate who *anticipates* these, not one who is
> caught out by them. For each: **the risk**, **why it matters for the mark**, and **the mitigation**.

The honest meta-point: the proposal is ambitious and technically literate, but it treats several
hard sub-problems as one-line steps. None is fatal. All are manageable if named early. Ranked roughly
by how much they threaten the result.

---

## 1. Grad-CAM on a Vision Transformer is not the CNN method the proposal implies

**Risk.** Grad-CAM (Selvaraju et al. 2017) was built for CNN feature maps with native 2-D spatial
structure. MedGemma's image encoder is a **SigLIP ViT**: activations are `[batch, tokens, channels]`,
not `[batch, C, H, W]`. Naïve Grad-CAM produces meaningless heatmaps. Worse, MedGemma is
**generative** — Task 1 emits the pathogen name as *text*, so there is no softmax class score to
backprop from. Both facts are unaddressed in the proposal.

**Why it matters for the mark.** "Grad-CAM localisation accuracy" is a headline metric (§3.3). If the
heatmaps are artefacts of a mis-applied method, the metric is invalid and an examiner who knows ViTs
will spot it. The XAI chapter is a core novelty claim.

**Mitigation.** (a) Use a `reshape_transform` to fold patch tokens back to a 2-D grid and pick the
correct target layer (final SigLIP encoder LayerNorm) — see IMPLEMENTATION_GUIDE §5.1. (b) Define a
differentiable **class target** by wrapping the logit of the class-name token(s), so Grad-CAM has a
real class signal. (c) Keep **attention rollout** as a disclosed fallback if the generative-target
Grad-CAM proves intractable — and then call it "attention-based localisation," not Grad-CAM.
**Disclose the exact target and reshape you used.** Turning this subtlety into an explicit methods
paragraph is itself a mark-earning contribution.

---

## 2. SHAP token attribution is expensive on a 4B generative model

**Risk.** SHAP for text generation (PartitionSHAP + TeacherForcing) scales badly with input and
generation length. Your T2 prompts are **long** (clinical notes + retrieved RAG context), and each
output token gets its own attribution. Running SHAP over the full test set is computationally
infeasible on Kaggle time budgets — the literature (e.g. TextGen-SHAP) exists precisely because
vanilla SHAP is impractical at this scale.

**Why it matters for the mark.** If you promise SHAP over all test cases and deliver a handful, that
reads as under-delivery unless you *planned* the reduction and justified it.

**Mitigation.** Scope SHAP to a **reduced probe set** (20–40 cases, including hallucination probes),
cap `max_new_tokens` to the clinically load-bearing span (drug/dose), and attribute over one prompt
segment at a time. Time a single explanation before scaling. Disclose the sample size as a deliberate
design choice driven by cost, not an accident. Have **LIME-for-text** or **integrated gradients
(captum)** ready as a cheaper, defensible substitute if SHAP stalls.

---

## 3. Dataset size vs a 4B model ("the biggest limitation" — the proposal already admits it)

**Risk.** The proposal names dataset size as its biggest limitation and leans on synthetic cases +
transfer learning. A 4B model can overfit a small instruction set fast, and **synthetic clinical data
can bake in the generator's own biases** — if an LLM wrote the treatment plans, fine-tuning on them
teaches the model to imitate an LLM, not ground truth. The RAG corpus being real (spec §4.3) is the
saving grace; the synthetic clinical text is the exposure.

**Why it matters for the mark.** Impressive metrics on a tiny, partly-synthetic test set are not
credible evidence of clinical capability. An examiner will probe generalisation.

**Mitigation.** Keep synthetic content clearly labelled (spec §0) and **validated by Cherehani staff**
before use — the spec already frames generated data as *pre-validation drafts*. Prefer real
microscopy images for the T1 test set; exclude the malaria proxy class from vet evaluation (spec §2).
Report train/val/test sizes honestly, watch val loss for overfitting (early stop), and frame results
as **feasibility evidence**, not deployment-ready performance.

---

## 4. Grouped-split leakage will silently inflate every number

**Risk.** The same `case_id` can generate both a T1 image record and a T2 text record. A naïve random
split puts related records in different splits → the model sees near-duplicates of test cases during
training → inflated accuracy/F1/ROUGE. The spec §5 demands grouped, stratified splits precisely to
prevent this, but it's easy to get wrong in code.

**Why it matters for the mark.** Leakage is a classic ML methodology failure; if discovered, it
invalidates the central base-vs-finetuned comparison. Reviewers actively look for it.

**Mitigation.** Use `StratifiedGroupKFold` with `groups=case_id`. Write and commit an assertion that
`set(train.case_id) ∩ set(test.case_id) == ∅` (and val). Verify the priority-four diseases appear in
all splits and no synthetic-only disease is test-exclusive (spec §5). Document the split seed for
reproducibility.

---

## 5. "Hallucination rate" is undefined in the proposal — you must define it or it's unmeasurable

**Risk.** The proposal uses "hallucination rate" as a headline metric (§3.3, and central to the RAG
hypothesis) but **never defines it**. An undefined metric cannot be computed, compared, or defended.

**Why it matters for the mark.** It's the metric that tests the proposal's core RAG claim ("grounding
reduces hallucination"). Leaving it undefined undermines the main quantitative result.

**Mitigation.** Adopt an explicit operational definition (proposed in IMPLEMENTATION_GUIDE §7.2):
**hallucination rate = unsupported-or-contradicted clinical claims ÷ total clinical claims**, where a
claim is an atomic assertion about drug/dose/route/frequency/duration/diagnosis, and "supported" means
backed by a retrieved RAG source or the case evidence. Complement with the **over-confident
fabrication rate** on the hallucination probe set (cases whose correct answer is "insufficient
evidence / refer to clinician"). Score by rules (citation presence + structured-field match) and/or an
LLM-judge validated against a clinician-audited subset. **State the definition and its limits up
front** — a well-argued definition is a contribution, not a weakness.

---

## 6. Clinician n is tiny — quantitative trust claims will be underpowered

**Risk.** Cherehani Labs has few clinicians (realistically n ≈ 3–8). Hoffman-style Likert comparisons
and any "XAI significantly improves trust" claim are **statistically underpowered** at that n. A
significance test on 5 respondents is not credible.

**Why it matters for the mark.** Over-claiming significance from a tiny sample is a methodological red
flag; examiners will downgrade unsupported quantitative inference.

**Mitigation.** Design Phase 5 as **qualitative-primary**: rich thematic analysis (Braun & Clarke)
plus descriptive stats, framed as **exploratory**. Add a **behavioural error-detection task** (does
XAI help the vet catch a wrong AI output?) — a within-subjects effect that is more informative than
Likert means at small n. If you report any test, use non-parametric (Wilcoxon signed-rank), report
effect sizes + CIs, and **explicitly state the power limitation**. Candour here earns marks; false
precision loses them.

---

## 7. Kaggle GPU quota vs the number of training/eval runs required

**Risk.** ~30 GPU-h/week and ~9–12 h session caps. You need: base-model eval, ≥1 successful finetune
(likely several attempts after collator/OOM bugs), XAI passes, and re-runs after fixes. Naïve
trial-and-error will exhaust the weekly quota and a session-kill mid-run **loses all progress** if you
haven't checkpointed off the ephemeral disk.

**Why it matters for the mark.** Compute starvation is the most likely cause of the timetable slipping
and of an incomplete results chapter — a delivery risk, not just an inconvenience.

**Mitigation.** Checkpoint every N steps and **push checkpoints to a persistent Kaggle Dataset**;
resume with `resume_from_checkpoint` (IMPLEMENTATION_GUIDE §4.5). Sanity-check the pipeline by
overfitting 10 samples (minutes, not hours) before committing a full run. Plan runs on paper; budget
~5–6 runs/week. Keep the eval harness ready **before** training finishes so a completed checkpoint is
evaluated immediately, not in a fresh quota-burning session. Consider Colab as an overflow GPU source.

---

## 8. Smaller but real

- **Base-vs-finetuned must use the same base revision.** MedGemma model cards get updated (note the
  9 Jul 2025 multimodal token bug fix). Pin the exact revision hash for both runs, or the comparison
  is confounded. (IMPLEMENTATION_GUIDE §8.)
- **Multimodal collator silent failure.** A mis-built collator can train with the image never reaching
  the model — loss looks normal, T1 never learns. Verify with the overfit-10 check. (§4.3.)
- **ROUGE-L rewards surface overlap, not clinical correctness.** Report it, but lead with
  factual-consistency for T2 claims. (§7.2.)
- **Merging a 4-bit base for inference** can misbehave; you may need to reload the base in bf16 before
  `merge_and_unload`, or serve base + adapter separately. (§4.6.)
- **Grad-CAM localisation accuracy needs ground-truth parasite locations.** If you lack boxes/masks,
  a clinician-rated "does the heatmap point at the parasite" measure is the honest fallback — but say
  which one you used. (§7.1.)
- **HAI-DEF license restricts use to research, not clinical advice.** Consistent with spec §0; state
  it in ethics and datasheets so there's no ambiguity about deployment claims.

---

*Cross-reference:* every mitigation here is implemented or expanded in
[`IMPLEMENTATION_GUIDE.md`](./IMPLEMENTATION_GUIDE.md). Use this file to seed your Limitations chapter;
use that file to do the work.
</content>
