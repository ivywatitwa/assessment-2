# Implementation Guide — MedGemma Veterinary Diagnostics

> **A step-by-step engineering runbook** for the dissertation *"Fine-Tuning MedGemma for
> Multimodal Explainable Veterinary Diagnostics"* (UEL-CN-7000). You run the work; this guide
> makes each step tractable. Read alongside [`PROJECT_SPEC.md`](./PROJECT_SPEC.md) (the data
> contract) and [`METHODOLOGICAL_RISKS.md`](./METHODOLOGICAL_RISKS.md) (where the plan is shaky).

**Conventions used in this guide**
- `> ⚠️ VERIFY:` — an API/detail that moves fast or that I could not confirm to the byte. Check it
  yourself before relying on it. Do **not** treat these as settled.
- Code blocks are runnable starting points, not copy-paste-and-forget. Paths follow the
  §7 directory contract in the spec.
- Everything is dated against the tool state of **mid-2026**. Re-verify model IDs and library
  versions when you actually run this — they change monthly.

---

## 0. Prerequisites & reality check

### 0.1 The single most important verified fact: which MedGemma?

There are **multiple MedGemma variants** on Hugging Face and choosing wrong wastes weeks.

| HF model ID | Params | Vision encoder? | Can do Task 1 (microscopy image)? | Fits Kaggle 16 GB QLoRA? |
|---|---|---|---|---|
| `google/medgemma-4b-it` | ~4B | **Yes** (SigLIP) | **Yes** | **Yes** |
| `google/medgemma-4b-pt` | ~4B | Yes (SigLIP) | Yes (but not instruction-tuned) | Yes |
| `google/medgemma-1.5-4b-it` | ~4B | Yes (SigLIP) | Yes (newer revision) | Yes |
| `google/medgemma-27b-text-it` | ~27B | **No — text only** | **NO** | No |
| `google/medgemma-27b-it` | ~27B | Yes | Yes, but | No (too big for 16 GB) |

**Decision: use `google/medgemma-4b-it`.** Reasons, all verified against the HF model card:

1. **Task 1 requires a vision encoder.** Microscopy classification takes a slide *image* as input.
   `medgemma-27b-text-it` is **text-only — it has no image pathway at all** and physically cannot
   ingest the slide. The proposal's "MedGemma4" is therefore correctly read as the **4B multimodal**
   variant, not the 27B text model.
2. **The 27B multimodal model (`medgemma-27b-it`) does exist** and could technically do both tasks,
   but even in 4-bit it will not train under QLoRA on a 16 GB Kaggle GPU (base weights alone are
   ~14–16 GB in NF4). The proposal commits to free Kaggle GPUs, so 27B is out.
3. `medgemma-4b-it` is Gemma-3-4B + a **SigLIP** image encoder pre-trained on de-identified medical
   images (CXR, derm, ophthalmology, **histopathology** — the last is the closest analogue to your
   microscopy slides). Image input is **896×896**, encoded to **256 image tokens**; context is up to
   **128K tokens**; output up to 8,192 tokens.

> ⚠️ VERIFY: `medgemma-1.5-4b-it` is a newer revision. If you start fresh, quickly compare its model
> card to `medgemma-4b-it` and pick the latest **4B multimodal instruction-tuned** one. Pin whichever
> you choose (record the exact revision hash) so your results are reproducible. **Never** silently
> swap variants mid-project — a base-vs-finetuned comparison across different base models is invalid.
>
> Note the **9 July 2025 bug fix** (missing end-of-image token) that affected multimodal performance.
> Make sure you pull a revision **after** that fix, or your base-model image numbers will be wrong.

### 0.2 The access-gate blocker (do this in week 1, not week 8)

MedGemma is **gated under the Health AI Developer Foundations (HAI-DEF) terms of use** — this is a
real blocker, not a formality:

1. Create/log in to a Hugging Face account.
2. Go to `https://huggingface.co/google/medgemma-4b-it` and click **"Acknowledge license"** to accept
   the HAI-DEF terms. Requests are processed immediately, but you **cannot download weights** until
   you do this.
3. Create a HF access token (read scope): `https://huggingface.co/settings/tokens`.
4. On Kaggle, add the token as a **Kaggle Secret** named `HF_TOKEN` (Add-ons → Secrets). Never paste
   it into a notebook cell.

> The HAI-DEF terms restrict use: synthetic/clinical outputs are **research only, not clinical
> advice** — this matches the spec's non-negotiable labelling rule (§0). State this in your ethics
> application and in every dataset datasheet.

### 0.3 Hardware reality check — Kaggle GPUs

Verified current free-tier limits (re-check on Kaggle, they shift):

- **~30 GPU-hours/week**, reset weekly.
- One session: choose **P100 (16 GB)** *or* **2×T4 (16 GB each, 32 GB total)**.
- **Max session length ~9–12 hours**, then the kernel is killed. **This is the single biggest
  practical trap** (see §4.5 checkpointing).
- 2×T4 consumes quota at 1×, but naïve single-process `Trainer` uses **only one** of the two T4s
  unless you configure multi-GPU. For a 4B QLoRA, **one 16 GB card is enough** — prefer P100 for
  simplicity, or a single T4.

### 0.4 Does a 4B QLoRA actually fit in 16 GB? — the VRAM arithmetic

Do this arithmetic yourself; it is the difference between "trains" and OOM.

| Component | Calculation | VRAM |
|---|---|---|
| Base weights, 4-bit NF4 | ~4.3B params × 0.5 byte + quant constants | **~2.5–3.0 GB** |
| Vision tower (SigLIP, ~400M) if kept in bf16 | 0.4B × 2 bytes | **~0.8 GB** |
| LoRA adapters (r=16, all-linear ≈ 40–80M trainable) weights | 60M × 2 bytes | ~0.12 GB |
| Optimizer states (AdamW: m+v fp32) on adapters | 60M × 8 bytes | ~0.48 GB |
| Gradients (bf16) on adapters | 60M × 2 bytes | ~0.12 GB |
| **Activations** (the variable one) — seq≈1024, batch=1, **grad checkpointing ON**, +256 image tokens | empirical | **~3–6 GB** |
| CUDA context / fragmentation | — | ~1–2 GB |
| **Total** | | **~8–13 GB** |

**Conclusion: a 4B QLoRA fine-tune fits on a single 16 GB P100/T4** with `per_device_batch_size=1`,
gradient accumulation for effective batch size, gradient checkpointing ON, and sequence length capped
at ~1024–2048. If you OOM: lower seq length first, then confirm checkpointing is actually enabled,
then reduce image count per sample. **Do not** raise batch size above 1–2 on 16 GB.

### 0.5 Skills you need before starting

PySpark basics, Hugging Face `transformers`/`datasets`, PEFT/LoRA concept, basic PyTorch hooks (for
Grad-CAM), and survey design. Budget a week of ramp-up if any are new.

---

## 1. Environment setup

Two environments: (a) a **local/Spark** env for Phase 1 ETL and RAG build (CPU is fine), and (b) the
**Kaggle GPU** env for Phases 3–5 (training + XAI). Keep them separate — Spark and CUDA training have
conflicting dependency trees.

### 1.1 `requirements-etl.txt` (local, Phase 1–2, CPU)

```
# Pin these — re-verify on install
pyspark==3.5.1
pandas==2.2.2
pyarrow==16.1.0
pillow==10.3.0
chromadb==0.5.5
langchain==0.2.16
langchain-community==0.2.16
langchain-huggingface==0.0.3
sentence-transformers==3.0.1
jsonschema==4.22.0
python-dotenv==1.0.1
```

> ⚠️ VERIFY: ChromaDB and LangChain APIs churn hard. `chromadb>=0.5` changed the client constructor
> (`chromadb.PersistentClient(path=...)`); LangChain split into `langchain-core` /
> `langchain-community` / provider packages. Pin whatever versions you install and check the
> import paths in §3 against the installed version's docs (use Context7 / the official docs).

### 1.2 Kaggle GPU env (Phase 3–5)

Kaggle images ship recent CUDA + torch. Install the ML stack at notebook top. Gemma 3 needs
**transformers ≥ 4.50.0** (verified: "Gemma 3 is supported starting from transformers 4.50.0").

```python
# First cell of every Kaggle training/XAI notebook
!pip install -q -U \
    "transformers>=4.50.0" \
    "peft>=0.12.0" \
    "trl>=0.9.6" \
    "bitsandbytes>=0.43.1" \
    "accelerate>=0.33.0" \
    "datasets>=2.20.0" \
    "evaluate>=0.4.2" \
    "rouge-score>=0.1.2" \
    "grad-cam>=1.5.0" \
    "shap>=0.46.0"
```

> ⚠️ VERIFY: The official MedGemma fine-tuning notebook
> (`Google-Health/medgemma/notebooks/fine_tune_with_hugging_face.ipynb`) installs
> `bitsandbytes datasets evaluate peft tensorboard transformers trl` **without pins** and uses
> `model_id = "google/medgemma-4b-it"`. **Open that notebook and copy its exact, current versions and
> `LoraConfig` — it is the ground truth and supersedes my pins here.** My versions are a
> known-good-shape starting point, not gospel.

### 1.3 Repo layout (matches spec §7)

```
data/
  clinical_text/         # synthetic clinical cases (JSONL) + generator
  rag_knowledge_base/    # curated veterinary literature chunks for ChromaDB
  microscopy/            # manifests, taxonomy, download + prep scripts
  instruction_tuning/    # merged T1+T2 set, split train/val/test
scripts/                 # generators, ETL, download helpers  (currently EMPTY — you create these)
docs/                    # this guide, spec, risks, datasheet
```

> The repo currently contains only `docs/PROJECT_SPEC.md`; `scripts/` and `data/` are empty
> directories. Every script referenced below is one **you** will create at the path given.

---

## 2. Phase 1 — Spark ETL data pipeline  *(timetable weeks 4–5)*

**Goal:** ingest heterogeneous sources → clean/normalise → emit the canonical schemas in spec §4,
with the grouped, stratified splits of §5. Spark is the "Big Data" competency the MSc is marking, so
make the distributed nature genuine (partitioned reads, DataFrame transforms) even if data is small.

### 2.1 Data sources (map each to spec §4)

| Source | Feeds | Notes |
|---|---|---|
| **NIH Malaria Cell Images** | T1 microscopy **pre-training** class `plasmodium_falciparum` | Public (Kaggle/NIH). Proxy only — excluded from vet test set. |
| Veterinary microscopy images (Cherehani + public parasitology sets) | T1 classes 0–6 | Real labelled slides; the scarce, valuable data. |
| **Synthetic clinical cases** (you generate) | T2 clinical text | Must carry `provenance:"synthetic"`, `clinical_use:false`. |
| VetCompass-style records (if obtainable) | T2 real-ish cases | Anonymise. |
| Curated veterinary literature | RAG KB (Phase 2) | **Real** citations only (spec §4.3). |

### 2.2 `scripts/etl_pipeline.py` — Spark skeleton

```python
"""Phase 1 ETL: ingest -> clean -> normalise -> emit canonical schemas.
Run: spark-submit scripts/etl_pipeline.py
"""
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (StructType, StructField, StringType, IntegerType,
                               DoubleType, ArrayType, BooleanType)

spark = (SparkSession.builder
         .appName("chl-vet-etl")
         .config("spark.sql.shuffle.partitions", "16")
         .getOrCreate())

# --- 2.2a Ingest clinical text (many small JSONL -> one DataFrame) ---
clinical = spark.read.json("data/clinical_text/raw/*.jsonl")

# --- 2.2b Cleaning & normalisation (immutable transforms, no in-place mutation) ---
clinical_clean = (clinical
    .withColumn("species", F.lower(F.trim("species")))
    .withColumn("weight_kg", F.col("weight_kg").cast(DoubleType()))
    # enforce the non-negotiable labelling rule (spec §0)
    .withColumn("provenance", F.coalesce("provenance", F.lit("synthetic")))
    .withColumn("clinical_use", F.lit(False))
    .withColumn("requires_expert_validation", F.lit(True))
    # C&S is null except for bacterial/mastitis cases (spec §3) — keep null, do not impute
    .dropDuplicates(["case_id"]))

# --- 2.2c Validate against the canonical schema (fail fast) ---
required = ["case_id","species","diagnosis","presenting_signs","treatment"]
bad = clinical_clean.filter(
    F.array_contains(F.array([F.col(c).isNull() for c in required]), True))
n_bad = bad.count()
assert n_bad == 0, f"{n_bad} records violate canonical schema — inspect before proceeding"

clinical_clean.write.mode("overwrite").json("data/clinical_text/clean")
```

### 2.3 Image manifest via Spark (don't shovel pixels through Spark)

Treat images as **paths + metadata**, not binary columns. Spark builds the manifest; a plain Python
step does the pixel resize.

```python
# scripts/build_image_manifest.py  (Spark builds the label manifest)
img = (spark.read.format("binaryFile")
       .option("pathGlobFilter", "*.png")
       .option("recursiveFileLookup", "true")
       .load("data/microscopy/images/raw"))

manifest = (img
    .withColumn("path", F.col("path"))
    # class = parent directory name (taxonomy dir layout from spec §4.2)
    .withColumn("cls", F.element_at(F.split("path", "/"), -2))
    .select("path", "cls"))
manifest.write.mode("overwrite").parquet("data/microscopy/manifest.parquet")
```

Resize/normalise to 896×896 (MedGemma's input size) in a separate non-Spark step (`Pillow`), writing
to `data/microscopy/images/<split>/<class>/`.

### 2.4 Grouped, stratified splits (spec §5) — **do this correctly, it's a marked point**

The spec demands: stratified by class, **grouped so no `case_id` leaks across splits**, priority-four
diseases in all three splits, and **no synthetic-only disease in test that's absent from train**.

```python
from sklearn.model_selection import StratifiedGroupKFold  # scikit-learn >= 1.0
# Load the merged instruction records to a pandas frame keyed by case_id + class label.
# Use StratifiedGroupKFold with groups=case_id, y=class/diagnosis, to get train/val/test
# WITHOUT any case_id spanning two splits. Naive random split WILL leak and inflate scores.
```

> ⚠️ VERIFY: Confirm every `case_id` that produced both a T1 image record and a T2 text record lands
> in the **same** split. This is the classic leakage trap — see METHODOLOGICAL_RISKS §"grouped-split
> leakage". Write an assertion that `set(train.case_id) ∩ set(test.case_id) == ∅`.

### 2.5 Emit the instruction-tuning set (spec §4.2)

Merge T1 (image → pathogen class) and T2 (clinical text → treatment) into one chat-format JSONL, then
split. Output `data/instruction_tuning/{train,val,test}.jsonl`. Keep the hallucination **probe set**
(spec §6) as a separate file `data/instruction_tuning/probe.jsonl` — cases whose correct answer is
"insufficient evidence / refer to clinician".

---

## 3. Phase 2 — ChromaDB + LangChain RAG  *(weeks 6–7)*

**Goal:** curate real veterinary literature → chunk → embed → store in ChromaDB → retrieve top-k at
inference for the **T2 treatment** task only (T1 image classification does not use RAG).

### 3.1 Build the knowledge base (spec §4.3 — REAL sources only)

The RAG corpus must be curated from **real literature** with citable sources. Fabricated sources
invalidate the entire hallucination-reduction hypothesis. Populate `data/rag_knowledge_base/*.json`
with chunks each carrying a real `source.citation`, `type`, and `url`.

### 3.2 `scripts/build_vector_db.py`

```python
import json, glob
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

# Local, free embedding model (no API cost). Biomedical model improves retrieval.
emb = HuggingFaceEmbeddings(model_name="pritamdeka/S-PubMedBert-MS-MARCO")
# ⚠️ VERIFY: model name/availability; a general 'all-MiniLM-L6-v2' is a safe fallback.

client = chromadb.PersistentClient(path="data/rag_knowledge_base/chroma")
coll = client.get_or_create_collection("vet_kb", metadata={"hnsw:space": "cosine"})

docs, ids, metas = [], [], []
for f in glob.glob("data/rag_knowledge_base/*.json"):
    d = json.load(open(f))
    docs.append(d["text"]); ids.append(d["doc_id"])
    metas.append({"disease": d["disease"], "section": d["section"],
                  "citation": d["source"]["citation"], "url": d["source"]["url"]})

coll.add(documents=docs, ids=ids, metadatas=metas,
         embeddings=emb.embed_documents(docs))
print("chunks indexed:", coll.count())
```

> ⚠️ VERIFY: `chromadb>=0.5` uses `PersistentClient(path=...)`. Older tutorials show
> `Settings(persist_directory=...)` and `client.persist()` — those are removed. Check the installed
> version's quickstart.

### 3.3 Retrieval at inference (LangChain)

```python
def retrieve_context(query: str, k: int = 4) -> list[dict]:
    q_emb = emb.embed_query(query)
    res = coll.query(query_embeddings=[q_emb], n_results=k,
                     include=["documents", "metadatas", "distances"])
    return [{"text": t, "citation": m["citation"], "url": m["url"]}
            for t, m in zip(res["documents"][0], res["metadatas"][0])]

def build_rag_prompt(clinical_notes: str, cs_report: str | None) -> str:
    ctx = retrieve_context(clinical_notes, k=4)
    sources = "\n\n".join(f"[{i+1}] {c['text']}\n(Source: {c['citation']})"
                          for i, c in enumerate(ctx))
    cs = cs_report or "No culture & sensitivity report available."
    return (f"Retrieved veterinary references:\n{sources}\n\n"
            f"Clinical notes: {clinical_notes}\nC&S: {cs}\n\n"
            f"Give a treatment plan grounded ONLY in the references above. "
            f"If evidence is insufficient, say so and recommend clinician referral. "
            f"Cite the reference number for each recommendation.")
```

The "cite the reference number" instruction is what makes the hallucination metric measurable later
(§7): an unsupported claim is one with no valid citation.

---

## 4. Phase 3 — QLoRA fine-tune  *(weeks 8–10)*

**Ground truth: the official `Google-Health/medgemma/notebooks/fine_tune_with_hugging_face.ipynb`.**
Start from it, adapt to your data. Below is the shape, annotated.

### 4.1 Load model in 4-bit

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from kaggle_secrets import UserSecretsClient

HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
MODEL_ID = "google/medgemma-4b-it"   # 4B MULTIMODAL — required for Task 1

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, quantization_config=bnb, torch_dtype=torch.bfloat16,
    device_map="auto", token=HF_TOKEN, attn_implementation="eager")
processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
model.config.use_cache = False
```

> ⚠️ VERIFY the loader class. Gemma-3 multimodal models load via
> `AutoModelForImageTextToText` in recent transformers; some examples use `AutoModelForCausalLM` or a
> model-specific `Gemma3ForConditionalGeneration`. **Check the model card's example code** — using the
> wrong class silently drops the vision pathway. `attn_implementation="eager"` is recommended for
> Gemma-3 stability and is also what Grad-CAM hooks need (§5).

### 4.2 LoRA config — target modules for the Gemma vision+text stack

```python
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",   # see caveats below
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

Decision points, stated honestly:

- **`target_modules="all-linear"`** is the simplest choice and what the official notebook uses. It
  adapts linear layers across attention (`q_proj,k_proj,v_proj,o_proj`) and MLP
  (`gate_proj,up_proj,down_proj`). If you prefer an explicit list, use those seven names for the
  **language tower**.
- **Vision tower (SigLIP): freeze it.** For your microscopy task the vision encoder is already
  medical-pretrained; adapting the *language* head to output your pathogen taxonomy is the cheaper,
  lower-risk win, and freezing the vision tower cuts VRAM and avoids catastrophic forgetting of the
  encoder. `"all-linear"` may pick up vision-tower linears — if VRAM is tight or T1 accuracy is
  unstable, restrict `target_modules` to language-tower module name patterns only.
  > ⚠️ VERIFY the exact module-name prefixes (`language_model.…` vs `vision_tower.…`) by printing
  > `[n for n,_ in model.named_modules()]` on the loaded model, then decide what `all-linear` actually
  > targeted. Do this — it materially changes what you're training.

### 4.3 Data collator — the multimodal trap

The collator must apply the chat template, tokenise text, **and** process images to the 896×896 /
256-token format, masking the prompt tokens from the loss (train only on the assistant answer). T2
records have `image: null` — the collator must handle mixed image/no-image batches.

> ⚠️ VERIFY: Copy the official notebook's `collate_fn` verbatim and adapt field names. Hand-rolling
> the image-token handling is the most common source of silent training failure (loss looks fine, the
> model never actually sees the image). Sanity-check by overfitting 10 image samples and confirming
> T1 accuracy → ~100% on those 10.

### 4.4 Training args — batch/grad-accum arithmetic for 16 GB

```python
from trl import SFTConfig, SFTTrainer

args = SFTConfig(
    output_dir="/kaggle/working/medgemma-vet-lora",
    per_device_train_batch_size=1,      # 16 GB forces this
    gradient_accumulation_steps=8,      # effective batch = 1*8 = 8
    gradient_checkpointing=True,        # REQUIRED to fit
    learning_rate=2e-4, warmup_ratio=0.03, lr_scheduler_type="cosine",
    num_train_epochs=3, bf16=True, optim="paged_adamw_8bit",
    logging_steps=10, save_strategy="steps", save_steps=100,
    save_total_limit=3, report_to="tensorboard",
    max_seq_length=1024, seed=42,
)
```

- **Effective batch size = `per_device_batch_size × grad_accum × num_gpus`.** With 1×8×1 = **8**. If
  you want effective 16, set `grad_accum=16` (costs time, not VRAM).
- `optim="paged_adamw_8bit"` keeps optimizer state small and pages to CPU on spikes — important on
  16 GB.
- `max_seq_length=1024`: your RAG-augmented T2 prompts can be long; measure your token-length
  distribution and set this to ~p95, not blindly higher (each +512 tokens costs activation VRAM).

### 4.5 Surviving Kaggle's 9–12 h session kill — **checkpoint or lose everything**

This is the practical trap the proposal ignores. A full 3-epoch run may exceed one session.

1. `save_steps=100`, `save_total_limit=3` → adapters written to `/kaggle/working` regularly.
2. **Persist checkpoints off the ephemeral session:** at session start, download the latest checkpoint
   from a Kaggle **Dataset** you own; at intervals, push `/kaggle/working/...` back as a new Dataset
   version (or use `output_dir` on a mounted persistent Kaggle Dataset). `/kaggle/working` is wiped
   when the kernel dies.
3. **Resume:** `trainer.train(resume_from_checkpoint="/path/to/last-checkpoint")`.
4. Budget: at ~30 GPU-h/week and (say) 4–6 h per training run, you get **~5–6 runs/week**. You need
   base-vs-finetuned × experiments — plan runs, don't burn quota on trial-and-error (see
   METHODOLOGICAL_RISKS §"Kaggle quota vs 5 runs").

### 4.6 Merge & save

```python
trainer.train()      # or resume_from_checkpoint=...
trainer.save_model("/kaggle/working/medgemma-vet-lora-final")  # adapter only
# For inference/XAI you can keep adapter separate (load base + adapter) or merge:
# merged = model.merge_and_unload()   # ⚠️ VERIFY merge works with a 4-bit base;
#          you may need to reload the base in bf16 (not 4-bit) before merging.
```

---

## 5. Phase 4 — XAI: Grad-CAM & SHAP  *(weeks 11–12)*

This phase has two **genuine technical subtleties** the proposal glosses over. Handle them explicitly
or your XAI chapter is weak.

### 5.1 Grad-CAM on a ViT is NOT the same as Grad-CAM on a CNN

Grad-CAM (Selvaraju et al., 2017) was designed for **CNN feature maps** that have native 2-D spatial
structure. MedGemma's image encoder is a **SigLIP Vision Transformer** — its layer activations are
shape `[batch, num_tokens, channels]` (a sequence of patch tokens, sometimes plus a class token),
**not** `[batch, C, H, W]`. Applying Grad-CAM naïvely produces garbage. Two things must be fixed:

**(a) `reshape_transform`** — convert the token sequence back to a 2-D grid. For 896×896 input with a
14×14 patch grid (verify the grid size for the actual SigLIP config), drop any class token and reshape
`196 tokens → 14×14×C`, then move channels first:

```python
import torch
def reshape_transform(tensor, height=14, width=14):
    # If a class token is present it is index 0; SigLIP often has NO cls token — VERIFY.
    if tensor.size(1) == height * width + 1:
        tensor = tensor[:, 1:, :]
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)   # -> [B, C, H, W] like a CNN feature map
```

> ⚠️ VERIFY the patch grid (14×14 vs 16×16 etc.) and whether SigLIP uses a class token. Print the
> chosen layer's output shape on one image: `height*width` must equal the token count (or token
> count − 1 if there's a cls token). Get this wrong and the heatmap is meaningless.

**(b) Target layer.** For ViTs, `pytorch-grad-cam` recommends the **norm before the last transformer
block's attention**, e.g. `model.blocks[-1].norm1` in a timm ViT. In MedGemma the path is different —
you must locate the equivalent inside the SigLIP tower, e.g. the LayerNorm of the **final encoder
layer**:

```python
from pytorch_grad_cam import GradCAM
# ⚠️ VERIFY the exact attribute path by printing named_modules() of the vision tower.
# Illustrative only:
target_layer = model.vision_tower.vision_model.encoder.layers[-1].layer_norm1
cam = GradCAM(model=<image_classification_head_wrapper>, target_layers=[target_layer],
              reshape_transform=reshape_transform)
```

**The harder subtlety:** Grad-CAM needs a **scalar class score** to backprop from. MedGemma is a
*generative* model — Task 1 outputs the pathogen name as **generated text**, not a softmax over 8
classes. To get a Grad-CAM you must define a differentiable target. Options, honestly ranked:

1. **Wrap the classification as logits (recommended).** Build a thin wrapper whose forward returns the
   logit of the **first generated token** corresponding to the target class name (or a constrained set
   of class-name tokens), and Grad-CAM against that scalar. This is defensible and the closest to a
   true class score.
2. **Attention rollout / attention maps** as an honest alternative if Grad-CAM on the generative head
   proves intractable. Report it as "attention-based localisation" — don't call it Grad-CAM.
3. **Grad-CAM on the final ViT block with `reshape_transform`** against the pooled image embedding's
   similarity to the class text embedding.

> State in your Methods chapter exactly which target you used and why. "Grad-CAM localisation
> accuracy" (§7) is only meaningful if the target is a genuine class signal. This is a **real
> limitation to disclose**, not to hide — see METHODOLOGICAL_RISKS.

### 5.2 SHAP token attribution on a 4B generative model — expensive, so scope it

SHAP for text generation uses `shap.Explainer` with a `TeacherForcing` model + `Text` masker
(PartitionSHAP under the hood). Verified reality: **it is slow and scales badly** with input length
and generation length. TextGen-SHAP exists precisely because vanilla SHAP is impractical on large
generative models with long prompts — and your RAG-augmented T2 prompts are long.

Realistic guidance:

- **Do not** run SHAP over the full test set. Define a **reduced probe set** (e.g. 20–40 T2 cases,
  including hallucination-probe cases) and run SHAP only there. State the sample size as a limitation.
- **Cap generation length** (`max_new_tokens` small, e.g. explain only the drug-name / dose span, not
  a 300-token plan). Attribution is per output token — pick the clinically load-bearing tokens.
- **Truncate/segment the prompt** — attribute over the clinical-notes span, holding retrieved context
  fixed, or vice versa, rather than all tokens at once.

```python
import shap
from transformers import pipeline
# ⚠️ VERIFY current SHAP text API — it changed across 0.4x releases.
gen = pipeline("text-generation", model=model, tokenizer=processor.tokenizer,
               max_new_tokens=24)
explainer = shap.Explainer(gen)          # uses Text masker + TeacherForcing internally
shap_values = explainer([one_prompt])    # ONE prompt; time this before scaling to 20–40
shap.plots.text(shap_values)
```

> ⚠️ VERIFY: SHAP's `Explainer(pipeline)` path for decoder-only generative models is fiddly and
> version-sensitive. Time a **single** explanation first. If it takes minutes, budget accordingly and
> keep the probe set small. Consider `shap.PartitionExplainer` with a token-cluster masker as the
> documented text path. Alternatives if SHAP is intractable: **LIME for text**, or **integrated
> gradients** over the embedding layer (`captum`) — cheaper, defensible, disclose the swap.

---

## 6. Phase 5 — Clinician trust evaluation  *(weeks 15–16)*

**Goal:** Cherehani Labs vets review base-vs-finetuned outputs, **with and without XAI**, and complete
a structured trust survey (proposal cites **Hoffman et al. 2018, "Metrics for Explainable AI"**).
Thematic analysis (Braun & Clarke 2006) on free-text.

### 6.1 Ethics & anonymisation (do the paperwork EARLY — weeks 4–6, not week 15)

- Apply for **UEL ethics approval** for human-participant research (the clinicians are participants).
  Approval can take weeks; starting late blocks Phase 5 entirely.
- **Informed consent** form: purpose, voluntary, withdrawal right, data handling, no clinical use.
- **Anonymise all clinical records** used as stimuli: strip owner names, exact GPS, farm IDs, any
  re-identifying detail (spec §0 already flags research-only use).
- Participant responses stored anonymised (participant codes, not names).

### 6.2 Survey instrument (Hoffman et al. 2018 constructs)

Design around Hoffman's XAI-trust dimensions. A defensible instrument:

- **Explanation Satisfaction Scale** (Hoffman) — Likert 1–5, e.g. "The explanation helped me
  understand how the AI reached its conclusion"; "…lets me judge when to trust it"; "…is
  sufficiently detailed"; "…is complete."
- **Trust in Automation** items — "I would rely on this system's recommendation," etc.
- **Task-based measure (behavioural, not just self-report):** show the vet an AI output where the model
  is **wrong**; does the XAI explanation help them **catch the error**? Report error-detection rate
  with vs without XAI — this is stronger evidence than Likert alone (Reyes et al. 2020 used this).
- **Within-subjects, counterbalanced:** each clinician sees {base, finetuned} × {no-XAI, XAI}, order
  randomised to control learning effects.
- Free-text: "What would make you trust/distrust this recommendation?" → thematic analysis.

### 6.3 The uncomfortable statistics reality (flag it now)

Cherehani Labs has **few clinicians** (likely n = 3–8). With that n, **quantitative trust claims are
underpowered** — you cannot credibly claim a statistically significant Likert difference. Plan for:

- **Descriptive stats + qualitative thematic depth** as the primary contribution, not p-values.
- If you report any test, use non-parametric (Wilcoxon signed-rank for within-subject) and **report
  effect sizes + CIs, framed as exploratory**. State the power limitation explicitly (see
  METHODOLOGICAL_RISKS §"clinician n"). Examiners reward candour here.

---

## 7. Evaluation harness — computing every metric in proposal §3.3

Build `scripts/evaluate.py`. Run **base MedGemma** and **QLoRA-finetuned MedGemma** through the
identical harness on the **test split**; the deliverable is the paired comparison.

### 7.1 Task 1 (microscopy classification)

- **Accuracy** = correct pathogen class / total. Parse the generated class name to the taxonomy
  (spec §2); use fuzzy/canonical matching so "T. parva" ↔ `theileria_parva`.
- **Macro-F1** = unweighted mean of per-class F1 (`sklearn.metrics.f1_score(..., average="macro")`).
  Macro (not micro) because classes are imbalanced and rare pathogens matter clinically.
- **Grad-CAM localisation accuracy** — needs ground-truth parasite locations. Options:
  (a) if you have bounding boxes/masks on a subset, compute the fraction of Grad-CAM mass inside the
  GT region (IoU or "pointing game": does the CAM peak fall on a parasite?); (b) if no boxes, have a
  clinician rate a sample of heatmaps as "localises the parasite: yes/no" and report the rate.
  **State which** — they are not the same metric.

### 7.2 Task 2 (treatment recommendation)

- **ROUGE-L** vs the reference treatment plan (`rouge-score`, report F-measure). Caveat: ROUGE rewards
  surface overlap, not clinical correctness — report it but don't over-claim.
- **Factual-consistency score** — define operationally: fraction of checkable claims (drug, dose,
  route, withdrawal period) that match the reference/KB. Implement as structured-field extraction from
  the generated plan compared to the case record's `treatment` object (spec §4.1). E.g. correct drug
  AND dose-within-tolerance AND route → 1, else partial credit per field.
- **Hallucination rate — the proposal does NOT define this; you must.** Proposed operational
  definition (defensible, disclose it):

  > **Hallucination rate = (number of generated clinical claims that are either (i) unsupported by any
  > retrieved RAG source / case evidence, or (ii) factually contradicted by the reference) ÷ total
  > generated clinical claims.** A "claim" = an atomic assertion about drug, dose, route, frequency,
  > duration, or diagnosis. Measured per output, averaged over the test set.

  Complement it with the **hallucination probe set** (spec §6): cases whose correct answer is
  "insufficient evidence / refer to clinician." **Over-confident fabrication rate** = fraction of
  probe cases where the model gives a confident specific recommendation instead of deferring. This
  directly tests the RAG hypothesis. Report both numbers.

  Scoring can be (a) rule-based against structured fields + citation presence (§3.3's "cite the
  reference number" makes unsupported = uncited), and/or (b) an LLM-as-judge with a clinician-audited
  subset for validity. State the method and its limits.

### 7.3 Reporting

One table: rows = metrics, columns = {base, finetuned, Δ}. Split T1/T2. Report the XAI results
(localisation accuracy, SHAP probe findings) and the Phase-5 trust results alongside.

---

## 8. Reproducibility

- **Seeds everywhere:** `seed=42` in `SFTConfig`; also
  `transformers.set_seed(42)`, `torch.manual_seed(42)`, `np.random.seed(42)`,
  `random.seed(42)`. Note GPU non-determinism (some CUDA kernels) — document residual variance.
- **Pin the base model revision:** record the exact HF commit hash of `medgemma-4b-it` you used
  (`revision=` in `from_pretrained`). Model cards get updated.
- **Environment capture:** `pip freeze > docs/env-kaggle-<date>.txt` at the end of each training run;
  commit it. Record CUDA/torch versions and GPU type (P100 vs T4).
- **Experiment tracking:** TensorBoard (already in `report_to`) for loss curves; keep a
  `docs/experiments.md` log — one row per run: model, data version, hyperparams, quota spent, result.
  Optionally Weights & Biases (free tier) for richer tracking.
- **Data versioning:** tag the `data/instruction_tuning/` split with a version + the split seed so the
  base-vs-finetuned comparison is provably on the same test set. Commit the split assertion output.

---

## 9. Risk register mapped to the 24-week timetable

| Weeks (proposal) | Activity | Realistic? | Risk / what actually happens | Mitigation |
|---|---|---|---|---|
| 1–3 | Lit review, proposal | OK | — | Do HF gating + **ethics application** now (both have lead time). |
| 4–5 | Spark ETL, data collection | **Optimistic** | Sourcing real vet microscopy images + curating a real RAG corpus is slow; grouped-split leakage bugs. | Start data sourcing in week 1. Reuse public parasitology datasets. Write the split-leakage assertion first. |
| 6–7 | ChromaDB + RAG | Tight but OK | ChromaDB/LangChain API churn eats time. | Pin versions; follow §3 caveats; use Context7 for current APIs. |
| 8–10 | QLoRA fine-tune | **Very optimistic** | Multimodal collator bugs, OOM, **Kaggle 9 h session kills**, 30 h/week quota vs many runs. | §4.5 checkpoint-to-Dataset + resume. Overfit-10 sanity check. Budget runs; don't trial-and-error on quota. Expect to spill into weeks 11–12. |
| 11–12 | Grad-CAM + SHAP | **Very optimistic** | Grad-CAM-on-ViT + generative-target subtleties (§5.1); SHAP cost on 4B (§5.2). Each can consume a week alone. | Decide Grad-CAM target early; prepare attention-rollout fallback. SHAP on a **reduced probe set** only. |
| 13–14 | Base vs finetuned comparison | OK if Phase 3–4 land | Slips if training slipped. | Build `evaluate.py` **during** Phase 3 so it's ready. |
| 15–16 | Clinician survey | **Blocked without early ethics** | Ethics not approved; too few clinicians; scheduling. | Ethics in weeks 4–6. Accept small-n → qualitative-primary framing (§6.3). |
| 17–20 | Dissertation writing | OK | — | Write Methods as you go, not at the end. |
| 21–24 | Revision, submission | OK | — | Leave real buffer; earlier phases will have eaten into this. |

**Weeks most likely to blow up: 8–10 (training) and 11–12 (XAI).** Treat the timetable as
best-case; the realistic critical path pushes XAI into weeks 12–14. See
[`METHODOLOGICAL_RISKS.md`](./METHODOLOGICAL_RISKS.md) for the deeper technical/statistical risks and
how to turn each into a Limitations-chapter strength.

---

## Sources (verified during authoring)

- [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) — architecture, SigLIP,
  896×896 / 256 image tokens, 128K context, HAI-DEF gating, transformers ≥ 4.50.
- [google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it) — text-only (cannot do Task 1).
- [MedGemma release collection](https://huggingface.co/collections/google/medgemma-release) & [Google-Health/medgemma](https://github.com/Google-Health/medgemma) — variants + official fine-tuning notebook.
- [MedGemma model card, HAI-DEF](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card) — license/access terms.
- [pytorch-grad-cam ViT tutorial](https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md) & [docs](https://jacobgil.github.io/pytorch-gradcam-book/vision_transformers.html) — reshape_transform, target-layer choice.
- [SHAP TeacherForcing](https://shap.readthedocs.io/en/latest/generated/shap.models.TeacherForcing.html) & [Multi-Level Explanations for Generative LMs](https://arxiv.org/pdf/2403.14459) — PartitionSHAP cost on generative models.
- [Kaggle GPU limits](https://www.kaggle.com/docs/efficient-gpu-usage) — ~30 GPU-h/week, 16 GB P100 / 2×T4, session cap.
</content>
</invoke>
