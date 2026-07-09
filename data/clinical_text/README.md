# Synthetic Clinical-Text Dataset (T2 — Treatment Recommendation)

> ## ⚠️ SYNTHETIC PRE-VALIDATION DATA — NOT VETERINARY ADVICE
> Every record here is **machine-generated** and carries
> `provenance="synthetic"`, `clinical_use=false`, `requires_expert_validation=true`.
> This is **research training data only** for fine-tuning MedGemma. It is **NOT clinical
> guidance** and must never be used to treat a real animal. Per the project proposal and
> `docs/PROJECT_SPEC.md §0`, these cases are **pre-validation drafts that require sign-off by
> Cherehani Labs clinical staff** before any downstream use. Treatment protocols reflect
> well-established, widely documented veterinary pharmacology but have **not** been clinically
> reviewed here.

This directory contains the synthetic clinical cases used for the **T2 treatment-recommendation
task** (spec §1). Records conform exactly to **schema §4.1** of `docs/PROJECT_SPEC.md`.

---

## Files

| File | Records | Purpose |
|---|---:|---|
| `cases_train.jsonl` | 700 | Training split (0.70) |
| `cases_val.jsonl` | 151 | Validation split (0.15) |
| `cases_test.jsonl` | 149 | Test split (0.15) |
| `hallucination_probe.jsonl` | 60 | Held-out probe: correct answer is **referral**, not a drug (spec §6) |

Splits are **stratified by diagnosis** with **no `case_id` leakage** across splits. The
priority-four diseases appear in all three splits, and no diagnosis appears in `test`
without also appearing in `train` (spec §5). Probe `case_id`s use the `CHL-SYN-P#####`
namespace and are disjoint from the main `CHL-SYN-######` ids.

---

## Reproducing

Only the Python standard library is required (deterministic, seeded).

```bash
# main clinical set (train/val/test)
python3 scripts/generate_clinical_cases.py --n 1000 --seed 42 --out data/clinical_text

# hallucination probe set
python3 scripts/generate_clinical_cases.py --hallucination-probe --probe-n 60 \
    --seed 42 --out data/clinical_text

# validate everything
python3 scripts/validate_clinical_cases.py --dir data/clinical_text
```

`--seed 42` reproduces the exact dataset described here.

---

## Generator design

`scripts/generate_clinical_cases.py` is driven by a single `DISEASE_PROFILES` data
structure. For each disease it declares: plausible species, presenting signs with sampling
probabilities, sick-state vital-sign ranges, the expected microscopy class (spec §2 taxonomy),
whether culture & sensitivity applies, differentials, and the treatment protocol
(drug, mg/kg dose range, route, frequency, duration, withdrawal period).

Key sampling rules:

- **Breed** is species-appropriate and Kenyan/East-African (Boran, Sahiwal, East African
  Zebu, Friesian/Ayrshire/Jersey crosses, Nganda for cattle; Small East African, Galla,
  Toggenburg/Saanen crosses for goats; Red Maasai, Dorper, Blackhead Persian for sheep; etc.).
- **County** is a real Kenyan county.
- **Weight is derived jointly from species, breed and age** (never sampled independently): a
  growth curve interpolates between species birth weight and a breed-specific adult weight by
  a maturity fraction, so a neonate never carries an adult's weight. The validator re-checks
  this with a shared plausibility rule.
- **`clinician_rationale`** is generated from the sampled facts via **five template variants**
  (plus three probe-specific variants), so the natural-language text is not a single repeated
  string.
- Every record is **validated against schema §4.1 before it is written**; any violation raises
  `SchemaError` and aborts (fail-loud).

### Culture & sensitivity (`culture_sensitivity`)

Populated **only for Bovine Mastitis** — the one disease for which C&S is routinely available
(spec §3) — and **`null` for every other case**. This `null` is a **first-class scenario**
the model must learn to handle (spec §3), not a data defect. Mastitis reports include a
realistic isolate (Staph aureus, Strep agalactiae/uberis/dysgalactiae, E. coli, Klebsiella,
CNS), a somatic cell count, and an antibiotic panel with **S / I / R** results. The
recommended systemic antibiotic is chosen to be one the isolate tests **sensitive** to, with
gram-negative isolates made resistant to the narrow-spectrum beta-lactams.

### Hallucination probe (`--hallucination-probe`)

Emits ambiguous / insufficient-evidence cases: sparse non-specific signs, near-normal vitals,
`null` microscopy and `null` C&S. The **correct `treatment` is a referral**
(`REFERRAL - insufficient evidence …`, `dose_mg_per_kg=null`) rather than a drug, and the
diagnosis is `Inconclusive - insufficient evidence`. This set measures over-confident
fabrication (spec §6).

---

## Priority weighting (explicit constant)

The four proposal-priority diseases (**East Coast Fever, Trypanosomiasis, Peste des Petits
Ruminants, Bovine Mastitis**) are over-represented relative to the long tail via documented
sampling-weight constants in the generator:

| Constant | Value | Applies to |
|---|---:|---|
| `PRIORITY_WEIGHT` | 5.0 | each of the priority-four |
| `COMMON_TAIL_WEIGHT` | 1.5 | Anaplasmosis, Babesiosis, Lumpy Skin Disease, CCPP, Coccidiosis, Haemonchosis |
| `RARE_TAIL_WEIGHT` | 1.0 | Canine Babesiosis/Trypanosomiasis, Ehrlichiosis, Canine Parvovirus, Feline URI, Haemoplasmosis |

At `--n 1000 --seed 42` the priority-four make up **55.3%** of the main set.

---

## Disease profiles & treatment protocols

Withdrawal periods are **meat** withdrawal (days) for food species and `0` for companion
animals. Protocols are grounded in well-established veterinary pharmacology.

| Diagnosis | Species | Microscopy class | C&S | Primary treatment | Dose | Route |
|---|---|---|:--:|---|---|---|
| **East Coast Fever** ★ | cattle | `theileria_parva` | – | Buparvaquone | 2.5 mg/kg | IM |
| **Trypanosomiasis** ★ | cattle | `trypanosoma_spp` | – | Diminazene aceturate | 3.5 mg/kg | deep IM |
| **Peste des Petits Ruminants** ★ | goat/sheep | – | – | **Supportive care (viral — no antiviral)** + secondary-infection control | – | – |
| **Bovine Mastitis** ★ | cattle | – | ✔ | Systemic antibiotic **guided by C&S** (+ stripping, NSAID) | e.g. 12.5 mg/kg | IM |
| Anaplasmosis | cattle | `anaplasma_spp` | – | Oxytetracycline (long-acting) | 20 mg/kg | IM |
| Babesiosis | cattle | `babesia_spp` | – | Diminazene aceturate | 3.5 mg/kg | deep IM |
| Lumpy Skin Disease | cattle | – | – | **Supportive care (viral — no antiviral)** + secondary-infection control | – | – |
| Contagious Caprine Pleuropneumonia | goat | – | – | Tylosin (oxytet alternative) | 10 mg/kg | IM |
| Coccidiosis | goat/sheep | `eimeria_spp` | – | Amprolium (toltrazuril alternative) | 10 mg/kg | PO |
| Haemonchosis | goat/sheep | `haemonchus_contortus` | – | Albendazole | 7.5 mg/kg | PO |
| Canine Babesiosis | dog | `babesia_spp` | – | Imidocarb dipropionate | 6.6 mg/kg | SC |
| Canine Trypanosomiasis | dog | `trypanosoma_spp` | – | Diminazene aceturate | 3.5–7 mg/kg | IM |
| Ehrlichiosis | dog | – | – | Doxycycline | 10 mg/kg | PO |
| Canine Parvovirus | dog | – | – | **Supportive care (viral — no antiviral)** | – | IV |
| Feline Upper Respiratory Complex | cat | – | – | Doxycycline (secondary/atypical cover) | 10 mg/kg | PO |
| Haemoplasmosis | cat | – | – | Doxycycline | 10 mg/kg | PO |

★ = proposal priority-four (over-represented).

**Clinically load-bearing decisions:**

- **PPR, Lumpy Skin Disease and Canine Parvovirus are VIRAL.** Their `primary_drug` is
  `"Supportive care (no specific antiviral)"` with `dose_mg_per_kg = null`; antibiotics appear
  only in `supportive_care` to control **secondary** bacterial infection. A model trained to
  prescribe an "antiviral for PPR" would be clinically wrong — this dataset deliberately avoids
  that. For PPR/LSD, `withdrawal_period_days` reflects the long-acting oxytetracycline given for
  the secondary-infection component.
- **East Coast Fever → Buparvaquone**, the standard theilericidal.
- **Trypanosomiasis / bovine Babesiosis → Diminazene aceturate**; Isometamidium is the common
  prophylactic/curative alternative (noted, not modelled as a separate record).
- **Anaplasmosis → oxytetracycline** (tetracyclines are the established choice).
- **Ehrlichiosis / feline haemoplasmosis → doxycycline** (drug of choice).

---

## Actual class distribution (`--n 1000 --seed 42`)

| Diagnosis | train | val | test | total |
|---|---:|---:|---:|---:|
| Trypanosomiasis ★ | 104 | 22 | 23 | 149 |
| Bovine Mastitis ★ | 101 | 22 | 21 | 144 |
| East Coast Fever ★ | 91 | 20 | 19 | 130 |
| Peste des Petits Ruminants ★ | 91 | 20 | 19 | 130 |
| Babesiosis | 34 | 7 | 8 | 49 |
| Lumpy Skin Disease | 33 | 7 | 7 | 47 |
| Anaplasmosis | 30 | 6 | 7 | 43 |
| Coccidiosis | 30 | 6 | 7 | 43 |
| Haemonchosis | 28 | 6 | 6 | 40 |
| Contagious Caprine Pleuropneumonia | 27 | 6 | 6 | 39 |
| Canine Trypanosomiasis | 27 | 6 | 5 | 38 |
| Feline Upper Respiratory Complex | 24 | 5 | 5 | 34 |
| Haemoplasmosis | 23 | 5 | 5 | 33 |
| Canine Babesiosis | 22 | 5 | 5 | 32 |
| Canine Parvovirus | 18 | 4 | 3 | 25 |
| Ehrlichiosis | 17 | 4 | 3 | 24 |
| **TOTAL** | **700** | **151** | **149** | **1000** |

Priority-four share: **55.3%**. Species: cattle 562, goat 150, dog 119, sheep 102, cat 67.
Hallucination probe: 60 records (all referrals).

---

## Schema (§4.1) recap

Each line is a JSON object with: `case_id`, `provenance`, `clinical_use`,
`requires_expert_validation`, `species`, `breed`, `age_months`, `weight_kg`, `sex`, `county`,
`presenting_signs[]`, `vitals{temp_c,hr_bpm,rr_bpm}`, `microscopy_finding{class,parasitaemia_pct}|null`,
`culture_sensitivity{…}|null`, `diagnosis`, `differentials[]`,
`treatment{primary_drug,dose_mg_per_kg,route,frequency,duration_days,supportive_care[]}`,
`follow_up_days`, `withdrawal_period_days`, `clinician_rationale`.

---

## References (generic — no fabricated citations)

Protocols follow the drug/dose/route conventions documented in standard veterinary references
such as the *MSD/Merck Veterinary Manual*, the *WOAH (OIE) Terrestrial Manual* (for notifiable
diseases including PPR, LSD and CCPP), and standard veterinary pharmacology/therapeutics texts
(e.g. Plumb's Veterinary Drug Handbook). No specific page-level citations are asserted for
individual synthetic records; the curated, citable grounding corpus lives separately in
`data/rag_knowledge_base/` (spec §4.3). Any protocol here must be confirmed against current
Kenyan/regional guidance and Cherehani Labs clinical judgement before use.
