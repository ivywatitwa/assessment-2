# Bounded XAI and T2 Results

## T2 Synthetic Generation

The run evaluated four synthetic validation cases and eight synthetic hallucination-probe cases. These records are auxiliary data with `clinical_use=false`.

| Result | Synthetic validation | Hallucination probe |
|---|---:|---:|
| Items | 4 | 8 |
| Diagnosis exact match | 0.0 | 0.0 |
| Drug exact match | 0.0 | 0.0 |
| Referral rate | Not applicable | 0.0 |

The raw outputs repeatedly generated fragments such as `trypanosome_present` and `model`. This is expected from applying a T1-trained adapter to T2 prompts and is evidence that the current adapter is not a T2 treatment model. It must not be described as a clinical recommendation result.

## T1 Image Attribution

Four balanced validation images were analysed with 4 by 4 image-tile occlusion. Two positive records had publisher boxes. The positive class margin was strongly positive for all four records, including negative controls. All 16 tile deltas for each of the two *box-annotated positive* images were negative, so positive-attribution mass inside boxes is **undefined** (`null`) for both. The original aggregate XAI report's mean of 0.0 and count of two “records with boxes” were incorrect for box-localisation: its two non-null zeros came from the *negative-control images without boxes*. The recomputed `data/secondary/reports/xai_corrected_summary.json` reports two box-bearing records, zero defined box-mass scores and a null mean and pointing-game score. The earlier coarse pointing-game score of 1.0 was based on negative evidence changes and cannot be interpreted as successful localisation.

This supports the main T1 finding that the adapter has a strong positive-class prior and does not demonstrate reliable image-grounded classification.

## T2 SHAP Attribution

Four synthetic validation cases received field-level KernelSHAP values for species, breed, age, weight, sex, location, presenting signs, vitals, microscopy and culture/sensitivity. The raw rows identify the method as `field_shap`; the original aggregate report incorrectly labelled them `field_occlusion`. The corrected summary preserves `field_shap`. These values attribute changes in a teacher-forced *reference* diagnosis score for synthetic prompts, not the model's generated treatment output. They do not establish factuality, clinical safety, or clinician trust.

## Scope

The XAI implementation is executable and produces saved attribution artifacts. T1 uses image occlusion as a defensible fallback rather than Grad-CAM because a verified scalar vision target and vision-token mapping were not established. T2 uses field-level KernelSHAP on synthetic auxiliary cases. Faithfulness and stability should be expanded before treating the explanations as general model behaviour.
