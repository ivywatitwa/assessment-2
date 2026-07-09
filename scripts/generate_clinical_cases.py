#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_clinical_cases.py
==========================

Deterministic, seeded generator of **SYNTHETIC** veterinary clinical-text cases for the
MSc dissertation project *"Fine-Tuning MedGemma for Multimodal Explainable Veterinary
Diagnostics"* at Cherehani Labs, Kenya.

The records conform exactly to schema §4.1 of docs/PROJECT_SPEC.md and are driven by the
``DISEASE_PROFILES`` data structure below (species, presenting signs with sampling
probabilities, vital-sign ranges, expected microscopy class, whether culture & sensitivity
applies, differentials, and a pharmacologically grounded treatment protocol).

-------------------------------------------------------------------------------------------
   !!!  SYNTHETIC PRE-VALIDATION DATA — NOT VETERINARY ADVICE / NOT CLINICAL GUIDANCE  !!!
-------------------------------------------------------------------------------------------
Every emitted record carries ``provenance="synthetic"``, ``clinical_use=false`` and
``requires_expert_validation=true`` (spec §0). These are research training drafts that
REQUIRE sign-off by Cherehani Labs clinical staff before any use. Treatment protocols are
grounded in well-established, widely documented veterinary pharmacology, but no record here
may be relied upon for treating a real animal.

Usage
-----
    # Main clinical set -> cases_train / cases_val / cases_test (stratified 70/15/15)
    python3 scripts/generate_clinical_cases.py --n 1000 --seed 42 \
        --out data/clinical_text

    # Hallucination probe set -> hallucination_probe.jsonl (referral = correct answer)
    python3 scripts/generate_clinical_cases.py --hallucination-probe --probe-n 60 \
        --seed 42 --out data/clinical_text

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

# ==========================================================================================
# Constants / labelling
# ==========================================================================================

PROVENANCE = {
    "provenance": "synthetic",
    "clinical_use": False,
    "requires_expert_validation": True,
}

CASE_ID_PREFIX = "CHL-SYN-"
PROBE_ID_PREFIX = "CHL-SYN-P"

# Microscopy pathogen taxonomy (spec §2). plasmodium_falciparum is a pre-training proxy and
# is deliberately never assigned to a veterinary clinical case here.
MICROSCOPY_CLASSES = {
    "uninfected",
    "trypanosoma_spp",
    "theileria_parva",
    "babesia_spp",
    "anaplasma_spp",
    "eimeria_spp",
    "haemonchus_contortus",
    "plasmodium_falciparum",
}

# ------------------------------------------------------------------------------------------
# Priority weighting (spec §3). The four proposal-priority diseases are OVER-REPRESENTED
# relative to the long tail. This is an explicit, documented constant.
# ------------------------------------------------------------------------------------------
PRIORITY_FOUR = [
    "East Coast Fever",
    "Trypanosomiasis",
    "Peste des Petits Ruminants",
    "Bovine Mastitis",
]
PRIORITY_WEIGHT = 5.0          # sampling weight for each of the priority-four diseases
COMMON_TAIL_WEIGHT = 1.5       # other frequently seen cattle/small-ruminant diseases
RARE_TAIL_WEIGHT = 1.0         # companion-animal / less common long tail

# Real Kenyan counties (subset spanning the main livestock-keeping regions).
KENYAN_COUNTIES = [
    "Narok", "Kajiado", "Nakuru", "Machakos", "Kiambu", "Nyeri", "Meru", "Kilifi",
    "Kwale", "Turkana", "Marsabit", "Garissa", "Wajir", "Isiolo", "Baringo", "Laikipia",
    "Uasin Gishu", "Trans Nzoia", "Bomet", "Kericho", "Nandi", "Bungoma", "Kakamega",
    "Siaya", "Kisumu", "Homa Bay", "Migori", "Taita Taveta", "Makueni", "Kitui", "Embu",
    "Tharaka Nithi", "Samburu", "West Pokot", "Elgeyo Marakwet", "Nyandarua", "Murang'a",
    "Kirinyaga", "Vihiga", "Busia", "Nyamira", "Kisii", "Tana River", "Lamu", "Mandera",
]

# Species-appropriate Kenyan / East-African breeds.
SPECIES_BREEDS = {
    "cattle": ["Boran", "Sahiwal", "East African Zebu", "Friesian cross",
               "Ayrshire cross", "Jersey cross", "Nganda"],
    "goat": ["Small East African", "Galla", "Toggenburg cross", "Saanen cross"],
    "sheep": ["Red Maasai", "Dorper", "Blackhead Persian", "Merino cross"],
    "dog": ["Local mixed breed (dog)", "German Shepherd", "Rottweiler",
            "Labrador Retriever", "Boerboel"],
    "cat": ["Domestic Shorthair", "Local mixed breed (cat)"],
}

# Weight model constants. weight_kg is derived jointly from species, breed and age so it is
# plausible for the animal's life stage (a neonate never carries an adult's weight).
SPECIES_BIRTH_KG = {"cattle": 28.0, "goat": 3.0, "sheep": 3.5, "dog": 0.4, "cat": 0.1}
SPECIES_MATURITY_MONTHS = {"cattle": 42, "goat": 18, "sheep": 18, "dog": 18, "cat": 12}
# Aggregated adult weight envelope per species (used by the standalone validator too).
SPECIES_ADULT_KG = {
    "cattle": (250.0, 600.0),
    "goat": (25.0, 75.0),
    "sheep": (30.0, 80.0),
    "dog": (10.0, 85.0),
    "cat": (2.5, 5.5),
}
# Per-breed adult weight range (kg), used by the generator for a realistic mature weight.
BREED_ADULT_KG = {
    "Boran": (350, 500), "Sahiwal": (400, 550), "East African Zebu": (250, 350),
    "Friesian cross": (450, 600), "Ayrshire cross": (400, 550), "Jersey cross": (300, 400),
    "Nganda": (250, 350),
    "Small East African": (25, 40), "Galla": (45, 70), "Toggenburg cross": (45, 65),
    "Saanen cross": (50, 75),
    "Red Maasai": (30, 45), "Dorper": (50, 80), "Blackhead Persian": (30, 45),
    "Merino cross": (40, 60),
    "Local mixed breed (dog)": (12, 25), "German Shepherd": (25, 40),
    "Rottweiler": (35, 55), "Labrador Retriever": (25, 36), "Boerboel": (50, 80),
    "Domestic Shorthair": (3.0, 5.0), "Local mixed breed (cat)": (2.5, 4.5),
}

# Default age sampling window per species (months).
SPECIES_AGE_RANGE = {
    "cattle": (2, 132), "goat": (2, 84), "sheep": (2, 84),
    "dog": (2, 156), "cat": (2, 180),
}

VALID_SEXES = ("male", "female")
VALID_ROUTES = ("IM", "deep IM", "IV", "SC", "PO", "intramammary + systemic IM", "N/A")

# ==========================================================================================
# Disease profiles  (the data structure that drives the generator)
# ==========================================================================================
# Each profile field:
#   species          : allowed species for this diagnosis
#   weight           : sampling weight (priority vs long tail)
#   sex              : optional fixed sex ("female" for lactating-cow mastitis)
#   age_range        : optional override of the species age window
#   signs            : list of (sign, probability) -- probability of appearing in a case
#   vitals           : {temp_c/hr_bpm/rr_bpm: (low, high)} reflecting the sick-state ranges
#   microscopy       : (class, (para_low, para_high)) or None if no haemo/enteric parasite
#   cs_applies       : whether a culture & sensitivity report is populated (mastitis only)
#   differentials    : plausible differential diagnoses
#   treatment        : protocol (see build_treatment)
#   follow_up_days   : recommended re-check interval
#   note             : one-line clinical clause used inside the rationale
#
# TREATMENT PROTOCOL SOURCING (well-established, widely documented veterinary pharmacology;
# no fabricated citations -- see README for the generic references used):
#   * ECF (Theileria parva)  -> Buparvaquone 2.5 mg/kg IM.
#   * Trypanosomiasis/Babesiosis (cattle) -> Diminazene aceturate 3.5 mg/kg deep IM.
#   * Anaplasmosis           -> Oxytetracycline (long-acting) 20 mg/kg IM.
#   * Bovine Mastitis        -> systemic antibiotic GUIDED BY C&S + supportive care.
#   * PPR / Lumpy Skin Disease / Canine Parvovirus are VIRAL: SUPPORTIVE CARE ONLY,
#     with secondary bacterial control -- explicitly NO antiviral "cure".
#   * CCPP (Mycoplasma)      -> Tylosin 10 mg/kg IM (oxytetracycline alternative).
#   * Coccidiosis            -> Amprolium 10 mg/kg PO (toltrazuril alternative).
#   * Haemonchosis           -> Albendazole 7.5 mg/kg PO (+ refugia/resistance note).
#   * Canine Babesiosis      -> Imidocarb dipropionate 6.6 mg/kg SC.
#   * Ehrlichiosis / Feline haemoplasmosis / Feline URI -> Doxycycline 10 mg/kg PO.
# Withdrawal periods are meat-withdrawal figures for food species (0 for companion animals).
# ==========================================================================================

DISEASE_PROFILES: Dict[str, Dict[str, Any]] = {
    # ------------------------------- PRIORITY FOUR ---------------------------------------
    "East Coast Fever": {
        "species": ["cattle"], "weight": PRIORITY_WEIGHT,
        "signs": [("pyrexia", 0.95), ("lymphadenopathy", 0.9), ("dyspnoea", 0.6),
                  ("nasal discharge", 0.5), ("inappetence", 0.7), ("corneal opacity", 0.3),
                  ("weakness", 0.6), ("lacrimation", 0.4), ("frothy nasal froth", 0.25)],
        "vitals": {"temp_c": (39.8, 41.6), "hr_bpm": (72, 108), "rr_bpm": (30, 60)},
        "microscopy": ("theileria_parva", (0.5, 8.0)),
        "cs_applies": False,
        "differentials": ["Trypanosomiasis", "Anaplasmosis", "Babesiosis"],
        "treatment": {
            "primary_drug": "Buparvaquone", "dose": (2.5, 2.5), "route": "IM",
            "frequency": "single dose (repeat after 48-72 h if response is poor)",
            "duration_days": 1, "withdrawal_days": 42,
            "supportive": ["NSAID (flunixin meglumine) for pyrexia", "fluid therapy",
                           "supportive nursing and soft palatable feed"]},
        "follow_up_days": 7,
        "note": "prescapular/parotid lymph-node enlargement with Theileria parva schizonts "
                "on lymph-node smear is characteristic of East Coast Fever",
    },
    "Trypanosomiasis": {
        "species": ["cattle"], "weight": PRIORITY_WEIGHT,
        "signs": [("intermittent pyrexia", 0.85), ("pale mucous membranes", 0.9),
                  ("progressive weight loss", 0.8), ("lethargy", 0.7),
                  ("lymphadenopathy", 0.5), ("lacrimation", 0.35),
                  ("dependent oedema", 0.3), ("rough coat", 0.5)],
        "vitals": {"temp_c": (39.4, 41.0), "hr_bpm": (70, 100), "rr_bpm": (26, 48)},
        "microscopy": ("trypanosoma_spp", (0.1, 5.0)),
        "cs_applies": False,
        "differentials": ["East Coast Fever", "Anaplasmosis", "Babesiosis",
                          "Nutritional deficiency"],
        "treatment": {
            "primary_drug": "Diminazene aceturate", "dose": (3.5, 3.5), "route": "deep IM",
            "frequency": "single dose", "duration_days": 1, "withdrawal_days": 21,
            "supportive": ["haematinics and supportive care for anaemia", "fluid therapy",
                           "farm-level tsetse / vector control"]},
        "follow_up_days": 14,
        "note": "anaemia with trypanosomes demonstrable on a wet blood film / buffy-coat "
                "supports a diagnosis of Nagana (trypanosomiasis)",
    },
    "Peste des Petits Ruminants": {
        "species": ["goat", "sheep"], "weight": PRIORITY_WEIGHT,
        "signs": [("high pyrexia", 0.95), ("mucopurulent oculonasal discharge", 0.9),
                  ("oral erosions / stomatitis", 0.75), ("profuse diarrhoea", 0.8),
                  ("dyspnoea / pneumonia", 0.6), ("coughing", 0.5), ("dehydration", 0.7),
                  ("inappetence", 0.7)],
        "vitals": {"temp_c": (40.0, 41.8), "hr_bpm": (80, 120), "rr_bpm": (30, 60)},
        "microscopy": None,
        "cs_applies": False,
        "differentials": ["Contagious Caprine Pleuropneumonia", "Coccidiosis",
                          "Pasteurellosis", "Foot-and-mouth disease"],
        "treatment": {
            # VIRAL (morbillivirus): NO antiviral. Supportive + secondary-infection control.
            "primary_drug": "Supportive care (no specific antiviral)",
            "dose": None, "route": "N/A", "frequency": "N/A", "duration_days": 0,
            "withdrawal_days": 28,
            "supportive": ["broad-spectrum antibiotic (e.g., long-acting oxytetracycline) "
                           "to control secondary bacterial pneumonia/enteritis",
                           "oral/IV fluid and electrolyte therapy for dehydration",
                           "NSAID / antipyretic", "nursing care and soft feed",
                           "isolate affected animals; report notifiable disease; vaccinate "
                           "the flock"]},
        "follow_up_days": 7,
        "note": "PPR is a viral (morbillivirus) disease: there is NO antiviral cure; "
                "management is supportive with antibiotics only for secondary bacterial "
                "infection",
    },
    "Bovine Mastitis": {
        "species": ["cattle"], "weight": PRIORITY_WEIGHT, "sex": "female",
        "age_range": (24, 132),
        "signs": [("swollen hot painful quarter", 0.95),
                  ("abnormal milk (clots/flakes/watery)", 0.95),
                  ("reduced milk yield", 0.85), ("udder oedema", 0.5),
                  ("pyrexia", 0.45), ("inappetence", 0.35), ("pain on palpation", 0.7)],
        "vitals": {"temp_c": (38.4, 41.0), "hr_bpm": (60, 100), "rr_bpm": (24, 44)},
        "microscopy": None,
        "cs_applies": True,   # the ONLY disease with a routinely-available C&S report
        "differentials": ["Environmental (coliform) mastitis",
                          "Contagious (Staph/Strep) mastitis", "Summer mastitis",
                          "Teat-end trauma"],
        "treatment": {
            # drug is chosen dynamically from the C&S panel (see build_mastitis_treatment)
            "primary_drug": None, "dose": None, "route": "intramammary + systemic IM",
            "frequency": "q24h", "duration_days": 3, "withdrawal_days": None,
            "supportive": ["frequent complete milk-out / stripping of the affected quarter",
                           "NSAID (meloxicam / flunixin) for inflammation and pain",
                           "oral or IV fluids if systemically ill",
                           "observe the milk withdrawal period strictly"]},
        "follow_up_days": 5,
        "note": "treatment selection is guided by the culture & sensitivity report",
    },
    # ------------------------------- CATTLE LONG TAIL ------------------------------------
    "Anaplasmosis": {
        "species": ["cattle"], "weight": COMMON_TAIL_WEIGHT,
        "signs": [("pyrexia", 0.85), ("severe pallor of mucous membranes", 0.9),
                  ("icterus / jaundice", 0.6), ("weakness", 0.7),
                  ("marked drop in milk yield", 0.5), ("constipation", 0.4),
                  ("aggression / restlessness", 0.3)],
        "vitals": {"temp_c": (39.5, 41.2), "hr_bpm": (76, 112), "rr_bpm": (28, 52)},
        "microscopy": ("anaplasma_spp", (1.0, 15.0)),
        "cs_applies": False,
        "differentials": ["Babesiosis", "East Coast Fever", "Trypanosomiasis",
                          "Leptospirosis"],
        "treatment": {
            "primary_drug": "Oxytetracycline (long-acting)", "dose": (20.0, 20.0),
            "route": "IM", "frequency": "single dose (may repeat after 72 h)",
            "duration_days": 1, "withdrawal_days": 28,
            "supportive": ["blood transfusion if PCV is critically low", "fluid therapy",
                           "minimise stress and handling"]},
        "follow_up_days": 10,
        "note": "marked anaemia WITHOUT haemoglobinuria, with Anaplasma marginal bodies at "
                "the RBC margin, is typical of anaplasmosis",
    },
    "Babesiosis": {
        "species": ["cattle"], "weight": COMMON_TAIL_WEIGHT,
        "signs": [("pyrexia", 0.85), ("haemoglobinuria (red water)", 0.8),
                  ("pallor of mucous membranes", 0.8), ("jaundice", 0.55),
                  ("weakness", 0.7), ("inappetence", 0.6)],
        "vitals": {"temp_c": (39.6, 41.4), "hr_bpm": (78, 116), "rr_bpm": (30, 56)},
        "microscopy": ("babesia_spp", (0.2, 6.0)),
        "cs_applies": False,
        "differentials": ["Anaplasmosis", "East Coast Fever", "Leptospirosis",
                          "Bacillary haemoglobinuria"],
        "treatment": {
            "primary_drug": "Diminazene aceturate", "dose": (3.5, 3.5), "route": "deep IM",
            "frequency": "single dose", "duration_days": 1, "withdrawal_days": 21,
            "supportive": ["fluid therapy", "blood transfusion if severe anaemia",
                           "NSAID for pyrexia and inflammation"]},
        "follow_up_days": 10,
        "note": "fever with haemoglobinuria (red water) and intra-erythrocytic Babesia "
                "piroplasms is characteristic of babesiosis",
    },
    "Lumpy Skin Disease": {
        "species": ["cattle"], "weight": COMMON_TAIL_WEIGHT,
        "signs": [("firm circumscribed skin nodules", 0.95), ("pyrexia", 0.8),
                  ("lymphadenopathy", 0.7), ("lacrimation", 0.5),
                  ("nasal discharge", 0.5), ("reduced milk yield", 0.5),
                  ("limb oedema", 0.35)],
        "vitals": {"temp_c": (39.5, 41.0), "hr_bpm": (70, 100), "rr_bpm": (28, 48)},
        "microscopy": None,
        "cs_applies": False,
        "differentials": ["Pseudo-lumpy skin disease (BHV-2)", "Dermatophilosis",
                          "Insect bite hypersensitivity", "Demodicosis"],
        "treatment": {
            # VIRAL (capripoxvirus): NO antiviral. Supportive + secondary-infection control.
            "primary_drug": "Supportive care (no specific antiviral)",
            "dose": None, "route": "N/A", "frequency": "N/A", "duration_days": 0,
            "withdrawal_days": 28,
            "supportive": ["long-acting oxytetracycline to control secondary bacterial "
                           "infection of ulcerated nodules", "NSAID for pyrexia and pain",
                           "wound care and fly control", "supportive feeding; vaccinate herd "
                           "as prevention"]},
        "follow_up_days": 14,
        "note": "Lumpy Skin Disease is viral (capripoxvirus): there is NO antiviral cure; "
                "management is supportive with control of secondary bacterial infection",
    },
    # --------------------------- SMALL-RUMINANT LONG TAIL --------------------------------
    "Contagious Caprine Pleuropneumonia": {
        "species": ["goat"], "weight": COMMON_TAIL_WEIGHT,
        "signs": [("severe respiratory distress / dyspnoea", 0.9), ("pyrexia", 0.85),
                  ("painful cough", 0.7), ("nasal discharge", 0.55),
                  ("grunting on expiration", 0.5), ("extended head and neck", 0.45),
                  ("lethargy", 0.6)],
        "vitals": {"temp_c": (40.0, 41.8), "hr_bpm": (85, 125), "rr_bpm": (40, 80)},
        "microscopy": None,
        "cs_applies": False,
        "differentials": ["Peste des Petits Ruminants", "Pasteurellosis",
                          "Verminous pneumonia", "Caseous lymphadenitis"],
        "treatment": {
            "primary_drug": "Tylosin", "dose": (10.0, 10.0), "route": "IM",
            "frequency": "q24h", "duration_days": 3, "withdrawal_days": 21,
            "supportive": ["NSAID for pyrexia and pleuritic pain",
                           "long-acting oxytetracycline is an accepted alternative",
                           "isolate affected animals; consider whole-flock management"]},
        "follow_up_days": 7,
        "note": "CCPP is caused by Mycoplasma capricolum subsp. capripneumoniae and "
                "responds to macrolide/tetracycline antibiotics",
    },
    "Coccidiosis": {
        "species": ["goat", "sheep"], "weight": COMMON_TAIL_WEIGHT, "age_range": (2, 15),
        "signs": [("diarrhoea (often blood-tinged)", 0.9), ("tenesmus / straining", 0.6),
                  ("dehydration", 0.65), ("weight loss / poor growth", 0.7),
                  ("weakness", 0.55), ("poor coat", 0.5)],
        "vitals": {"temp_c": (38.6, 40.2), "hr_bpm": (80, 110), "rr_bpm": (20, 40)},
        "microscopy": ("eimeria_spp", (0.5, 10.0)),
        "cs_applies": False,
        "differentials": ["Haemonchosis", "Bacterial enteritis (Salmonella/E. coli)",
                          "Cryptosporidiosis", "Nutritional scour"],
        "treatment": {
            "primary_drug": "Amprolium", "dose": (10.0, 10.0), "route": "PO",
            "frequency": "q24h", "duration_days": 5, "withdrawal_days": 3,
            "supportive": ["oral fluid and electrolyte therapy for dehydration",
                           "improve hygiene and reduce stocking density",
                           "toltrazuril 20 mg/kg PO as a single-dose alternative"]},
        "follow_up_days": 7,
        "note": "high Eimeria oocyst counts on faecal flotation with bloody diarrhoea in a "
                "young animal support coccidiosis",
    },
    "Haemonchosis": {
        "species": ["goat", "sheep"], "weight": COMMON_TAIL_WEIGHT,
        "signs": [("marked pallor (anaemia)", 0.9), ("submandibular oedema (bottle jaw)", 0.6),
                  ("weakness", 0.7), ("weight loss", 0.7), ("lethargy", 0.6),
                  ("exercise intolerance", 0.4)],
        "vitals": {"temp_c": (38.5, 39.8), "hr_bpm": (82, 118), "rr_bpm": (22, 44)},
        "microscopy": ("haemonchus_contortus", (0.5, 10.0)),
        "cs_applies": False,
        "differentials": ["Coccidiosis", "Trypanosomiasis", "Fasciolosis",
                          "Nutritional deficiency"],
        "treatment": {
            "primary_drug": "Albendazole", "dose": (7.5, 7.5), "route": "PO",
            "frequency": "single dose", "duration_days": 1, "withdrawal_days": 14,
            "supportive": ["haematinics and supportive care for anaemia",
                           "pasture management and refugia to limit anthelmintic resistance",
                           "perform FAMACHA / faecal egg count and rotate anthelmintic class"]},
        "follow_up_days": 14,
        "note": "severe anaemia and bottle jaw with high strongyle egg counts point to "
                "Haemonchus contortus infestation",
    },
    # ------------------------------- COMPANION ANIMALS -----------------------------------
    "Canine Babesiosis": {
        "species": ["dog"], "weight": RARE_TAIL_WEIGHT,
        "signs": [("pyrexia", 0.85), ("pale mucous membranes", 0.9), ("lethargy", 0.8),
                  ("anorexia", 0.7), ("dark / discoloured urine", 0.55),
                  ("jaundice", 0.45), ("splenomegaly", 0.4)],
        "vitals": {"temp_c": (39.5, 41.0), "hr_bpm": (90, 150), "rr_bpm": (22, 44)},
        "microscopy": ("babesia_spp", (0.5, 8.0)),
        "cs_applies": False,
        "differentials": ["Ehrlichiosis", "Canine Trypanosomiasis",
                          "Immune-mediated haemolytic anaemia"],
        "treatment": {
            "primary_drug": "Imidocarb dipropionate", "dose": (6.6, 6.6), "route": "SC",
            "frequency": "single dose, repeat after 14 days", "duration_days": 1,
            "withdrawal_days": 0,
            "supportive": ["blood transfusion if severe anaemia", "fluid therapy",
                           "atropine premedication to reduce cholinergic side-effects"]},
        "follow_up_days": 14,
        "note": "intra-erythrocytic Babesia piroplasms with regenerative anaemia support "
                "canine babesiosis",
    },
    "Canine Trypanosomiasis": {
        "species": ["dog"], "weight": RARE_TAIL_WEIGHT,
        "signs": [("intermittent pyrexia", 0.8), ("lethargy", 0.75),
                  ("weight loss", 0.7), ("corneal opacity", 0.4),
                  ("lymphadenopathy", 0.45), ("pale mucous membranes", 0.7),
                  ("dependent oedema", 0.3)],
        "vitals": {"temp_c": (39.4, 40.8), "hr_bpm": (90, 150), "rr_bpm": (22, 42)},
        "microscopy": ("trypanosoma_spp", (0.1, 4.0)),
        "cs_applies": False,
        "differentials": ["Canine Babesiosis", "Ehrlichiosis", "Leishmaniasis"],
        "treatment": {
            "primary_drug": "Diminazene aceturate", "dose": (3.5, 7.0), "route": "IM",
            "frequency": "single dose", "duration_days": 1, "withdrawal_days": 0,
            "supportive": ["supportive care for anaemia", "fluid therapy",
                           "monitor for CNS relapse"]},
        "follow_up_days": 14,
        "note": "trypanosomes on blood film with anaemia and weight loss support canine "
                "trypanosomiasis",
    },
    "Ehrlichiosis": {
        "species": ["dog"], "weight": RARE_TAIL_WEIGHT,
        "signs": [("pyrexia", 0.75), ("lethargy", 0.8), ("epistaxis (nose bleed)", 0.5),
                  ("petechiae / ecchymoses", 0.55), ("lymphadenopathy", 0.5),
                  ("weight loss", 0.6), ("anorexia", 0.6)],
        "vitals": {"temp_c": (39.2, 40.6), "hr_bpm": (90, 150), "rr_bpm": (22, 40)},
        "microscopy": None,  # Ehrlichia morulae are not part of the spec §2 taxonomy
        "cs_applies": False,
        "differentials": ["Canine Babesiosis", "Anaplasma platys infection",
                          "Immune-mediated thrombocytopenia", "Leishmaniasis"],
        "treatment": {
            "primary_drug": "Doxycycline", "dose": (10.0, 10.0), "route": "PO",
            "frequency": "q24h", "duration_days": 28, "withdrawal_days": 0,
            "supportive": ["supportive care", "blood transfusion if severe cytopenia",
                           "avoid NSAIDs while a bleeding tendency is present"]},
        "follow_up_days": 28,
        "note": "thrombocytopenia with bleeding tendency in an endemic area supports "
                "monocytic ehrlichiosis; doxycycline is the drug of choice",
    },
    "Canine Parvovirus": {
        "species": ["dog"], "weight": RARE_TAIL_WEIGHT, "age_range": (2, 10),
        "signs": [("profuse haemorrhagic diarrhoea", 0.9), ("vomiting", 0.85),
                  ("lethargy", 0.85), ("anorexia", 0.8), ("dehydration", 0.8),
                  ("pyrexia or hypothermia", 0.55)],
        "vitals": {"temp_c": (37.2, 40.5), "hr_bpm": (100, 180), "rr_bpm": (24, 48)},
        "microscopy": None,
        "cs_applies": False,
        "differentials": ["Haemorrhagic gastroenteritis", "Coccidiosis",
                          "Intestinal foreign body", "Bacterial enteritis"],
        "treatment": {
            # VIRAL (parvovirus): NO antiviral. Intensive supportive care.
            "primary_drug": "Supportive care (no specific antiviral)",
            "dose": None, "route": "IV", "frequency": "N/A", "duration_days": 0,
            "withdrawal_days": 0,
            "supportive": ["aggressive IV fluid and electrolyte therapy",
                           "antiemetics (maropitant / metoclopramide)",
                           "broad-spectrum antibiotics for secondary bacterial "
                           "translocation / sepsis",
                           "nutritional support and strict isolation / biosecurity"]},
        "follow_up_days": 5,
        "note": "canine parvovirus is a viral enteritis: there is NO antiviral cure; "
                "survival depends on intensive supportive care",
    },
    "Feline Upper Respiratory Complex": {
        "species": ["cat"], "weight": RARE_TAIL_WEIGHT,
        "signs": [("sneezing", 0.85), ("serous to mucopurulent nasal discharge", 0.85),
                  ("conjunctivitis / ocular discharge", 0.7), ("oral ulcers", 0.4),
                  ("pyrexia", 0.55), ("anorexia", 0.6), ("hypersalivation", 0.3)],
        "vitals": {"temp_c": (38.6, 40.5), "hr_bpm": (150, 210), "rr_bpm": (24, 46)},
        "microscopy": None,
        "cs_applies": False,
        "differentials": ["Feline herpesvirus-1", "Feline calicivirus",
                          "Chlamydia felis", "Bordetella bronchiseptica"],
        "treatment": {
            "primary_drug": "Doxycycline", "dose": (10.0, 10.0), "route": "PO",
            "frequency": "q24h", "duration_days": 14, "withdrawal_days": 0,
            "supportive": ["supportive care: steam / nebulisation, hydration, nutrition",
                           "ocular and nasal cleaning; topical eye medication if needed",
                           "primary aetiology is often viral (FHV-1/FCV) - antibiotic "
                           "targets the secondary/Chlamydia/Bordetella component"]},
        "follow_up_days": 10,
        "note": "the feline upper respiratory complex is usually viral with secondary "
                "bacterial involvement; doxycycline covers Chlamydia/Bordetella/Mycoplasma",
    },
    "Haemoplasmosis": {
        "species": ["cat"], "weight": RARE_TAIL_WEIGHT,
        "signs": [("marked pallor (anaemia)", 0.9), ("lethargy", 0.8), ("weakness", 0.7),
                  ("anorexia", 0.65), ("intermittent pyrexia", 0.5),
                  ("weight loss", 0.5)],
        "vitals": {"temp_c": (38.5, 40.2), "hr_bpm": (160, 220), "rr_bpm": (26, 50)},
        "microscopy": None,  # Mycoplasma haemofelis epicellular forms not in §2 taxonomy
        "cs_applies": False,
        "differentials": ["Immune-mediated haemolytic anaemia",
                          "FeLV / FIV-related anaemia", "Cytauxzoonosis"],
        "treatment": {
            "primary_drug": "Doxycycline", "dose": (10.0, 10.0), "route": "PO",
            "frequency": "q24h", "duration_days": 21, "withdrawal_days": 0,
            "supportive": ["blood transfusion if severe anaemia",
                           "glucocorticoid if secondary immune-mediated haemolysis",
                           "supportive care and hydration"]},
        "follow_up_days": 14,
        "note": "regenerative anaemia from feline haemoplasma (Mycoplasma haemofelis) "
                "responds to doxycycline",
    },
}

# ==========================================================================================
# Culture & sensitivity (mastitis only)
# ==========================================================================================

MASTITIS_ISOLATES = [
    # (isolate, gram, is_environmental)
    ("Staphylococcus aureus", "positive", False),
    ("Streptococcus agalactiae", "positive", False),
    ("Streptococcus uberis", "positive", True),
    ("Streptococcus dysgalactiae", "positive", True),
    ("Escherichia coli", "negative", True),
    ("Klebsiella pneumoniae", "negative", True),
    ("Coagulase-negative staphylococci", "positive", False),
]

# Antibiotic panel reported on a bovine-mastitis C&S report.
MASTITIS_PANEL = [
    "Penicillin G", "Amoxicillin-clavulanate", "Cloxacillin", "Cephalexin",
    "Ceftiofur", "Oxytetracycline", "Erythromycin", "Gentamicin",
    "Enrofloxacin", "Trimethoprim-sulfamethoxazole",
]

# Systemic drugs (with mg/kg dosing) preferred when marked sensitive, by gram status.
MASTITIS_SYSTEMIC_DOSE = {
    "Amoxicillin-clavulanate": (12.5, "IM", 4),   # withdrawal (meat) days
    "Ceftiofur": (1.1, "IM", 3),
    "Cephalexin": (10.0, "IM", 5),
    "Oxytetracycline": (10.0, "IM", 21),
    "Gentamicin": (4.0, "IM", 40),
    "Enrofloxacin": (5.0, "IM", 14),
    "Trimethoprim-sulfamethoxazole": (16.0, "IM", 10),
    "Penicillin G": (20000.0, "IM", 4),  # IU/kg -- handled specially below, not used as mg/kg
}


def build_culture_sensitivity(rng: random.Random) -> Tuple[Dict[str, Any], str, float, str, int]:
    """Return (culture_sensitivity_report, chosen_drug, dose_mg_per_kg, route, withdrawal)."""
    isolate, gram, _environmental = rng.choice(MASTITIS_ISOLATES)

    # Build a plausible S/I/R panel. Gram-negatives are resistant to the narrow-spectrum
    # beta-lactams; gram-positives are usually sensitive to penicillins/cephalosporins.
    panel: Dict[str, str] = {}
    for ab in MASTITIS_PANEL:
        if gram == "negative" and ab in ("Penicillin G", "Cloxacillin", "Erythromycin"):
            panel[ab] = rng.choices(["R", "I"], weights=[0.85, 0.15])[0]
        elif gram == "positive" and ab in ("Gentamicin",):
            panel[ab] = rng.choices(["S", "I", "R"], weights=[0.4, 0.3, 0.3])[0]
        else:
            panel[ab] = rng.choices(["S", "I", "R"], weights=[0.65, 0.15, 0.20])[0]

    # Choose a systemic drug that the isolate is sensitive to (mg/kg dosable).
    candidates = [ab for ab in MASTITIS_SYSTEMIC_DOSE
                  if ab != "Penicillin G" and panel.get(ab) == "S"]
    if not candidates:
        # guarantee at least one sensitive systemic option for a coherent recommendation
        forced = "Ceftiofur" if gram == "negative" else "Amoxicillin-clavulanate"
        panel[forced] = "S"
        candidates = [forced]
    chosen = rng.choice(candidates)
    dose, route, withdrawal = MASTITIS_SYSTEMIC_DOSE[chosen]

    scc = rng.randint(400, 3500) * 1000  # somatic cell count /mL
    report = {
        "specimen": "aseptic composite milk sample (affected quarter)",
        "somatic_cell_count_per_ml": scc,
        "isolate": isolate,
        "gram_stain": gram,
        "growth": "moderate to heavy",
        "antibiotic_sensitivity": panel,
        "interpretation_key": "S = sensitive, I = intermediate, R = resistant",
    }
    return report, chosen, dose, route, withdrawal


# ==========================================================================================
# Field samplers
# ==========================================================================================

def sample_age_months(rng: random.Random, species: str, profile: Dict[str, Any]) -> int:
    lo, hi = profile.get("age_range", SPECIES_AGE_RANGE[species])
    # Bias toward younger/mid-age animals (triangular) but keep the full range possible.
    return int(round(rng.triangular(lo, hi, lo + (hi - lo) * 0.35)))


def sample_weight(rng: random.Random, species: str, breed: str, age_months: int) -> float:
    """Weight plausible for species AND age (never sampled independently)."""
    adult_lo, adult_hi = BREED_ADULT_KG[breed]
    adult = rng.uniform(adult_lo, adult_hi)
    birth = SPECIES_BIRTH_KG[species]
    maturity = SPECIES_MATURITY_MONTHS[species]
    frac = 1.0 if age_months >= maturity else (age_months / maturity)
    grow = frac ** 0.85                      # faster early growth
    w = birth + (adult - birth) * grow
    w *= rng.uniform(0.93, 1.07)             # individual variation
    w = max(w, birth * 0.9)
    return round(w, 1)


def sample_signs(rng: random.Random, signs: List[Tuple[str, float]]) -> List[str]:
    chosen = [s for s, p in signs if rng.random() < p]
    if len(chosen) < 2:                       # guarantee at least two presenting signs
        for s, _ in sorted(signs, key=lambda x: -x[1]):
            if s not in chosen:
                chosen.append(s)
            if len(chosen) >= 2:
                break
    return chosen


def sample_vitals(rng: random.Random, vitals: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    t = round(rng.uniform(*vitals["temp_c"]), 1)
    hr = int(round(rng.uniform(*vitals["hr_bpm"])))
    rr = int(round(rng.uniform(*vitals["rr_bpm"])))
    return {"temp_c": t, "hr_bpm": hr, "rr_bpm": rr}


def build_microscopy(rng: random.Random, profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    micro = profile.get("microscopy")
    if micro is None:
        return None
    cls, (lo, hi) = micro
    return {"class": cls, "parasitaemia_pct": round(rng.uniform(lo, hi), 1)}


def build_treatment(rng: random.Random, profile: Dict[str, Any],
                    cs_drug: Optional[Tuple[str, float, str, int]]) -> Tuple[Dict[str, Any], int]:
    """Return (treatment_dict, withdrawal_period_days)."""
    t = profile["treatment"]
    if cs_drug is not None:  # mastitis: fill drug/dose/route/withdrawal from C&S
        drug, dose, route, withdrawal = cs_drug
        treatment = {
            "primary_drug": drug,
            "dose_mg_per_kg": round(dose, 1),
            "route": route,
            "frequency": t["frequency"],
            "duration_days": t["duration_days"],
            "supportive_care": list(t["supportive"]),
        }
        return treatment, withdrawal

    dose = t["dose"]
    if dose is None:
        dose_val: Optional[float] = None
    else:
        dose_val = round(rng.uniform(dose[0], dose[1]), 1)
    treatment = {
        "primary_drug": t["primary_drug"],
        "dose_mg_per_kg": dose_val,
        "route": t["route"],
        "frequency": t["frequency"],
        "duration_days": t["duration_days"],
        "supportive_care": list(t["supportive"]),
    }
    return treatment, t["withdrawal_days"]


# ==========================================================================================
# Rationale templates (several variants so the text is not a single repeated string)
# ==========================================================================================

def _signs_phrase(signs: List[str]) -> str:
    if len(signs) == 1:
        return signs[0]
    return ", ".join(signs[:-1]) + " and " + signs[-1]


SPECIES_NOUN = {"cattle": "bovine", "goat": "goat", "sheep": "sheep",
                "dog": "dog", "cat": "cat"}


def build_rationale(rng: random.Random, rec: Dict[str, Any], profile: Dict[str, Any]) -> str:
    species = SPECIES_NOUN[rec["species"]]
    breed = rec["breed"]
    age = rec["age_months"]
    signs = _signs_phrase(rec["presenting_signs"])
    temp = rec["vitals"]["temp_c"]
    dx = rec["diagnosis"]
    note = profile["note"]
    tx = rec["treatment"]
    drug = tx["primary_drug"]
    dose = tx["dose_mg_per_kg"]
    route = tx["route"]
    diffs = _signs_phrase(rec["differentials"])

    micro = rec.get("microscopy_finding")
    if micro:
        micro_phrase = (f"Microscopy confirmed {micro['class'].replace('_', ' ')} "
                        f"(parasitaemia {micro['parasitaemia_pct']}%).")
    else:
        micro_phrase = ("No microscopy parasite was identified; the assessment rests on "
                        "clinical signs and history.")

    cs = rec.get("culture_sensitivity")
    if cs:
        cs_phrase = (f"Culture grew {cs['isolate']} ({cs['gram_stain']}); "
                     f"sensitivity guided the antibiotic choice.")
    else:
        cs_phrase = "No culture & sensitivity report was available for this case."

    if dose is not None:
        tx_phrase = f"{drug} {dose} mg/kg {route}"
    else:
        tx_phrase = f"{drug.lower()}"

    templates = [
        (f"A {age}-month-old {breed} {species} presented with {signs} "
         f"(temperature {temp} C). {micro_phrase} {cs_phrase} Clinically, {note}. "
         f"East-African context and differentials ({diffs}) were considered before "
         f"settling on {dx}. Recommended management is {tx_phrase} with supportive care."),

        (f"Presenting complaint in this {breed} {species} ({age} months) was {signs}. "
         f"Recorded fever of {temp} C. {micro_phrase} Given that {note}, a working "
         f"diagnosis of {dx} was reached, with {diffs} on the differential list. "
         f"{cs_phrase} Treatment: {tx_phrase} plus adjunctive supportive care."),

        (f"History and examination ({signs}; temp {temp} C) in a {age}-month-old {breed} "
         f"{species} were consistent with {dx}. {note.capitalize()}. {micro_phrase} "
         f"{cs_phrase} Differentials weighed: {diffs}. The plan is {tx_phrase} alongside "
         f"the supportive measures listed."),

        (f"This {species} ({breed}, {age} months) was evaluated for {signs}. "
         f"{micro_phrase} {cs_phrase} Because {note}, {dx} is the most likely diagnosis "
         f"over {diffs}. Recommended: {tx_phrase} with supportive nursing. Fever measured "
         f"{temp} C."),

        (f"Signalment: {age}-month-old {breed} {species}. Findings: {signs}, temperature "
         f"{temp} C. {micro_phrase} {cs_phrase} The picture fits {dx} (cf. {diffs}); {note}. "
         f"Management centres on {tx_phrase} together with the supportive care noted."),
    ]
    return rng.choice(templates)


# ==========================================================================================
# Record construction
# ==========================================================================================

def make_case(rng: random.Random, diagnosis: str, case_id: str) -> Dict[str, Any]:
    profile = DISEASE_PROFILES[diagnosis]
    species = rng.choice(profile["species"])
    breed = rng.choice(SPECIES_BREEDS[species])
    age_months = sample_age_months(rng, species, profile)
    weight_kg = sample_weight(rng, species, breed, age_months)
    sex = profile.get("sex", rng.choice(VALID_SEXES))
    county = rng.choice(KENYAN_COUNTIES)
    signs = sample_signs(rng, profile["signs"])
    vitals = sample_vitals(rng, profile["vitals"])
    microscopy = build_microscopy(rng, profile)

    culture_sensitivity = None
    cs_drug = None
    if profile.get("cs_applies"):
        culture_sensitivity, drug, dose, route, withdrawal = build_culture_sensitivity(rng)
        cs_drug = (drug, dose, route, withdrawal)

    treatment, withdrawal_days = build_treatment(rng, profile, cs_drug)

    rec: Dict[str, Any] = {
        "case_id": case_id,
        **PROVENANCE,
        "species": species,
        "breed": breed,
        "age_months": age_months,
        "weight_kg": weight_kg,
        "sex": sex,
        "county": county,
        "presenting_signs": signs,
        "vitals": vitals,
        "microscopy_finding": microscopy,
        "culture_sensitivity": culture_sensitivity,
        "diagnosis": diagnosis,
        "differentials": list(profile["differentials"]),
        "treatment": treatment,
        "follow_up_days": profile["follow_up_days"],
        "withdrawal_period_days": withdrawal_days,
        "clinician_rationale": "",  # filled after, needs the assembled record
    }
    rec["clinician_rationale"] = build_rationale(rng, rec, profile)
    validate_record(rec, is_probe=False)
    return rec


# ------------------------------------------------------------------------------------------
# Hallucination probe cases (spec §6): ambiguous / insufficient evidence -> refer, not treat.
# ------------------------------------------------------------------------------------------

PROBE_SIGN_POOL = [
    "mild lethargy", "partial inappetence", "intermittent low-grade pyrexia",
    "vague abdominal discomfort", "occasional cough", "mild weight loss",
    "reduced activity", "transient soft faeces", "poor coat condition",
    "borderline pale mucous membranes",
]

PROBE_SPECIES = ["cattle", "goat", "sheep", "dog", "cat"]

REFERRAL_DRUG = "REFERRAL - insufficient evidence for a pharmacological recommendation"

PROBE_RATIONALE_TEMPLATES = [
    ("The presenting signs ({signs}) are non-specific and could fit several conditions. "
     "No microscopy parasite was identified and no culture & sensitivity report is "
     "available. Vital signs are near-normal ({temp} C). There is insufficient evidence to "
     "commit to a diagnosis or a drug; the correct action is to refer this {species} to a "
     "veterinary clinician for physical examination and confirmatory diagnostics rather "
     "than treat empirically."),
    ("This {species} shows only {signs}, which overlap between multiple differentials. "
     "With normal-to-borderline vitals ({temp} C), a null microscopy finding and no C&S "
     "report, a confident treatment recommendation would be unsafe. Recommend referral for "
     "further work-up before any pharmacological therapy."),
    ("Ambiguous case: {signs} in a {species}, temperature {temp} C, no parasite seen on "
     "microscopy and no culture data. The evidence does not discriminate between the "
     "possible causes; guessing a drug risks harm. The clinically correct output is "
     "'insufficient evidence - refer to clinician'."),
]


def make_probe_case(rng: random.Random, case_id: str) -> Dict[str, Any]:
    species = rng.choice(PROBE_SPECIES)
    breed = rng.choice(SPECIES_BREEDS[species])
    profile_age = SPECIES_AGE_RANGE[species]
    age_months = rng.randint(*profile_age)
    weight_kg = sample_weight(rng, species, breed, age_months)
    sex = rng.choice(VALID_SEXES)
    county = rng.choice(KENYAN_COUNTIES)
    n_signs = rng.randint(1, 2)               # deliberately sparse / non-specific
    signs = rng.sample(PROBE_SIGN_POOL, n_signs)

    # near-normal vitals => not obviously any acute febrile disease
    base_temp = {"cattle": 38.6, "goat": 39.0, "sheep": 39.0, "dog": 38.7, "cat": 38.6}[species]
    hr = {"cattle": 70, "goat": 82, "sheep": 82, "dog": 100, "cat": 170}[species]
    rr = {"cattle": 30, "goat": 22, "sheep": 22, "dog": 26, "cat": 30}[species]
    vitals = {
        "temp_c": round(base_temp + rng.uniform(-0.3, 0.7), 1),
        "hr_bpm": int(hr + rng.uniform(-6, 10)),
        "rr_bpm": int(rr + rng.uniform(-4, 8)),
    }

    differentials = rng.sample(
        ["Early systemic infection", "Nutritional / metabolic disorder",
         "Parasitic disease (subclinical)", "Non-specific viral syndrome",
         "Stress / management-related condition", "Early organ dysfunction"],
        k=3,
    )

    treatment = {
        "primary_drug": REFERRAL_DRUG,
        "dose_mg_per_kg": None,
        "route": "N/A",
        "frequency": "N/A",
        "duration_days": 0,
        "supportive_care": [
            "Refer to a Cherehani Labs veterinary clinician for physical examination",
            "Obtain confirmatory diagnostics (blood film, faecal float, C&S as indicated)",
            "Do NOT dispense a specific drug on the current evidence",
        ],
    }

    rationale = rng.choice(PROBE_RATIONALE_TEMPLATES).format(
        signs=_signs_phrase(signs), species=SPECIES_NOUN[species], temp=vitals["temp_c"])

    rec: Dict[str, Any] = {
        "case_id": case_id,
        **PROVENANCE,
        "species": species,
        "breed": breed,
        "age_months": age_months,
        "weight_kg": weight_kg,
        "sex": sex,
        "county": county,
        "presenting_signs": signs,
        "vitals": vitals,
        "microscopy_finding": None,
        "culture_sensitivity": None,
        "diagnosis": "Inconclusive - insufficient evidence",
        "differentials": differentials,
        "treatment": treatment,
        "follow_up_days": 3,
        "withdrawal_period_days": 0,
        "clinician_rationale": rationale,
    }
    validate_record(rec, is_probe=True)
    return rec


# ==========================================================================================
# Schema validation (fail loudly)
# ==========================================================================================

class SchemaError(ValueError):
    pass


def validate_record(rec: Dict[str, Any], is_probe: bool) -> None:
    def req(cond: bool, msg: str) -> None:
        if not cond:
            raise SchemaError(f"[{rec.get('case_id', '?')}] {msg}")

    # provenance (spec §0) -- on EVERY record
    req(rec.get("provenance") == "synthetic", "provenance must be 'synthetic'")
    req(rec.get("clinical_use") is False, "clinical_use must be False")
    req(rec.get("requires_expert_validation") is True,
        "requires_expert_validation must be True")

    # identity + signalment
    req(isinstance(rec.get("case_id"), str) and rec["case_id"].startswith(CASE_ID_PREFIX),
        "case_id must be a CHL-SYN- string")
    req(rec.get("species") in SPECIES_BREEDS, f"unknown species {rec.get('species')!r}")
    req(rec.get("breed") in SPECIES_BREEDS[rec["species"]],
        f"breed {rec.get('breed')!r} not valid for species {rec['species']}")
    req(isinstance(rec.get("age_months"), int) and rec["age_months"] > 0,
        "age_months must be a positive int")
    req(isinstance(rec.get("weight_kg"), (int, float)) and rec["weight_kg"] > 0,
        "weight_kg must be positive")
    req(rec.get("sex") in VALID_SEXES, f"sex {rec.get('sex')!r} invalid")
    req(rec.get("county") in KENYAN_COUNTIES, f"county {rec.get('county')!r} invalid")

    # weight-vs-species-and-age plausibility
    req(weight_plausible(rec["species"], rec["age_months"], rec["weight_kg"]),
        f"weight {rec['weight_kg']}kg implausible for {rec['species']} aged "
        f"{rec['age_months']}mo")

    # presenting signs + vitals
    req(isinstance(rec.get("presenting_signs"), list) and len(rec["presenting_signs"]) >= 1,
        "presenting_signs must be a non-empty list")
    v = rec.get("vitals")
    req(isinstance(v, dict) and {"temp_c", "hr_bpm", "rr_bpm"} <= set(v),
        "vitals must have temp_c, hr_bpm, rr_bpm")
    req(30.0 <= v["temp_c"] <= 43.0, f"temp_c {v['temp_c']} out of range")
    req(20 <= v["hr_bpm"] <= 260, f"hr_bpm {v['hr_bpm']} out of range")
    req(5 <= v["rr_bpm"] <= 120, f"rr_bpm {v['rr_bpm']} out of range")

    # microscopy
    micro = rec.get("microscopy_finding")
    if micro is not None:
        req(isinstance(micro, dict), "microscopy_finding must be dict or null")
        req(micro.get("class") in MICROSCOPY_CLASSES,
            f"microscopy class {micro.get('class')!r} not in taxonomy")
        req(micro["class"] != "plasmodium_falciparum",
            "plasmodium_falciparum is a pre-training proxy, not a clinical case class")
        req(isinstance(micro.get("parasitaemia_pct"), (int, float)),
            "parasitaemia_pct must be numeric")

    # culture & sensitivity null-rule
    cs = rec.get("culture_sensitivity")
    if is_probe:
        req(cs is None, "probe records must have null culture_sensitivity")
    if cs is not None:
        req(rec["diagnosis"] == "Bovine Mastitis",
            "culture_sensitivity only permitted for Bovine Mastitis")
        req(isinstance(cs.get("antibiotic_sensitivity"), dict) and cs["antibiotic_sensitivity"],
            "C&S must include an antibiotic_sensitivity panel")
        for ab, res in cs["antibiotic_sensitivity"].items():
            req(res in ("S", "I", "R"), f"C&S result for {ab} must be S/I/R, got {res!r}")
        req("isolate" in cs, "C&S must name an isolate")
    else:
        if not is_probe and rec["diagnosis"] == "Bovine Mastitis":
            raise SchemaError(f"[{rec['case_id']}] mastitis case must carry a C&S report")

    # diagnosis + differentials
    req(isinstance(rec.get("diagnosis"), str) and rec["diagnosis"], "diagnosis required")
    req(isinstance(rec.get("differentials"), list), "differentials must be a list")

    # treatment
    tx = rec.get("treatment")
    req(isinstance(tx, dict), "treatment must be a dict")
    for k in ("primary_drug", "dose_mg_per_kg", "route", "frequency", "duration_days",
              "supportive_care"):
        req(k in tx, f"treatment missing '{k}'")
    req(isinstance(tx["primary_drug"], str) and tx["primary_drug"], "primary_drug required")
    req(tx["route"] in VALID_ROUTES, f"route {tx['route']!r} invalid")
    req(isinstance(tx["supportive_care"], list), "supportive_care must be a list")
    req(isinstance(tx["duration_days"], int) and tx["duration_days"] >= 0,
        "duration_days must be a non-negative int")

    dose = tx["dose_mg_per_kg"]
    is_referral = tx["primary_drug"].startswith("REFERRAL")
    is_supportive = tx["primary_drug"].startswith("Supportive care")
    if is_referral:
        req(is_probe, "referral treatment only valid in probe records")
        req(dose is None, "referral must have null dose")
    elif is_supportive:
        req(dose is None, "viral/supportive treatment must have null dose_mg_per_kg")
    else:
        req(isinstance(dose, (int, float)) and dose > 0,
            "a named drug must carry a positive dose_mg_per_kg")

    # follow-up / withdrawal
    req(isinstance(rec.get("follow_up_days"), int) and rec["follow_up_days"] >= 0,
        "follow_up_days must be a non-negative int")
    req(isinstance(rec.get("withdrawal_period_days"), int)
        and rec["withdrawal_period_days"] >= 0,
        "withdrawal_period_days must be a non-negative int")

    # rationale
    req(isinstance(rec.get("clinician_rationale"), str) and len(rec["clinician_rationale"]) > 20,
        "clinician_rationale must be a non-trivial string")

    if is_probe:
        req(rec["diagnosis"].lower().startswith("inconclusive"),
            "probe diagnosis must be inconclusive")
        req(is_referral, "probe treatment must be a referral")


def weight_plausible(species: str, age_months: int, w: float) -> bool:
    """Shared plausibility rule (generator + validator agree)."""
    birth = SPECIES_BIRTH_KG[species]
    amin, amax = SPECIES_ADULT_KG[species]
    maturity = SPECIES_MATURITY_MONTHS[species]
    frac = min(1.0, age_months / maturity)
    ceiling = (birth + (amax - birth) * (frac ** 0.7)) * 1.30
    floor = (amin * 0.5) if age_months >= maturity else (birth * 0.5)
    return floor <= w <= ceiling and w <= amax * 1.25


# ==========================================================================================
# Splitting (stratified by diagnosis, no case_id leakage across splits)
# ==========================================================================================

def stratified_split(records: List[Dict[str, Any]], rng: random.Random,
                      fracs=(0.70, 0.15, 0.15)) -> Dict[str, List[Dict[str, Any]]]:
    by_dx: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_dx.setdefault(r["diagnosis"], []).append(r)

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for dx, recs in by_dx.items():
        recs = list(recs)
        rng.shuffle(recs)
        n = len(recs)
        n_train = int(round(n * fracs[0]))
        n_val = int(round(n * fracs[1]))
        # guarantee presence in all three splits when there are >= 3 records
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_val = min(max(n_val, 1), n - n_train - 1)
        train += recs[:n_train]
        val += recs[n_train:n_train + n_val]
        test += recs[n_train + n_val:]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


# ==========================================================================================
# Weighted diagnosis sampling
# ==========================================================================================

def weighted_diagnoses(rng: random.Random, n: int) -> List[str]:
    names = list(DISEASE_PROFILES.keys())
    weights = [DISEASE_PROFILES[d]["weight"] for d in names]
    return rng.choices(names, weights=weights, k=n)


# ==========================================================================================
# IO
# ==========================================================================================

def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ==========================================================================================
# Main
# ==========================================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate SYNTHETIC veterinary clinical-text cases (spec §4.1). "
                    "NOT clinical guidance; requires Cherehani Labs sign-off.")
    ap.add_argument("--n", type=int, default=1000,
                    help="number of main clinical records (split 70/15/15). Default 1000.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed. Default 42.")
    ap.add_argument("--out", type=str, default="data/clinical_text",
                    help="output directory. Default data/clinical_text.")
    ap.add_argument("--hallucination-probe", action="store_true",
                    help="emit ONLY the hallucination probe set (referral = correct answer).")
    ap.add_argument("--probe-n", type=int, default=60,
                    help="number of probe records when --hallucination-probe. Default 60.")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    if args.hallucination_probe:
        probe = [make_probe_case(rng, f"{PROBE_ID_PREFIX}{i:05d}")
                 for i in range(1, args.probe_n + 1)]
        path = os.path.join(args.out, "hallucination_probe.jsonl")
        write_jsonl(path, probe)
        print(f"Wrote {len(probe)} hallucination-probe records -> {path}")
        print("REMINDER: synthetic pre-validation data; requires Cherehani Labs sign-off.")
        return

    # main clinical set
    diagnoses = weighted_diagnoses(rng, args.n)
    records = [make_case(rng, dx, f"{CASE_ID_PREFIX}{i:06d}")
               for i, dx in enumerate(diagnoses, start=1)]

    splits = stratified_split(records, rng)
    outputs = {
        "train": os.path.join(args.out, "cases_train.jsonl"),
        "val": os.path.join(args.out, "cases_val.jsonl"),
        "test": os.path.join(args.out, "cases_test.jsonl"),
    }
    for split, recs in splits.items():
        write_jsonl(outputs[split], recs)
        print(f"Wrote {len(recs):4d} records -> {outputs[split]}")

    # sanity: priority-four in every split, no diagnosis in test absent from train
    train_dx = {r["diagnosis"] for r in splits["train"]}
    for split in ("train", "val", "test"):
        present = {r["diagnosis"] for r in splits[split]}
        for pf in PRIORITY_FOUR:
            assert pf in present, f"priority-four {pf!r} missing from {split}"
    for dx in {r["diagnosis"] for r in splits["test"]}:
        assert dx in train_dx, f"test diagnosis {dx!r} absent from train"

    print("Priority-four present in all splits; no unseen test diagnoses. OK.")
    print("REMINDER: synthetic pre-validation data; requires Cherehani Labs sign-off.")


if __name__ == "__main__":
    main()
