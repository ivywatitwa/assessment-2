#!/usr/bin/env python3
"""Build the RAG knowledge-base chunk file (kb_chunks.jsonl).

This is the reproducible provenance record for the grounding corpus described in
docs/PROJECT_SPEC.md (schema 4.3). Every chunk is derived from a REAL, verified,
open-access veterinary source. No source, citation, DOI or URL in this file was
invented: each was located and fetched during curation (see sources.json for the
verification status of every source, and data/rag_knowledge_base/README.md for
the methodology).

Run:  python scripts/build_kb_chunks.py
Emits: data/rag_knowledge_base/kb_chunks.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

ACCESS_DATE = "2026-07-09"
PROVENANCE = "curated_from_literature"

# --------------------------------------------------------------------------- #
# Source manifest. Each entry is a real source that was fetched and verified
# during curation. `verification` is one of:
#   verified_fetched        -> full text was retrieved and read during curation
#   verified_metadata_only  -> existence/metadata confirmed but full text not read
# --------------------------------------------------------------------------- #
SOURCES: dict[str, dict] = {
    "merck_theileriosis": {
        "citation": (
            "Merck Veterinary Manual. (2022). Theileriosis in animals. In "
            "Merck Veterinary Manual (online ed.). Merck & Co. Retrieved "
            f"{ACCESS_DATE}, from https://www.merckvetmanual.com/"
            "circulatory-system/blood-parasites/theileriosis-in-animals"
        ),
        "type": "manual",
        "url": "https://www.merckvetmanual.com/circulatory-system/blood-parasites/theileriosis-in-animals",
        "verification": "verified_fetched",
    },
    "cfsph_theileriosis": {
        "citation": (
            "Center for Food Security and Public Health. (2019). Theileriosis "
            "in cattle and small ruminants (Theileria parva: East Coast fever, "
            "Corridor disease; Theileria annulata: Tropical theileriosis). Iowa "
            f"State University. Retrieved {ACCESS_DATE}, from "
            "https://www.cfsph.iastate.edu/Factsheets/pdfs/"
            "theileriosis_theileria_parva_and_theileria_annulata.pdf"
        ),
        "type": "guideline",
        "url": "https://www.cfsph.iastate.edu/Factsheets/pdfs/theileriosis_theileria_parva_and_theileria_annulata.pdf",
        "verification": "verified_fetched",
    },
    "patel_2019_itm": {
        "citation": (
            "Patel, E., Mwaura, S., Di Giulio, G., Cook, E. A. J., Lynen, G., & "
            "Toye, P. (2019). Infection and treatment method (ITM) vaccine "
            "against East Coast fever: reducing the number of doses per straw "
            "for use in smallholder dairy herds by thawing, diluting and "
            "refreezing already packaged vaccine. BMC Veterinary Research, "
            "15, 46. https://doi.org/10.1186/s12917-019-1787-y"
        ),
        "type": "peer_reviewed",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6357393/",
        "verification": "verified_fetched",
    },
    "merck_trypanosomiasis": {
        "citation": (
            "Merck Veterinary Manual. (2022). Trypanosomiasis in animals. In "
            "Merck Veterinary Manual (online ed.). Merck & Co. Retrieved "
            f"{ACCESS_DATE}, from https://www.merckvetmanual.com/"
            "circulatory-system/blood-parasites/trypanosomiasis-in-animals"
        ),
        "type": "manual",
        "url": "https://www.merckvetmanual.com/circulatory-system/blood-parasites/trypanosomiasis-in-animals",
        "verification": "verified_fetched",
    },
    "okello_2022_aat": {
        "citation": (
            "Okello, I., Mafie, E., Eastwood, G., Nzalawahe, J., Mboera, "
            "L. E. G., & Onyoyo, S. (2022). Prevalence and associated risk "
            "factors of African animal trypanosomiasis in cattle in Lambwe, "
            "Kenya. Journal of Parasitology Research, 2022, 5984376. "
            "https://doi.org/10.1155/2022/5984376"
        ),
        "type": "peer_reviewed",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9303511/",
        "verification": "verified_fetched",
    },
    "merck_ppr": {
        "citation": (
            "Merck Veterinary Manual. (2022). Peste des petits ruminants. In "
            "Merck Veterinary Manual (online ed.). Merck & Co. Retrieved "
            f"{ACCESS_DATE}, from https://www.merckvetmanual.com/"
            "generalized-conditions/peste-des-petits-ruminants/"
            "peste-des-petits-ruminants"
        ),
        "type": "manual",
        "url": "https://www.merckvetmanual.com/generalized-conditions/peste-des-petits-ruminants/peste-des-petits-ruminants",
        "verification": "verified_fetched",
    },
    "kihu_2015_ppr": {
        "citation": (
            "Kihu, S. M., Gachohi, J. M., Ndungu, E. K., Gitao, G. C., Bebora, "
            "L. C., John, N. M., Wairire, G. G., Maingi, N., Wahome, R. G., & "
            "Ireri, R. (2015). Sero-epidemiology of Peste des petits ruminants "
            "virus infection in Turkana County, Kenya. BMC Veterinary "
            "Research, 11, 87. https://doi.org/10.1186/s12917-015-0401-1"
        ),
        "type": "peer_reviewed",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4396631/",
        "verification": "verified_fetched",
    },
    "merck_mastitis": {
        "citation": (
            "Merck Veterinary Manual. (2022). Mastitis in cattle. In Merck "
            "Veterinary Manual (online ed.). Merck & Co. Retrieved "
            f"{ACCESS_DATE}, from https://www.merckvetmanual.com/"
            "reproductive-system/mastitis-in-large-animals/mastitis-in-cattle"
        ),
        "type": "manual",
        "url": "https://www.merckvetmanual.com/reproductive-system/mastitis-in-large-animals/mastitis-in-cattle",
        "verification": "verified_fetched",
    },
    "mbindyo_2020_mastitis": {
        "citation": (
            "Mbindyo, C. M., Gitao, G. C., & Mulei, C. M. (2020). Prevalence, "
            "etiology, and risk factors of mastitis in dairy cattle in Embu "
            "and Kajiado Counties, Kenya. Veterinary Medicine International, "
            "2020, 8831172. https://doi.org/10.1155/2020/8831172"
        ),
        "type": "peer_reviewed",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7424489/",
        "verification": "verified_fetched",
    },
    # ---- long-tail Merck Veterinary Manual sources (fetched during curation) ----
    "merck_anaplasmosis": {
        "citation": (
            "Merck Veterinary Manual. (2022). Anaplasmosis in ruminants. In "
            "Merck Veterinary Manual (online ed.). Merck & Co. Retrieved "
            f"{ACCESS_DATE}, from https://www.merckvetmanual.com/"
            "circulatory-system/blood-parasites/anaplasmosis-in-ruminants"
        ),
        "type": "manual",
        "url": "https://www.merckvetmanual.com/circulatory-system/blood-parasites/anaplasmosis-in-ruminants",
        "verification": "verified_fetched",
    },
    "merck_babesiosis": {
        "citation": (
            "Merck Veterinary Manual. (2022). Babesiosis in animals. In Merck "
            "Veterinary Manual (online ed.). Merck & Co. Retrieved "
            f"{ACCESS_DATE}, from https://www.merckvetmanual.com/"
            "circulatory-system/blood-parasites/babesiosis-in-animals"
        ),
        "type": "manual",
        "url": "https://www.merckvetmanual.com/circulatory-system/blood-parasites/babesiosis-in-animals",
        "verification": "verified_fetched",
    },
    "merck_lsd": {
        "citation": (
            "Merck Veterinary Manual. (2022). Lumpy skin disease in cattle. In "
            "Merck Veterinary Manual (online ed.). Merck & Co. Retrieved "
            f"{ACCESS_DATE}, from https://www.merckvetmanual.com/"
            "integumentary-system/pox-diseases/lumpy-skin-disease-in-cattle"
        ),
        "type": "manual",
        "url": "https://www.merckvetmanual.com/integumentary-system/pox-diseases/lumpy-skin-disease-in-cattle",
        "verification": "verified_fetched",
    },
    "merck_ccpp": {
        "citation": (
            "Merck Veterinary Manual. (2022). Mycoplasma pneumonias in goats. "
            "In Merck Veterinary Manual (online ed.). Merck & Co. Retrieved "
            f"{ACCESS_DATE}, from https://www.merckvetmanual.com/"
            "respiratory-system/respiratory-diseases-of-sheep-and-goats/"
            "mycoplasma-pneumonias-in-goats"
        ),
        "type": "manual",
        "url": "https://www.merckvetmanual.com/respiratory-system/respiratory-diseases-of-sheep-and-goats/mycoplasma-pneumonias-in-goats",
        "verification": "verified_fetched",
    },
    "merck_coccidiosis": {
        "citation": (
            "Merck Veterinary Manual. (2022). Overview of coccidiosis in "
            "animals. In Merck Veterinary Manual (online ed.). Merck & Co. "
            f"Retrieved {ACCESS_DATE}, from https://www.merckvetmanual.com/"
            "digestive-system/coccidiosis/overview-of-coccidiosis-in-animals"
        ),
        "type": "manual",
        "url": "https://www.merckvetmanual.com/digestive-system/coccidiosis/overview-of-coccidiosis-in-animals",
        "verification": "verified_fetched",
    },
    "merck_gi_smallrum": {
        "citation": (
            "Merck Veterinary Manual. (2022). Common gastrointestinal "
            "parasites of small ruminants. In Merck Veterinary Manual (online "
            f"ed.). Merck & Co. Retrieved {ACCESS_DATE}, from "
            "https://www.merckvetmanual.com/digestive-system/"
            "gastrointestinal-parasites-of-ruminants/"
            "common-gastrointestinal-parasites-of-small-ruminants"
        ),
        "type": "manual",
        "url": "https://www.merckvetmanual.com/digestive-system/gastrointestinal-parasites-of-ruminants/common-gastrointestinal-parasites-of-small-ruminants",
        "verification": "verified_fetched",
    },
    "merck_gi_overview": {
        "citation": (
            "Merck Veterinary Manual. (2022). Overview of gastrointestinal "
            "parasites of ruminants. In Merck Veterinary Manual (online ed.). "
            f"Merck & Co. Retrieved {ACCESS_DATE}, from "
            "https://www.merckvetmanual.com/digestive-system/"
            "gastrointestinal-parasites-of-ruminants/"
            "overview-of-gastrointestinal-parasites-of-ruminants"
        ),
        "type": "manual",
        "url": "https://www.merckvetmanual.com/digestive-system/gastrointestinal-parasites-of-ruminants/overview-of-gastrointestinal-parasites-of-ruminants",
        "verification": "verified_fetched",
    },
    "merck_ehrlichiosis": {
        "citation": (
            "Merck Veterinary Manual. (2022). Ehrlichiosis in dogs. In Merck "
            "Veterinary Manual (online ed.). Merck & Co. Retrieved "
            f"{ACCESS_DATE}, from https://www.merckvetmanual.com/"
            "infectious-diseases/rickettsial-diseases-in-dogs/"
            "ehrlichiosis-in-dogs"
        ),
        "type": "manual",
        "url": "https://www.merckvetmanual.com/infectious-diseases/rickettsial-diseases-in-dogs/ehrlichiosis-in-dogs",
        "verification": "verified_fetched",
    },
    "merck_parvo": {
        "citation": (
            "Merck Veterinary Manual. (2022). Canine parvovirus infection "
            "(parvoviral enteritis in dogs). In Merck Veterinary Manual "
            f"(online ed.). Merck & Co. Retrieved {ACCESS_DATE}, from "
            "https://www.merckvetmanual.com/digestive-system/"
            "infectious-diseases-of-the-gastrointestinal-tract-in-small-animals/"
            "canine-parvovirus-infection-parvoviral-enteritis-in-dogs"
        ),
        "type": "manual",
        "url": "https://www.merckvetmanual.com/digestive-system/infectious-diseases-of-the-gastrointestinal-tract-in-small-animals/canine-parvovirus-infection-parvoviral-enteritis-in-dogs",
        "verification": "verified_fetched",
    },
    "merck_feline_uri": {
        "citation": (
            "Merck Veterinary Manual. (2022). Feline respiratory disease "
            "complex. In Merck Veterinary Manual (online ed.). Merck & Co. "
            f"Retrieved {ACCESS_DATE}, from https://www.merckvetmanual.com/"
            "respiratory-system/respiratory-diseases-of-small-animals/"
            "feline-respiratory-disease-complex"
        ),
        "type": "manual",
        "url": "https://www.merckvetmanual.com/respiratory-system/respiratory-diseases-of-small-animals/feline-respiratory-disease-complex",
        "verification": "verified_fetched",
    },
    "merck_haemoplasma": {
        "citation": (
            "Merck Veterinary Manual. (2022). Hemotropic mycoplasma infections "
            "in animals. In Merck Veterinary Manual (online ed.). Merck & Co. "
            f"Retrieved {ACCESS_DATE}, from https://www.merckvetmanual.com/"
            "circulatory-system/blood-parasites/"
            "hemotropic-mycoplasma-infections-in-animals"
        ),
        "type": "manual",
        "url": "https://www.merckvetmanual.com/circulatory-system/blood-parasites/hemotropic-mycoplasma-infections-in-animals",
        "verification": "verified_fetched",
    },
}

# --------------------------------------------------------------------------- #
# Chunks. text is a faithful paraphrase / close quotation of the fetched source.
# --------------------------------------------------------------------------- #
CHUNKS: list[dict] = [
    # ------------------------- EAST COAST FEVER ------------------------- #
    {
        "doc_id": "vet-kb-ecf-001",
        "title": "East Coast Fever - Aetiology and Transmission",
        "disease": "East Coast Fever",
        "species": ["cattle"],
        "section": "aetiology",
        "source": "merck_theileriosis",
        "text": (
            "East Coast fever is an acute disease of cattle caused by the "
            "protozoan parasite Theileria parva. Sporozoites of T. parva are "
            "injected into cattle by infected vector ticks, principally the "
            "brown ear tick Rhipicephalus appendiculatus, during feeding. The "
            "African buffalo (Syncerus caffer) is an important reservoir host "
            "that carries the parasite without showing clinical signs; strains "
            "maintained in buffalo cause the closely related Corridor disease. "
            "After a tick inoculates sporozoites, the parasite invades "
            "lymphocytes and multiplies as schizonts, later producing "
            "intra-erythrocytic piroplasms. Because ticks acquire infection by "
            "feeding on parasitaemic cattle and transmit it transstadially, "
            "control of the disease is tightly linked to control of the tick "
            "vector. East Coast fever is a serious constraint on cattle "
            "production in eastern, central and southern Africa."
        ),
    },
    {
        "doc_id": "vet-kb-ecf-002",
        "title": "East Coast Fever - Clinical Signs",
        "disease": "East Coast Fever",
        "species": ["cattle"],
        "section": "clinical_signs",
        "source": "merck_theileriosis",
        "text": (
            "Following an incubation period of about 5-10 days after infected "
            "ticks attach, fever develops 7-10 days after parasites are "
            "introduced and can exceed 42 C (107 F). The classic presentation "
            "of East Coast fever is high fever with pronounced generalised "
            "swelling of the superficial lymph nodes (lymphadenopathy), which "
            "is often first noticed in the parotid node draining the ear where "
            "the tick fed. Affected cattle show anorexia and rapid loss of "
            "condition, lacrimation and nasal discharge. In the terminal "
            "stages there is severe respiratory distress (dyspnoea) due to "
            "pulmonary oedema, with frothy exudate at the nostrils. Death "
            "typically occurs 18-24 days after infection. At post-mortem there "
            "is generalised lymph node enlargement, massive pulmonary oedema "
            "and hyperaemia, and haemorrhages on serosal and mucosal surfaces. "
            "Naive animals and introduced exotic breeds suffer the highest "
            "mortality."
        ),
    },
    {
        "doc_id": "vet-kb-ecf-003",
        "title": "East Coast Fever - Diagnosis",
        "disease": "East Coast Fever",
        "species": ["cattle"],
        "section": "diagnosis",
        "source": "merck_theileriosis",
        "text": (
            "Diagnosis of East Coast fever is based on clinical and "
            "epidemiological findings confirmed by demonstrating the parasite. "
            "The primary laboratory method is microscopic examination of "
            "Giemsa-stained needle aspirates (smears) from an enlarged lymph "
            "node, looking for the characteristic multinucleate schizonts "
            "(Koch's blue bodies) inside infected leukocytes. Intra-erythrocytic "
            "piroplasms, which are small and rod-shaped or oval, can be seen in "
            "Giemsa-stained peripheral blood smears and support the diagnosis. "
            "Antigen-specific ELISAs and PCR performed on lymph node aspirates "
            "or blood provide more sensitive and specific confirmation and can "
            "distinguish Theileria species. Serology detects antibodies and is "
            "therefore useful mainly for identifying previously infected or "
            "recovered animals rather than acute cases."
        ),
    },
    {
        "doc_id": "vet-kb-ecf-004",
        "title": "East Coast Fever - Treatment",
        "disease": "East Coast Fever",
        "species": ["cattle"],
        "section": "treatment",
        "source": "merck_theileriosis",
        "text": (
            "Buparvaquone is the only compound available for the treatment of "
            "diseases caused by Theileria parva. It is a hydroxynaphthoquinone "
            "given by intramuscular injection and is most effective when "
            "administered in the early stages of clinical disease; treatment "
            "may require more than one dose. Buparvaquone is less effective in "
            "the advanced stages, when there has been extensive destruction of "
            "lymphoid and haematopoietic tissue, so early recognition and "
            "prompt treatment are critical to survival. Supportive treatment "
            "with anti-inflammatory drugs and, where pulmonary oedema is "
            "present, antidiuretics can improve outcomes. Development of "
            "buparvaquone resistance has been documented in some parasite "
            "strains, underscoring the importance of correct dosing and of "
            "integrating treatment with tick control and immunisation rather "
            "than relying on chemotherapy alone."
        ),
    },
    {
        "doc_id": "vet-kb-ecf-005",
        "title": "East Coast Fever - Prevention and Control",
        "disease": "East Coast Fever",
        "species": ["cattle"],
        "section": "prevention",
        "source": "merck_theileriosis",
        "text": (
            "Prevention of East Coast fever rests on tick control and "
            "immunisation. Spraying or dipping cattle with acaricides is the "
            "most frequently used method of preventing theileriosis, but it "
            "must be applied at regular intervals to remain effective; "
            "pyrethroid compounds are often used where animals face multiple "
            "tick-borne disease challenges. Cattle can be immunised against "
            "T. parva by the infection-and-treatment method, in which a "
            "cryopreserved sporozoite stabilate prepared from homogenised "
            "infected ticks is inoculated together with a simultaneous dose of "
            "long-acting oxytetracycline. The oxytetracycline inhibits "
            "development of the parasite at the outset of infection, allowing "
            "the animal to mount protective immunity without severe disease. "
            "Cattle should be immunised 3-4 weeks before being allowed onto "
            "infected pasture so that immunity is established before challenge."
        ),
    },
    {
        "doc_id": "vet-kb-ecf-006",
        "title": "East Coast Fever - Geographic Distribution in East Africa",
        "disease": "East Coast Fever",
        "species": ["cattle"],
        "section": "epidemiology_east_africa",
        "source": "cfsph_theileriosis",
        "text": (
            "Theileria parva, the cause of East Coast fever and Corridor "
            "disease, occurs in sub-Saharan Africa and is one of the two most "
            "economically important Theileria species of cattle. The parasite "
            "is transmitted by Rhipicephalus ticks (notably R. appendiculatus) "
            "acting as biological vectors; transmission is transstadial and "
            "transovarial transmission is not thought to occur. African buffalo "
            "and cattle are important reservoir hosts, and waterbuck are also "
            "susceptible. Sporozoites are injected in the saliva of a feeding "
            "tick; T. parva ordinarily matures only after an infected tick "
            "attaches to a host and must be attached for a few days before "
            "transmission, although the parasite can develop to the infectious "
            "stage on ticks on the ground when environmental temperatures are "
            "high, allowing transmission within hours of attachment. Ruminants "
            "that recover can remain carriers for months, sustaining endemic "
            "cycles across the East African cattle-keeping regions."
        ),
    },
    {
        "doc_id": "vet-kb-ecf-007",
        "title": "East Coast Fever - ITM Vaccine Control in Kenya",
        "disease": "East Coast Fever",
        "species": ["cattle"],
        "section": "epidemiology_east_africa",
        "source": "patel_2019_itm",
        "text": (
            "East Coast fever is responsible for economic losses of over "
            "US$300 million per year in affected regions, with approximately "
            "40 million cattle exposed. The infection-and-treatment method "
            "(ITM) is the principal live vaccination approach: animals receive "
            "live Theileria parva sporozoites together with a long-acting "
            "oxytetracycline, which prevents the development of severe clinical "
            "disease from the otherwise lethal dose of sporozoites while "
            "protective immunity develops. The widely used Muguga cocktail "
            "combines stabilates from three different T. parva isolates to give "
            "broader cross-protection. A practical constraint for East African "
            "smallholders is that vaccine is packaged at roughly 30-40 doses "
            "per 0.5 mL straw, which is poorly matched to farmers who keep only "
            "a few animals; work on thawing, diluting and refreezing packaged "
            "vaccine aims to make ITM more accessible to smallholder dairy "
            "herds in Kenya and the wider region."
        ),
    },
    # ------------------------- TRYPANOSOMIASIS ------------------------- #
    {
        "doc_id": "vet-kb-tryp-001",
        "title": "African Animal Trypanosomiasis - Aetiology and Transmission",
        "disease": "Trypanosomiasis",
        "species": ["cattle", "dog"],
        "section": "aetiology",
        "source": "merck_trypanosomiasis",
        "text": (
            "African animal trypanosomiasis (nagana) is caused by protozoan "
            "parasites of the genus Trypanosoma. The most important species in "
            "cattle are Trypanosoma congolense, T. vivax and T. brucei brucei; "
            "in dogs, T. brucei is probably the most important. The parasites "
            "are transmitted cyclically by the tsetse fly (genus Glossina), "
            "which is restricted to Africa roughly between latitudes 15 N and "
            "29 S. Different Glossina species occupy distinct habitats: "
            "G. morsitans in savanna, G. palpalis in riverine areas and "
            "G. fusca in forests. When an infected tsetse fly feeds, it "
            "inoculates metacyclic trypomastigotes into the skin, where they "
            "establish a localised reaction (chancre) before spreading to the "
            "lymph nodes and bloodstream. T. vivax can also be transmitted "
            "mechanically by biting flies outside the tsetse belt. The disease "
            "is a major constraint on livestock production across sub-Saharan "
            "Africa."
        ),
    },
    {
        "doc_id": "vet-kb-tryp-002",
        "title": "African Animal Trypanosomiasis - Clinical Signs",
        "disease": "Trypanosomiasis",
        "species": ["cattle", "dog"],
        "section": "clinical_signs",
        "source": "merck_trypanosomiasis",
        "text": (
            "The cardinal clinical signs of African animal trypanosomiasis are "
            "intermittent fever, progressive anaemia and weight loss. Cattle "
            "typically develop a chronic wasting disease; enlarged lymph nodes, "
            "poor body condition, reduced productivity and, in many cases, "
            "eventual death are common, especially where nutrition is poor or "
            "the animals are stressed. Anaemia is the most consistent feature "
            "and correlates with disease severity. Dogs and other susceptible "
            "species may show progressive emaciation, fever, anaemia, oedema "
            "and, if untreated, death. A presumptive clinical diagnosis is "
            "supported by finding an anaemic animal in poor condition in a "
            "tsetse-endemic area. The chronic, relapsing course reflects "
            "antigenic variation by the parasite, which repeatedly evades the "
            "host immune response."
        ),
    },
    {
        "doc_id": "vet-kb-tryp-003",
        "title": "African Animal Trypanosomiasis - Diagnosis",
        "disease": "Trypanosomiasis",
        "species": ["cattle", "dog"],
        "section": "diagnosis",
        "source": "merck_trypanosomiasis",
        "text": (
            "Definitive diagnosis of trypanosomiasis requires demonstrating "
            "trypanosomes, usually by microscopy. The most sensitive rapid "
            "method is to examine a wet mount of the buffy-coat layer of a "
            "microhaematocrit (PCV) tube after centrifugation, looking for "
            "motile parasites; this also allows the packed cell volume to be "
            "measured as an indicator of anaemia. Trypanosomes may also be seen "
            "in Giemsa-stained thick and thin blood smears, and species can be "
            "differentiated on morphology. Because parasitaemia fluctuates and "
            "may be low in chronic infections, molecular methods (PCR) are more "
            "sensitive and allow species-level identification, and serological "
            "antigen- or antibody-detection tests are useful for herd-level "
            "screening. A presumptive diagnosis is often made on the basis of "
            "anaemia and poor condition in an animal from an endemic area."
        ),
    },
    {
        "doc_id": "vet-kb-tryp-004",
        "title": "African Animal Trypanosomiasis - Treatment",
        "disease": "Trypanosomiasis",
        "species": ["cattle", "dog"],
        "section": "treatment",
        "source": "merck_trypanosomiasis",
        "text": (
            "Treatment of animal trypanosomiasis relies on a small number of "
            "trypanocidal drugs. In cattle, diminazene aceturate is used "
            "therapeutically at 3.5-7 mg/kg by intramuscular or subcutaneous "
            "injection; homidium (as the bromide or chloride) is given at about "
            "1 mg/kg IM; and isometamidium chloride at 0.25-1 mg/kg IM is used "
            "both therapeutically and prophylactically, providing protection "
            "for up to several months. In dogs, suramin has been used at about "
            "10 mg/kg IV. Because only these few compounds are available and "
            "have been in use for decades, drug resistance is an important and "
            "growing problem; correct dosing, avoiding under-dosing, and "
            "alternating drug classes (sanative pairs) help to limit "
            "resistance. Chemotherapy is most effective when combined with "
            "tsetse control rather than used alone."
        ),
    },
    {
        "doc_id": "vet-kb-tryp-005",
        "title": "African Animal Trypanosomiasis - Prevention and Control",
        "disease": "Trypanosomiasis",
        "species": ["cattle", "dog"],
        "section": "prevention",
        "source": "merck_trypanosomiasis",
        "text": (
            "Control of African animal trypanosomiasis integrates vector "
            "control with chemoprophylaxis and, where feasible, use of "
            "trypanotolerant livestock breeds. Tsetse populations are reduced "
            "by frequent spraying and dipping of animals with insecticides, "
            "aerial and ground spraying of insecticides over fly-breeding "
            "areas, insecticide-treated targets and traps, and habitat "
            "modification. In high-risk zones, animals can be given prophylactic "
            "trypanocides such as isometamidium to prevent infection. "
            "Large-scale area-wide initiatives include the Sterile Insect "
            "Technique (SIT) and the Pan African Tsetse and Trypanosomiasis "
            "Eradication Campaign (PATTEC). Because tsetse reinvasion and drug "
            "resistance both undermine single-method approaches, sustainable "
            "control depends on combining vector suppression, rational drug "
            "use and herd management."
        ),
    },
    {
        "doc_id": "vet-kb-tryp-006",
        "title": "African Animal Trypanosomiasis - Epidemiology in Kenya",
        "disease": "Trypanosomiasis",
        "species": ["cattle"],
        "section": "epidemiology_east_africa",
        "source": "okello_2022_aat",
        "text": (
            "A cross-sectional survey of cattle in Lambwe Valley, Homa Bay "
            "County, south-western Kenya, a densely forested tsetse-endemic "
            "region adjoining Ruma National Park, found an overall "
            "trypanosome prevalence of 15.63% (71/454) by PCR, compared with "
            "only 3.30% by microscopy, illustrating how insensitive routine "
            "microscopy can be. Trypanosoma vivax was the most prevalent "
            "species at 10.31%, followed by T. congolense Savannah (6.01%), "
            "with smaller proportions of T. congolense Forest (1.78%) and "
            "T. congolense Kilifi (1.12%); no T. brucei was detected. Infection "
            "was strongly associated with clinical signs, proximity to the "
            "national park, and communal grazing (20.00% infected versus 1.60% "
            "under zero-grazing). Herds managed with farmer self-treatment "
            "(27.23% infected) and single-drug therapy had higher infection "
            "rates, highlighting the risks of unsupervised trypanocide use in "
            "the region."
        ),
    },
    # ------------------------- PPR ------------------------- #
    {
        "doc_id": "vet-kb-ppr-001",
        "title": "Peste des Petits Ruminants - Aetiology",
        "disease": "Peste des Petits Ruminants",
        "species": ["goat", "sheep"],
        "section": "aetiology",
        "source": "merck_ppr",
        "text": (
            "Peste des petits ruminants (PPR) is an acute, highly contagious "
            "viral disease of goats and sheep caused by a morbillivirus of the "
            "family Paramyxoviridae, closely related to the (now eradicated) "
            "rinderpest virus. The virus preferentially replicates in lymphoid "
            "tissue and in the epithelium of the gastrointestinal and "
            "respiratory tracts, where it produces the characteristic erosive "
            "and necrotic lesions. PPR virus and rinderpest virus are "
            "cross-protective; the global eradication of rinderpest has removed "
            "the natural cross-immunity that previously protected some small "
            "ruminant populations, contributing to the spread of PPR. "
            "Transmission is mainly by close contact through inhalation of "
            "aerosols and via ocular, nasal and oral secretions and faeces of "
            "infected animals; there is no long-term carrier state, so "
            "continued transmission depends on movement of infected animals "
            "into susceptible flocks."
        ),
    },
    {
        "doc_id": "vet-kb-ppr-002",
        "title": "Peste des Petits Ruminants - Clinical Signs",
        "disease": "Peste des Petits Ruminants",
        "species": ["goat", "sheep"],
        "section": "clinical_signs",
        "source": "merck_ppr",
        "text": (
            "After an incubation period of typically 4-5 days, the acute form "
            "of PPR begins with sudden high fever (104-106 F, about 40-41 C), "
            "dullness, restlessness, a dry muzzle, dull coat and depressed "
            "appetite. A serous nasal and ocular discharge develops and becomes "
            "mucopurulent with a characteristic putrid odour, often crusting "
            "and occluding the nostrils. Necrotic stomatitis produces erosions "
            "on the lips, gums, dental pad and around the incisors. Profuse, "
            "sometimes blood-stained, diarrhoea follows, leading to dehydration "
            "and emaciation. Coughing and respiratory distress from secondary "
            "bronchopneumonia are common in the later stages, and abortion may "
            "occur in pregnant animals. Morbidity and mortality are variable "
            "but can reach 80-100% in some outbreaks, with young animals "
            "suffering higher mortality than adults."
        ),
    },
    {
        "doc_id": "vet-kb-ppr-003",
        "title": "Peste des Petits Ruminants - Diagnosis",
        "disease": "Peste des Petits Ruminants",
        "species": ["goat", "sheep"],
        "section": "diagnosis",
        "source": "merck_ppr",
        "text": (
            "A presumptive diagnosis of PPR is based on the characteristic "
            "clinical picture (fever, oculonasal discharge, stomatitis, "
            "diarrhoea, pneumonia) together with epidemiological findings of "
            "rapid spread and high mortality in goats and sheep. Definitive "
            "laboratory confirmation is required because several differential "
            "diagnoses look similar. Antigen-capture ELISA and "
            "reverse-transcription PCR are the preferred laboratory tests for "
            "detecting virus antigen or nucleic acid, and virus neutralisation "
            "or competitive ELISA on paired serum samples showing a four-fold "
            "or greater rise in titre confirms recent infection. Suitable "
            "specimens for antigen or nucleic acid detection include lymph "
            "nodes, tonsils, spleen and lung collected at necropsy, plus "
            "ocular, nasal and oral swabs from live animals; serum is collected "
            "for antibody testing. PPR is a WOAH-listed notifiable disease."
        ),
    },
    {
        "doc_id": "vet-kb-ppr-004",
        "title": "Peste des Petits Ruminants - Treatment",
        "disease": "Peste des Petits Ruminants",
        "species": ["goat", "sheep"],
        "section": "treatment",
        "source": "merck_ppr",
        "text": (
            "There is no specific antiviral treatment for peste des petits "
            "ruminants. However, treatment directed at the secondary bacterial "
            "and parasitic complications decreases mortality rates in affected "
            "flocks and herds. Supportive care is the mainstay: broad-spectrum "
            "antibiotics to control secondary bacterial pneumonia and enteritis, "
            "fluid and electrolyte therapy to counter dehydration from "
            "diarrhoea, and good nursing, shelter and nutrition. Anthelmintic "
            "and anticoccidial treatment may be indicated where parasitism "
            "compounds the disease. Because supportive treatment does not clear "
            "the virus or prevent spread, sick animals should be isolated and "
            "outbreaks reported and controlled by vaccination and movement "
            "restrictions rather than by treatment of individuals alone."
        ),
    },
    {
        "doc_id": "vet-kb-ppr-005",
        "title": "Peste des Petits Ruminants - Prevention and Control",
        "disease": "Peste des Petits Ruminants",
        "species": ["goat", "sheep"],
        "section": "prevention",
        "source": "merck_ppr",
        "text": (
            "Prevention of PPR is based on vaccination and biosecurity. A live "
            "attenuated PPR vaccine prepared in Vero cell culture affords "
            "protection from natural disease for more than one year (a single "
            "dose is generally considered to confer lifelong immunity in small "
            "ruminants), and homologous PPR vaccines have replaced the earlier "
            "heterologous rinderpest vaccine. In free areas, introduction is "
            "prevented by controlling animal movement, quarantine of "
            "introduced stock, and rapid reporting; when PPR is detected in a "
            "previously free country, eradication by stamping-out, movement "
            "control and ring vaccination is recommended. WOAH and FAO are "
            "pursuing global eradication of PPR by 2030, using mass vaccination "
            "campaigns in endemic regions supported by surveillance."
        ),
    },
    {
        "doc_id": "vet-kb-ppr-006",
        "title": "Peste des Petits Ruminants - Sero-epidemiology in Kenya",
        "disease": "Peste des Petits Ruminants",
        "species": ["goat", "sheep"],
        "section": "epidemiology_east_africa",
        "source": "kihu_2015_ppr",
        "text": (
            "A sero-epidemiological survey in Turkana County, in the arid "
            "pastoral north-west of Kenya, tested serum from 431 sheep and 538 "
            "goats across six administrative divisions. Goats had a "
            "significantly higher apparent PPR sero-positivity of 40% (95% CI "
            "36-44%) than sheep at 32% (95% CI 27-36%). Seroprevalence was "
            "highly heterogeneous geographically, ranging from 22% to 65% "
            "between administrative divisions and from 0% to 78% between "
            "sub-locations, indicating that PPR is established but unevenly "
            "distributed across the pastoral ecosystem. Age, species and "
            "administrative division were associated with sero-positivity. The "
            "high antibody prevalence in unvaccinated age groups is consistent "
            "with endemic circulation of PPR virus in the Turkana pastoral "
            "system, driven by uncontrolled animal movement and shared grazing "
            "and watering points typical of East African pastoralism."
        ),
    },
    # ------------------------- MASTITIS ------------------------- #
    {
        "doc_id": "vet-kb-mast-001",
        "title": "Bovine Mastitis - Aetiology",
        "disease": "Bovine Mastitis",
        "species": ["cattle"],
        "section": "aetiology",
        "source": "merck_mastitis",
        "text": (
            "Bovine mastitis is inflammation of the mammary gland, almost "
            "always caused by intramammary infection following microbial "
            "invasion through the teat canal. Most infections are caused by "
            "various species of streptococci (and similar gram-positive cocci), "
            "staphylococci, and gram-negative rods, especially "
            "lactose-fermenting organisms of enteric origin commonly termed "
            "coliforms. Pathogens are traditionally divided into contagious and "
            "environmental groups. Contagious pathogens - notably "
            "Staphylococcus aureus, Streptococcus agalactiae and "
            "Corynebacterium bovis - spread from cow to cow during milking, "
            "chiefly on milkers' hands and on teatcup liners. Environmental "
            "pathogens, including environmental streptococci and coliforms such "
            "as Escherichia coli, originate from the cow's surroundings; bedding "
            "used to house cattle is the primary source of environmental "
            "pathogens, along with contaminated teat dips, wash water, mud, "
            "skin lesions and flies."
        ),
    },
    {
        "doc_id": "vet-kb-mast-002",
        "title": "Bovine Mastitis - Clinical Signs",
        "disease": "Bovine Mastitis",
        "species": ["cattle"],
        "section": "clinical_signs",
        "source": "merck_mastitis",
        "text": (
            "Mastitis is classified as subclinical or clinical. Subclinical "
            "mastitis is the presence of infection without apparent local "
            "inflammation or systemic involvement, although transient episodes "
            "of abnormal milk may appear; it is detected by raised somatic cell "
            "counts and reduced milk yield, and is far more common than clinical "
            "disease. Chronic infections may persist for two months or longer, "
            "often for the whole lactation, with milk production falling in "
            "proportion to the rise in somatic cell count. Clinical mastitis is "
            "characterised by visibly abnormal milk (discolouration, clots or "
            "flakes of fibrin) and by inflammation of the udder with swelling, "
            "heat, pain and redness of the affected quarter. Cases are graded as "
            "mild (abnormal milk only), moderate (abnormal milk plus udder "
            "inflammation) and severe, in which systemic signs such as fever, "
            "anorexia and shock occur, most often with coliform infection."
        ),
    },
    {
        "doc_id": "vet-kb-mast-003",
        "title": "Bovine Mastitis - Diagnosis and Culture & Sensitivity",
        "disease": "Bovine Mastitis",
        "species": ["cattle"],
        "section": "diagnosis",
        "source": "merck_mastitis",
        "text": (
            "Subclinical mastitis is detected indirectly by measuring somatic "
            "cell count. An SCC of 200,000 cells/mL or more in an individual "
            "cow indicates a high likelihood of infection, and herd bulk-tank "
            "SCCs below 200,000 cells/mL are considered desirable. Cow-side the "
            "California Mastitis Test estimates SCC by the degree of gelling of "
            "milk mixed with reagent, and Dairy Herd Improvement testing "
            "provides monthly individual-cow SCC. Culture of milk samples "
            "collected aseptically from affected quarters is the only reliable "
            "method to determine the aetiology of clinical cases; combined with "
            "antimicrobial susceptibility (culture and sensitivity) testing it "
            "guides rational drug selection. Culture-based decisions also allow "
            "cases yielding no growth, gram-negative organisms or unusual "
            "pathogens to be excluded from antibiotic treatment, which can "
            "reduce antimicrobial use for mild clinical mastitis by 50-75%."
        ),
    },
    {
        "doc_id": "vet-kb-mast-004",
        "title": "Bovine Mastitis - Treatment",
        "disease": "Bovine Mastitis",
        "species": ["cattle"],
        "section": "treatment",
        "source": "merck_mastitis",
        "text": (
            "Treatment of mastitis depends on the pathogen and the severity. "
            "For subclinical Streptococcus agalactiae infection, labelled use "
            "of commercial intramammary products containing amoxicillin, "
            "penicillin or cephalosporins is preferred and achieves cure rates "
            "of 75-90%, using strict aseptic technique for teat-end "
            "preparation. Staphylococcus aureus responds poorly, with "
            "intramammary cure rates of only 20-40% because of deep-seated "
            "infection and antimicrobial resistance. For severe (usually "
            "coliform) clinical mastitis, aggressive supportive care with fluid "
            "and electrolyte therapy is the priority; ceftiofur sodium "
            "(2.2 mg/kg IM every 24 h) can decrease mortality and culling, and "
            "oxytetracycline (11 mg/kg IV every 24 h) may also help. Flunixin "
            "meglumine (1.1-2.2 mg/kg IV) is the only FDA-approved NSAID for "
            "mastitis in cattle and is used to control inflammation and "
            "endotoxaemia. Milk and meat withdrawal periods must be observed "
            "for all antimicrobials used."
        ),
    },
    {
        "doc_id": "vet-kb-mast-005",
        "title": "Bovine Mastitis - Prevention and Control",
        "disease": "Bovine Mastitis",
        "species": ["cattle"],
        "section": "prevention",
        "source": "merck_mastitis",
        "text": (
            "Mastitis control aims to reduce the rate of new intramammary "
            "infections and their duration. The single most important "
            "management practice to prevent transmission of new infections is "
            "the use of an effective germicide as a post-milking teat dip. "
            "Additional measures include wearing gloves, drying each cow's teats "
            "with an individual towel, pre-milking teat disinfection, and "
            "correct milking-machine function and maintenance. Blanket dry-cow "
            "therapy - treating all quarters of all cows at drying off with a "
            "long-acting intramammary product (containing, for example, "
            "penicillin, cloxacillin, cephapirin, ceftiofur or novobiocin) - "
            "has been a foundation of mastitis control for more than 50 years; "
            "internal teat sealants provide an additional physical barrier. "
            "Environmental control matters because bedding is the main source "
            "of environmental pathogens: inorganic bedding such as sand "
            "supports less bacterial growth than organic materials like sawdust "
            "or straw."
        ),
    },
    {
        "doc_id": "vet-kb-mast-006",
        "title": "Bovine Mastitis - Prevalence and Aetiology in Kenya",
        "disease": "Bovine Mastitis",
        "species": ["cattle"],
        "section": "epidemiology_east_africa",
        "source": "mbindyo_2020_mastitis",
        "text": (
            "A cross-sectional study of smallholder dairy cattle in Embu and "
            "Kajiado Counties, Kenya, found an overall mastitis prevalence of "
            "80% (316/395) based on the California Mastitis Test and clinical "
            "examination, comprising 6.8% clinical and 73.1% subclinical "
            "mastitis; the very high subclinical burden is typical of East "
            "African smallholder systems where the disease often goes "
            "undetected. Of 1,016 bacterial isolates from milk, the most "
            "frequent were coagulase-negative staphylococci (42.8%), "
            "Streptococcus species (22.2%) and Staphylococcus aureus (15.7%), "
            "with smaller proportions of Pseudomonas aeruginosa (5.1%) and "
            "Enterobacter species (0.7%). Significant risk factors included not "
            "milking infected cows last, not using an individual udder-drying "
            "towel per cow, and a previous history of mastitis, all of which "
            "are modifiable through improved milking hygiene - underlining that "
            "management interventions are central to mastitis control in Kenyan "
            "dairy herds."
        ),
    },
    # ------------------------- ANAPLASMOSIS ------------------------- #
    {
        "doc_id": "vet-kb-anap-001",
        "title": "Bovine Anaplasmosis - Aetiology and Transmission",
        "disease": "Anaplasmosis",
        "species": ["cattle"],
        "section": "aetiology",
        "source": "merck_anaplasmosis",
        "text": (
            "Clinical bovine anaplasmosis is usually caused by Anaplasma "
            "marginale, a rickettsial parasite that infects red blood cells. "
            "Up to 17 tick species are able to transmit it, including "
            "Dermacentor, Rhipicephalus, Ixodes, Hyalomma and Argas ticks; "
            "Rhipicephalus (Boophilus) spp are major vectors in Australia and "
            "Africa, while Dermacentor spp are the main vectors in the USA. In "
            "addition to tick transmission, the organism can be spread "
            "mechanically by biting flies and by blood-contaminated "
            "instruments such as re-used needles, castration or dehorning "
            "equipment. After infection the prepatent (incubation) period "
            "typically ranges from 15 to 36 days and is directly related to the "
            "infective dose of organisms."
        ),
    },
    {
        "doc_id": "vet-kb-anap-002",
        "title": "Bovine Anaplasmosis - Clinical Signs",
        "disease": "Anaplasmosis",
        "species": ["cattle"],
        "section": "clinical_signs",
        "source": "merck_anaplasmosis",
        "text": (
            "The severity of bovine anaplasmosis is strongly age dependent. In "
            "animals under 1 year old the infection is usually subclinical; in "
            "yearlings and 2-year-olds it is moderately severe; and in older "
            "cattle it is severe and often fatal. The dominant feature is a "
            "progressive anaemia caused by extravascular destruction of both "
            "infected and uninfected erythrocytes. Affected cattle show "
            "inappetence, loss of coordination, breathlessness on exertion and "
            "a rapid, bounding pulse. Unlike babesiosis, haemoglobinuria does "
            "not occur because red-cell destruction is extravascular rather "
            "than intravascular. Jaundice may develop later in the course, and "
            "pregnant cows may abort. Surviving animals become persistently "
            "infected carriers that are a reservoir for further transmission."
        ),
    },
    {
        "doc_id": "vet-kb-anap-003",
        "title": "Bovine Anaplasmosis - Diagnosis",
        "disease": "Anaplasmosis",
        "species": ["cattle"],
        "section": "diagnosis",
        "source": "merck_anaplasmosis",
        "text": (
            "In clinically affected animals, microscopic examination of "
            "Giemsa-stained thin and thick blood films is critical: Anaplasma "
            "marginale appears as dense, rounded inclusion bodies located "
            "towards the margin of infected erythrocytes. Because parasitaemia "
            "falls after the acute phase, carrier animals are difficult to "
            "detect on blood films and are instead identified by serological "
            "testing using the msp5 competitive ELISA, complement fixation, or "
            "card agglutination tests, or by PCR. Anaemia is confirmed by a low "
            "packed cell volume. Differentiation from babesiosis is important "
            "because the two tick-borne diseases share features but "
            "haemoglobinuria is present in babesiosis and absent in "
            "anaplasmosis."
        ),
    },
    {
        "doc_id": "vet-kb-anap-004",
        "title": "Bovine Anaplasmosis - Treatment",
        "disease": "Anaplasmosis",
        "species": ["cattle"],
        "section": "treatment",
        "source": "merck_anaplasmosis",
        "text": (
            "Tetracyclines are the mainstay of treatment. A single "
            "intramuscular injection of long-acting oxytetracycline at a "
            "dosage of 20 mg/kg is commonly used to treat clinical cases. "
            "Imidocarb is also effective, given as the dihydrochloride salt at "
            "1.5 mg/kg SC or as imidocarb dipropionate at 3 mg/kg SC. Severely "
            "anaemic animals must be handled gently and may require a blood "
            "transfusion, because the stress of handling or exertion can cause "
            "sudden death. Antimicrobial withdrawal periods for meat and milk "
            "must be observed. Treatment given early in the course, before the "
            "packed cell volume falls critically, gives the best prognosis; "
            "eliminating the carrier state requires prolonged tetracycline "
            "therapy and is not always achieved."
        ),
    },
    {
        "doc_id": "vet-kb-anap-005",
        "title": "Bovine Anaplasmosis - Prevention and Control",
        "disease": "Anaplasmosis",
        "species": ["cattle"],
        "section": "prevention",
        "source": "merck_anaplasmosis",
        "text": (
            "Control of anaplasmosis combines tick control, hygienic "
            "veterinary practice and, in some regions, vaccination. Mechanical "
            "transmission is reduced by using clean or single-use needles and "
            "disinfecting surgical instruments between animals. In South "
            "Africa, Australia, Israel and South America, deliberate infection "
            "with the less pathogenic Anaplasma centrale has been used as a "
            "live vaccine to give cattle partial protection against A. "
            "marginale. Killed A. marginale vaccines have been used in the past "
            "in the USA but are no longer available. Because recovered animals "
            "remain carriers, endemic stability - where calves are infected "
            "young while relatively resistant - can protect a herd, whereas "
            "introducing naive adult cattle into endemic areas carries a high "
            "risk of severe disease."
        ),
    },
    # ------------------------- BOVINE BABESIOSIS ------------------------- #
    {
        "doc_id": "vet-kb-babc-001",
        "title": "Bovine Babesiosis - Aetiology and Transmission",
        "disease": "Babesiosis",
        "species": ["cattle"],
        "section": "aetiology",
        "source": "merck_babesiosis",
        "text": (
            "Babesia are intra-erythrocytic protozoan parasites of the phylum "
            "Apicomplexa, order Piroplasmida. In cattle the most important "
            "species are Babesia bovis and Babesia bigemina, with B. bovis "
            "being a much more virulent organism than B. bigemina. The main "
            "vectors of B. bigemina and B. bovis are one-host Rhipicephalus "
            "(Boophilus) spp ticks, which are widespread in tropical and "
            "subtropical areas. Transmission within the tick is transovarial, "
            "so infection passes from an infected female tick to her progeny, "
            "which then infect cattle at their next feeding. Within the "
            "vertebrate host the parasites invade and multiply inside red blood "
            "cells, causing their destruction and the clinical syndrome of "
            "redwater fever."
        ),
    },
    {
        "doc_id": "vet-kb-babc-002",
        "title": "Bovine Babesiosis - Clinical Signs",
        "disease": "Babesiosis",
        "species": ["cattle"],
        "section": "clinical_signs",
        "source": "merck_babesiosis",
        "text": (
            "Acute babesiosis generally runs a course of about one week or "
            "less. The first signs are lethargy, weakness, depression and fever "
            "that is frequently 41 C (106 F) or higher. As intravascular "
            "destruction of red blood cells progresses, affected cattle develop "
            "inappetence, anaemia, jaundice and weight loss, and in the final "
            "stages haemoglobinaemia and haemoglobinuria give the "
            "characteristic red-brown urine that gives redwater its name. "
            "Babesia bovis infections may additionally cause nervous signs "
            "(cerebral babesiosis) when parasitised erythrocytes sequester in "
            "brain capillaries. Case-fatality rates can be high in naive or "
            "introduced animals if treatment is not given promptly."
        ),
    },
    {
        "doc_id": "vet-kb-babc-003",
        "title": "Bovine Babesiosis - Diagnosis",
        "disease": "Babesiosis",
        "species": ["cattle"],
        "section": "diagnosis",
        "source": "merck_babesiosis",
        "text": (
            "Examination of Giemsa-stained blood or organ smears by light "
            "microscopy is essential to confirm babesiosis; the paired, "
            "pear-shaped piroplasms are seen within red blood cells. For "
            "Babesia bovis, which sequesters in deep capillaries, smears "
            "prepared from capillary blood (for example from the ear or "
            "tail-tip) are more likely to reveal the parasite than smears from "
            "large vessels. Molecular methods such as PCR assays are more "
            "sensitive than light microscopy, especially for detecting "
            "low-level carrier infections, and serologic antibody tests are "
            "used for epidemiological surveys and to screen carrier animals."
        ),
    },
    {
        "doc_id": "vet-kb-babc-004",
        "title": "Bovine Babesiosis - Treatment",
        "disease": "Babesiosis",
        "species": ["cattle"],
        "section": "treatment",
        "source": "merck_babesiosis",
        "text": (
            "The two principal babesiacidal drugs are diminazene and "
            "imidocarb. Diminazene is administered at 3.5 mg/kg IM once. "
            "Imidocarb is administered at 1.2 mg/kg SC once for treatment; at a "
            "higher dosage of 3 mg/kg it provides protection from babesiosis "
            "for approximately 4 weeks and can therefore be used "
            "prophylactically. Supportive care with anti-inflammatory drugs, "
            "corticosteroids and fluid therapy is valuable, and blood "
            "transfusions may be life-saving in very anaemic animals. As with "
            "other production animals, drug withdrawal periods must be "
            "respected, and severely affected cattle should be handled with "
            "minimal stress."
        ),
    },
    {
        "doc_id": "vet-kb-babc-005",
        "title": "Bovine Babesiosis - Prevention and Control",
        "disease": "Babesiosis",
        "species": ["cattle"],
        "section": "prevention",
        "source": "merck_babesiosis",
        "text": (
            "Prevention of bovine babesiosis centres on tick control, host "
            "resistance and vaccination. Acaricides reduce tick numbers but "
            "cannot be relied upon to prevent transmission completely, and "
            "excessive dipping can undermine the endemic stability that "
            "protects herds. Bos indicus and Bos indicus-cross cattle are more "
            "resistant to ticks and to clinical babesiosis than Bos taurus "
            "breeds. Vaccination using live attenuated strains of the Babesia "
            "parasites has been used successfully in countries such as "
            "Argentina, Australia, Brazil, Israel, South Africa and Uruguay to "
            "immunise young or introduced cattle before exposure."
        ),
    },
    # ------------------------- LUMPY SKIN DISEASE ------------------------- #
    {
        "doc_id": "vet-kb-lsd-001",
        "title": "Lumpy Skin Disease - Aetiology and Transmission",
        "disease": "Lumpy Skin Disease",
        "species": ["cattle"],
        "section": "aetiology",
        "source": "merck_lsd",
        "text": (
            "Lumpy skin disease is caused by lumpy skin disease virus (LSDV), a "
            "double-stranded DNA virus of the genus Capripoxvirus (the same "
            "genus as sheeppox and goatpox virus). LSDV is thought to be "
            "transmitted mainly mechanically by biting insects, including "
            "biting flies, midges and mosquitoes, so outbreaks are associated "
            "with warm, humid weather that favours these blood-feeding "
            "arthropods. There is evidence of transstadial and even "
            "transovarial survival of the virus in some ticks, the virus can "
            "often be found in the oral secretions of infected cattle, and "
            "there is evidence of emerging LSDV strains that are more "
            "transmissible without the need for vectors. The disease was "
            "originally limited to southern and eastern Africa but has since "
            "spread to the Middle East, Asia, and eastern and southern Europe."
        ),
    },
    {
        "doc_id": "vet-kb-lsd-002",
        "title": "Lumpy Skin Disease - Clinical Signs",
        "disease": "Lumpy Skin Disease",
        "species": ["cattle"],
        "section": "clinical_signs",
        "source": "merck_lsd",
        "text": (
            "Lumpy skin disease begins with fever, lacrimation, nasal "
            "discharge and hypersalivation, followed by the characteristic "
            "eruption of skin nodules that are well circumscribed, round, "
            "slightly raised, firm and painful. Nodules can also form on the "
            "mucous membranes of the trachea, lungs and abomasum. The disease "
            "is associated with decreased milk yield, mastitis, sometimes "
            "abortion, infertility in bulls, decreased appetite and pneumonia. "
            "Morbidity can reach almost 100% in susceptible herds, while the "
            "mortality rate is usually less than 10% but has reached as high as "
            "80% in some outbreaks (for example in India). Secondary bacterial "
            "infection of ruptured nodules can cause extensive suppuration and "
            "sloughing."
        ),
    },
    {
        "doc_id": "vet-kb-lsd-003",
        "title": "Lumpy Skin Disease - Diagnosis",
        "disease": "Lumpy Skin Disease",
        "species": ["cattle"],
        "section": "diagnosis",
        "source": "merck_lsd",
        "text": (
            "A presumptive diagnosis of lumpy skin disease is based on the "
            "characteristic generalised skin nodules together with fever and "
            "the epidemiological setting. Laboratory confirmation uses PCR "
            "assay to detect capripoxvirus DNA, virus isolation, and "
            "histological evaluation of nodule biopsies. It is important to "
            "differentiate true lumpy skin disease from pseudo-lumpy skin "
            "disease caused by bovine herpesvirus 2, which produces more "
            "superficial lesions; the distinction is confirmed by virus "
            "isolation and/or PCR assay. Because LSD is a notifiable "
            "transboundary disease, suspicion should trigger prompt reporting "
            "to veterinary authorities."
        ),
    },
    {
        "doc_id": "vet-kb-lsd-004",
        "title": "Lumpy Skin Disease - Treatment and Control",
        "disease": "Lumpy Skin Disease",
        "species": ["cattle"],
        "section": "treatment",
        "source": "merck_lsd",
        "text": (
            "There is no specific antiviral treatment for lumpy skin disease; "
            "management is supportive and aimed at the skin lesions and "
            "secondary complications. Affected animals are given supportive "
            "care, and secondary bacterial infection of ruptured nodules, which "
            "can cause extensive suppuration and sloughing, is treated with "
            "antibiotics and wound care; severely affected animals may be "
            "euthanised. Because treatment cannot cure the viral infection or "
            "prevent spread, control depends on movement restriction, vector "
            "control and vaccination rather than on treating individual "
            "animals."
        ),
    },
    {
        "doc_id": "vet-kb-lsd-005",
        "title": "Lumpy Skin Disease - Prevention and Vaccination",
        "disease": "Lumpy Skin Disease",
        "species": ["cattle"],
        "section": "prevention",
        "source": "merck_lsd",
        "text": (
            "Prevention of lumpy skin disease relies on vaccination, hygiene, "
            "quarantine and, in previously free regions, stamping-out with "
            "emergency vaccination. Live attenuated LSDV vaccine provides the "
            "best protection, although there are concerns about reversion to "
            "virulence and evidence of vaccine-strain spread. Live attenuated "
            "sheeppox and goatpox virus vaccines are slightly less protective "
            "in cattle but are possibly safer, and killed homologous vaccines, "
            "although less effective, can provide reasonable protection while "
            "removing the risk of reversion. Some vaccines permit DIVA "
            "(differentiating infected from vaccinated animals) testing, which "
            "supports surveillance and trade. Vector control and restriction of "
            "animal movement complement vaccination during outbreaks."
        ),
    },
    # ------------------------- CCPP ------------------------- #
    {
        "doc_id": "vet-kb-ccpp-001",
        "title": "Contagious Caprine Pleuropneumonia - Aetiology",
        "disease": "Contagious Caprine Pleuropneumonia",
        "species": ["goat", "sheep"],
        "section": "aetiology",
        "source": "merck_ccpp",
        "text": (
            "Contagious caprine pleuropneumonia (CCPP) is a severe, highly "
            "contagious respiratory disease of goats caused by Mycoplasma "
            "capricolum subsp. capripneumoniae (Mccp). It occurs in areas of "
            "Africa, Asia and the Middle East. Goats are the primary host, "
            "although infection of sheep and wild ruminants has been "
            "documented. Transmission is by aerosol droplets between animals in "
            "close contact, so introduction of infected or carrier goats into a "
            "naive flock can trigger explosive outbreaks. CCPP is a WOAH-listed "
            "disease of major economic importance for goat-keeping communities, "
            "including pastoralists in East Africa."
        ),
    },
    {
        "doc_id": "vet-kb-ccpp-002",
        "title": "Contagious Caprine Pleuropneumonia - Clinical Signs",
        "disease": "Contagious Caprine Pleuropneumonia",
        "species": ["goat", "sheep"],
        "section": "clinical_signs",
        "source": "merck_ccpp",
        "text": (
            "Clinical CCPP presents with weakness, anorexia, coughing, rapid "
            "breathing (tachypnoea) and nasal discharge, accompanied by fever "
            "of 40.5-41.5 C (104.5-106 F). Exercise intolerance progresses to "
            "marked respiratory distress with open-mouth breathing, an extended "
            "neck and frothy salivation as the pleuropneumonia advances. In "
            "naive herds the disease is devastating: morbidity is often 100% "
            "and mortality may reach 80%. Post-mortem findings are dominated by "
            "unilateral fibrinous pleuropneumonia with straw-coloured pleural "
            "fluid and hepatised lung, reflecting the pleural localisation of "
            "the mycoplasma."
        ),
    },
    {
        "doc_id": "vet-kb-ccpp-003",
        "title": "Contagious Caprine Pleuropneumonia - Diagnosis and Treatment",
        "disease": "Contagious Caprine Pleuropneumonia",
        "species": ["goat", "sheep"],
        "section": "diagnosis",
        "source": "merck_ccpp",
        "text": (
            "Diagnosis of CCPP has been greatly facilitated by PCR assay, which "
            "can be performed directly on pleural fluid or affected lung to "
            "detect Mycoplasma capricolum subsp. capripneumoniae. Bacterial "
            "culture is also used but requires special media and extended "
            "incubation times, and the organism is fastidious and slow-growing. "
            "The characteristic clinical picture of acute fibrinous "
            "pleuropneumonia with very high morbidity in goats supports a "
            "presumptive diagnosis pending laboratory confirmation, which is "
            "important because CCPP must be differentiated from other causes of "
            "caprine respiratory disease and is notifiable."
        ),
    },
    {
        "doc_id": "vet-kb-ccpp-004",
        "title": "Contagious Caprine Pleuropneumonia - Treatment and Prevention",
        "disease": "Contagious Caprine Pleuropneumonia",
        "species": ["goat", "sheep"],
        "section": "treatment",
        "source": "merck_ccpp",
        "text": (
            "CCPP responds to antimicrobial therapy if given early. Tylosin at "
            "10 mg/kg IM every 24 hours for 3 days, or long-acting "
            "oxytetracycline at 20 mg/kg IM once, is effective in the treatment "
            "of CCPP. Prevention and control rest on biosecurity and "
            "vaccination: quarantine of affected flocks and strict biosecurity "
            "protocols for the introduction of new animals are necessary to "
            "prevent spread, and vaccines are available in some countries, with "
            "good to excellent protection reported. Because carrier animals "
            "sustain the infection, movement control and vaccination are more "
            "reliable than treatment alone for herd-level control."
        ),
    },
    # ------------------------- COCCIDIOSIS ------------------------- #
    {
        "doc_id": "vet-kb-cocc-001",
        "title": "Coccidiosis - Aetiology and Transmission",
        "disease": "Coccidiosis",
        "species": ["cattle", "goat", "sheep"],
        "section": "aetiology",
        "source": "merck_coccidiosis",
        "text": (
            "Coccidiosis is caused by obligate intracellular protozoan "
            "parasites of the genus Eimeria (class Conoidasida, phylum "
            "Apicomplexa) that infect the intestinal epithelium of ruminants "
            "and other hosts. Infection is acquired orally through ingestion of "
            "infective sporulated oocysts from a contaminated environment. "
            "Oocysts shed in the faeces require moisture, oxygen and warmth to "
            "sporulate and can remain viable in the environment for a year or "
            "more, although they survive poorly at temperatures below about "
            "30 C or above 40 C. Young animals are most susceptible, and "
            "crowded, damp, faecally contaminated housing favours heavy "
            "infection."
        ),
    },
    {
        "doc_id": "vet-kb-cocc-002",
        "title": "Coccidiosis - Clinical Signs and Diagnosis",
        "disease": "Coccidiosis",
        "species": ["cattle", "goat", "sheep"],
        "section": "clinical_signs",
        "source": "merck_coccidiosis",
        "text": (
            "The main clinical sign of coccidiosis is diarrhoea, ranging from "
            "mild diarrhoea with decreased growth to severe dysentery with "
            "dehydration and tenesmus (straining). Chronic infections cause "
            "pasty faeces, staring coats and poor growth, while the overall "
            "mortality rate is generally low. Diagnosis is based on clinical "
            "signs supported by faecal examination: oocysts are identified in "
            "faeces using salt or sugar flotation methods, direct intestinal "
            "smears, or a McMaster counting chamber. A count of more than 5,000 "
            "oocysts per gram in an animal with compatible clinical signs is "
            "suggestive of clinical coccidiosis, though counts must be "
            "interpreted alongside clinical findings because healthy animals "
            "can shed oocysts."
        ),
    },
    {
        "doc_id": "vet-kb-cocc-003",
        "title": "Coccidiosis - Treatment",
        "disease": "Coccidiosis",
        "species": ["cattle", "goat", "sheep"],
        "section": "treatment",
        "source": "merck_coccidiosis",
        "text": (
            "Several anticoccidial drugs are used to treat clinical "
            "coccidiosis. Sulfonamides in soluble form are commonly "
            "administered orally to calves and other animals with clinical "
            "disease. Amprolium, a thiamine (vitamin B1) antagonist, is given "
            "orally in the drinking water to calves with clinical coccidiosis. "
            "Decoquinate, a 4-hydroxyquinolone coccidiostat, is licensed in "
            "some countries for in-feed treatment of cattle and sheep, and "
            "toltrazuril is approved for oral use in cattle, goats, pigs and "
            "sheep. (The Merck overview does not specify mg/kg doses for these "
            "products, which vary by formulation and country of registration.) "
            "Supportive fluid therapy is important in severely affected, "
            "dehydrated animals."
        ),
    },
    {
        "doc_id": "vet-kb-cocc-004",
        "title": "Coccidiosis - Prevention and Control",
        "disease": "Coccidiosis",
        "species": ["cattle", "goat", "sheep"],
        "section": "prevention",
        "source": "merck_coccidiosis",
        "text": (
            "Control of coccidiosis is based on limiting the intake of "
            "sporulated oocysts by young animals so that protective immunity "
            "is induced without clinical disease. Animals should be kept in "
            "clean, dry quarters, and feeding and watering devices should be "
            "clean and protected from faecal contamination. Ionophores such as "
            "lasalocid and monensin are widely used for prevention in young "
            "ruminants; continuous low-level feeding of lasalocid or monensin "
            "during the first month of feedlot confinement is reported to be "
            "preventive. Reducing stocking density and avoiding wet, "
            "contaminated bedding further lowers the infection pressure."
        ),
    },
    # ------------------------- HAEMONCHOSIS ------------------------- #
    {
        "doc_id": "vet-kb-haem-001",
        "title": "Haemonchosis - Aetiology and Clinical Signs",
        "disease": "Haemonchosis",
        "species": ["goat", "sheep"],
        "section": "aetiology",
        "source": "merck_gi_smallrum",
        "text": (
            "Haemonchus contortus, the barber's pole worm, is the most common "
            "and most pathogenic gastrointestinal nematode of small ruminants, "
            "prevalent in tropical and subtropical regions and in areas with "
            "summer rainfall - conditions typical of much of East Africa. It is "
            "a blood-feeding parasite of the abomasum. Disease is classified as "
            "hyperacute, in which death may occur within one week of a heavy "
            "infection without notable clinical signs; acute, characterised by "
            "severe anaemia accompanied by generalised oedema (including "
            "submandibular oedema, or bottle jaw); and chronic, with anaemia "
            "and progressive weight loss. Importantly, diarrhoea is not a "
            "characteristic sign of pure Haemonchus infection. A periparturient "
            "rise in worm burden means mature ewes and does may develop fatal "
            "infections in late pregnancy and early lactation."
        ),
    },
    {
        "doc_id": "vet-kb-haem-002",
        "title": "Haemonchosis - Diagnosis and FAMACHA",
        "disease": "Haemonchosis",
        "species": ["goat", "sheep"],
        "section": "diagnosis",
        "source": "merck_gi_smallrum",
        "text": (
            "Because the dominant effect of Haemonchus contortus is blood loss, "
            "the FAMACHA scoring system is an important tool in the diagnosis, "
            "monitoring and control of haemonchosis in flocks of goats and "
            "sheep. FAMACHA evaluates the degree of pallor of the ocular "
            "(conjunctival) mucous membranes as an indicator of anaemia, "
            "matching the colour to a numerical scale in which higher scores "
            "reflect paler membranes and more severe anaemia. Faecal egg counts "
            "(for example by the McMaster method) support the diagnosis and are "
            "used to monitor worm burdens and the efficacy of treatment. "
            "Combining FAMACHA scoring with selective treatment markedly "
            "decreases dewormer use while maintaining production."
        ),
    },
    {
        "doc_id": "vet-kb-haem-003",
        "title": "Haemonchosis - Treatment and Anthelmintic Resistance",
        "disease": "Haemonchosis",
        "species": ["goat", "sheep"],
        "section": "treatment",
        "source": "merck_gi_overview",
        "text": (
            "Treatment of haemonchosis uses anthelmintics from three broad "
            "classes: benzimidazoles (for example albendazole, fenbendazole, "
            "oxfendazole and thiabendazole), the imidazothiazole levamisole, "
            "and the macrocyclic lactones (ivermectin, doramectin, moxidectin "
            "and eprinomectin). Fenbendazole has a wide margin of safety "
            "whereas albendazole and levamisole have narrower therapeutic "
            "indices. A critical problem is that resistance of gastrointestinal "
            "nematodes to every anthelmintic class has been documented, and "
            "susceptibility is generally not regained even when a drug is "
            "withheld for several years on a farm. Because rotational deworming "
            "perpetuates the evolution of multidrug resistance, it is no longer "
            "recommended; instead, targeted selective treatment guided by "
            "FAMACHA and faecal egg counts is advised."
        ),
    },
    {
        "doc_id": "vet-kb-haem-004",
        "title": "Haemonchosis - Prevention and Refugia-Based Control",
        "disease": "Haemonchosis",
        "species": ["goat", "sheep"],
        "section": "prevention",
        "source": "merck_gi_overview",
        "text": (
            "Sustainable control of Haemonchus contortus aims to slow "
            "anthelmintic resistance by preserving refugia - the proportion of "
            "the parasite population not exposed to a dewormer, which dilutes "
            "resistant alleles with susceptible ones. Practical approaches "
            "include targeted selective treatment (treating only anaemic "
            "animals identified by FAMACHA), selective non-treatment, and "
            "whole-herd targeted treatment. Non-chemical measures reduce "
            "reliance on drugs: copper oxide wire particles given as a bolus "
            "decrease Haemonchus burdens in sheep and goats (with caution about "
            "copper toxicity), the nematode-trapping fungus Duddingtonia "
            "flagrans fed continuously lowers worm burdens and faecal egg "
            "counts, and condensed-tannin forages act as an adjunct. Pasture "
            "management - resting pastures, keeping forage above about 10 cm, "
            "and mixed-species grazing - further limits larval intake."
        ),
    },
    # ------------------------- CANINE TRYPANOSOMIASIS ------------------------- #
    {
        "doc_id": "vet-kb-ctryp-001",
        "title": "Canine Trypanosomiasis - Aetiology and Management",
        "disease": "Canine Trypanosomiasis",
        "species": ["dog"],
        "section": "aetiology",
        "source": "merck_trypanosomiasis",
        "text": (
            "Dogs are susceptible to African animal trypanosomiasis; among the "
            "tsetse-transmitted trypanosomes, Trypanosoma brucei is probably "
            "the most important species in dogs. As in other hosts, infected "
            "tsetse flies (genus Glossina) inoculate metacyclic trypomastigotes "
            "when they feed, and the parasites then spread through the lymphatic "
            "system and bloodstream. Affected dogs typically show intermittent "
            "fever, progressive anaemia, weight loss and, if untreated, death. "
            "Diagnosis relies on demonstrating trypanosomes microscopically - "
            "most sensitively in a wet mount of the centrifuged buffy coat - "
            "supported by PCR where available. Suramin has been used in dogs at "
            "about 10 mg/kg IV; because the drug options are limited and "
            "resistance is a concern, treatment should be combined with tsetse "
            "and biting-fly control."
        ),
    },
    # ------------------------- CANINE BABESIOSIS ------------------------- #
    {
        "doc_id": "vet-kb-cbab-001",
        "title": "Canine Babesiosis - Aetiology and Vectors",
        "disease": "Canine Babesiosis",
        "species": ["dog"],
        "section": "aetiology",
        "source": "merck_babesiosis",
        "text": (
            "Canine babesiosis is caused by intra-erythrocytic Babesia "
            "protozoa. The large-form species affecting dogs include Babesia "
            "canis, B. vogeli and B. rossi, while B. gibsoni is a much smaller "
            "parasite. The vectors differ by species and region: Babesia canis "
            "is transmitted by Dermacentor reticulatus in Europe, B. vogeli by "
            "the brown dog tick Rhipicephalus sanguineus in tropical and "
            "subtropical countries, and B. rossi by Haemaphysalis elliptica in "
            "South Africa. Because R. sanguineus is widespread in Africa, "
            "B. vogeli is the form most likely to be encountered in East "
            "African dogs. Infection can also be spread by dog-fighting (bite "
            "wounds) and by blood transfusion."
        ),
    },
    {
        "doc_id": "vet-kb-cbab-002",
        "title": "Canine Babesiosis - Clinical Signs and Diagnosis",
        "disease": "Canine Babesiosis",
        "species": ["dog"],
        "section": "clinical_signs",
        "source": "merck_babesiosis",
        "text": (
            "Babesiosis is characterised by fever and intravascular haemolysis "
            "leading to progressive anaemia, haemoglobinuria and jaundice; the "
            "clinical consequences vary from a mild, transient illness to acute "
            "disease that rapidly results in death. Babesia gibsoni "
            "characteristically causes a chronic disease with progressive, "
            "severe anaemia. Diagnosis is confirmed by examination of "
            "Giemsa-stained blood or organ smears by light microscopy, in which "
            "the piroplasms are seen within red cells; capillary blood smears "
            "improve detection. Molecular methods such as PCR assays are more "
            "sensitive than light microscopy and allow species identification, "
            "and serologic tests can detect antibodies to Babesia spp in "
            "carrier animals."
        ),
    },
    {
        "doc_id": "vet-kb-cbab-003",
        "title": "Canine Babesiosis - Treatment and Prevention",
        "disease": "Canine Babesiosis",
        "species": ["dog"],
        "section": "treatment",
        "source": "merck_babesiosis",
        "text": (
            "The babesiacidal drugs used against Babesia infection are "
            "diminazene aceturate, given at 3.5 mg/kg IM once, and imidocarb "
            "dipropionate, given at 1.2 mg/kg SC once for treatment (a 3 mg/kg "
            "dose confers about 4 weeks of protection). Supportive care with "
            "anti-inflammatory drugs, corticosteroids and fluid therapy is "
            "important, and blood transfusions may be life-saving in very "
            "anaemic dogs. Note that these doses are given on the general "
            "'Babesiosis in Animals' reference and are not specific to a single "
            "canine Babesia species; small-form B. gibsoni in particular often "
            "responds poorly to imidocarb and may require combination protocols "
            "under veterinary supervision. Prevention centres on tick control "
            "with acaricides and prompt removal of attached ticks."
        ),
    },
    # ------------------------- CANINE EHRLICHIOSIS ------------------------- #
    {
        "doc_id": "vet-kb-ehrl-001",
        "title": "Canine Ehrlichiosis - Aetiology and Transmission",
        "disease": "Ehrlichiosis",
        "species": ["dog"],
        "section": "aetiology",
        "source": "merck_ehrlichiosis",
        "text": (
            "Canine monocytic ehrlichiosis is caused by Ehrlichia canis, a "
            "rickettsial organism that infects host monocytes. It is "
            "transmitted mainly by the brown dog tick, Rhipicephalus "
            "sanguineus, which is widely distributed in tropical and "
            "subtropical regions including East Africa, and can also be spread "
            "through blood transfusions. After transmission the organism "
            "replicates within mononuclear cells and disseminates through the "
            "lymphatic and vascular systems, producing an acute phase 1-3 weeks "
            "after infection that may resolve or progress to a debilitating "
            "chronic phase."
        ),
    },
    {
        "doc_id": "vet-kb-ehrl-002",
        "title": "Canine Ehrlichiosis - Clinical Signs",
        "disease": "Ehrlichiosis",
        "species": ["dog"],
        "section": "clinical_signs",
        "source": "merck_ehrlichiosis",
        "text": (
            "In the acute phase, 1-3 weeks after infection, dogs show fever, "
            "bleeding tendencies (petechiae, ecchymoses, melaena, epistaxis), "
            "splenomegaly, lymphadenomegaly and sometimes neurological signs. "
            "Many dogs recover or enter a subclinical phase, but some progress "
            "to a chronic form characterised by bone-marrow hypoplasia and "
            "deposition of immune complexes in various organs, producing "
            "anterior uveitis, polymyositis, vasculitis and glomerulonephritis. "
            "Physical examination findings include fever, splenomegaly, "
            "lymphadenomegaly, bleeding, pitting oedema, pain and neurological "
            "abnormalities. Chronic ehrlichiosis with pancytopenia carries a "
            "guarded prognosis."
        ),
    },
    {
        "doc_id": "vet-kb-ehrl-003",
        "title": "Canine Ehrlichiosis - Diagnosis and Treatment",
        "disease": "Ehrlichiosis",
        "species": ["dog"],
        "section": "diagnosis",
        "source": "merck_ehrlichiosis",
        "text": (
            "Diagnosis of canine ehrlichiosis combines cytological examination "
            "(occasionally revealing morulae in monocytes), serological testing "
            "and PCR assay. A four-fold or greater increase in antibody titres "
            "is consistent with a diagnosis of ehrlichiosis, and PCR enables "
            "speciation of the organism and allows earlier diagnosis before "
            "antibodies develop. Treatment is with doxycycline (5 mg/kg every "
            "12 hours or 10 mg/kg every 24 hours, PO or IV, for 28 days), which "
            "is recommended for dogs of all ages; minocycline (5-10 mg/kg PO "
            "every 12 hours for 28 days) is an alternative. Supportive care is "
            "provided for fever, bleeding and organ dysfunction, and severely "
            "anaemic or thrombocytopenic dogs may need a transfusion."
        ),
    },
    {
        "doc_id": "vet-kb-ehrl-004",
        "title": "Canine Ehrlichiosis - Prevention",
        "disease": "Ehrlichiosis",
        "species": ["dog"],
        "section": "prevention",
        "source": "merck_ehrlichiosis",
        "text": (
            "Prevention of ehrlichiosis depends on tick control, because no "
            "vaccine is available. Effective measures include topical or "
            "systemic acaricidal products containing fipronil, an isoxazoline, "
            "a pyrethroid or amitraz, restricting dogs' access to "
            "heavily tick-infested areas, and careful, prompt removal of "
            "attached ticks (which reduces transmission because ticks must feed "
            "for a period before transmitting the organism). Screening blood "
            "donors helps prevent transfusion-associated infection. Consistent "
            "year-round tick prevention is especially important in warm regions "
            "where Rhipicephalus sanguineus is active throughout the year."
        ),
    },
    # ------------------------- CANINE PARVOVIRUS ------------------------- #
    {
        "doc_id": "vet-kb-cpv-001",
        "title": "Canine Parvovirus - Aetiology and Transmission",
        "disease": "Canine Parvovirus",
        "species": ["dog"],
        "section": "aetiology",
        "source": "merck_parvo",
        "text": (
            "Canine parvovirus (CPV) is a non-enveloped, single-stranded DNA "
            "virus believed to have originated from feline panleukopenia virus. "
            "Antigenic variants circulate; in North America CPV-2b causes most "
            "clinical disease, though CPV-2c is increasingly common. "
            "Transmission occurs by direct oral or nasal contact with "
            "virus-containing faeces or indirectly through contact with "
            "virus-contaminated fomites, which is important because the virus "
            "is extremely hardy in the environment. Infected dogs begin "
            "shedding virus 4-5 days after exposure, throughout the illness, "
            "and for about 10 days after clinical recovery. Unvaccinated "
            "puppies between weaning and about 6 months of age are at greatest "
            "risk."
        ),
    },
    {
        "doc_id": "vet-kb-cpv-002",
        "title": "Canine Parvovirus - Clinical Signs and Diagnosis",
        "disease": "Canine Parvovirus",
        "species": ["dog"],
        "section": "clinical_signs",
        "source": "merck_parvo",
        "text": (
            "Clinical signs of parvoviral enteritis appear within 5-7 days of "
            "infection (range 2-14 days), beginning with lethargy, anorexia and "
            "fever and progressing to vomiting and haemorrhagic small-bowel "
            "diarrhoea within 24-48 hours; about a quarter of dogs have "
            "non-haemorrhagic diarrhoea. Examination reveals depression, fever, "
            "dehydration and fluid-dilated intestinal loops. Diagnosis is "
            "suspected from the signalment, history and clinical signs and "
            "confirmed by commercial faecal antigen ELISA or by PCR; "
            "characteristic laboratory findings include leukopenia, lymphopenia "
            "and neutropenia together with electrolyte abnormalities. Prompt "
            "recognition is important because early aggressive care greatly "
            "improves survival."
        ),
    },
    {
        "doc_id": "vet-kb-cpv-003",
        "title": "Canine Parvovirus - Treatment",
        "disease": "Canine Parvovirus",
        "species": ["dog"],
        "section": "treatment",
        "source": "merck_parvo",
        "text": (
            "Treatment of parvoviral enteritis is intensive supportive care. "
            "Intravenous balanced electrolyte solutions correct dehydration, "
            "replace ongoing losses and provide maintenance needs. Antiemetics "
            "used include maropitant (1 mg/kg IV or SC every 24 hours as "
            "needed), ondansetron (0.5 mg/kg IV every 8 hours as needed) and "
            "metoclopramide (0.2-0.5 mg/kg IM or SC every 6-8 hours as needed). "
            "Because of the risk of sepsis from bacterial translocation across "
            "the damaged gut, broad-spectrum antibiotics are given, such as a "
            "beta-lactam (ampicillin sodium 22 mg/kg IV every 8 hours), often "
            "with enrofloxacin (10-20 mg/kg IV every 24 hours) or gentamicin. "
            "Early enteral nutrition is recommended. With appropriate supportive "
            "care 70-90% of dogs with parvoviral enteritis survive."
        ),
    },
    {
        "doc_id": "vet-kb-cpv-004",
        "title": "Canine Parvovirus - Prevention and Vaccination",
        "disease": "Canine Parvovirus",
        "species": ["dog"],
        "section": "prevention",
        "source": "merck_parvo",
        "text": (
            "Prevention of canine parvovirus rests on vaccination and "
            "environmental control. Vaccination with a modified-live virus "
            "vaccine is recommended at 6-8, 10-12 and 14-16 weeks of age, "
            "followed by a booster one year later and then every 3 years; "
            "completing the puppy series is essential because maternal "
            "antibody can block earlier doses. Inactivated rather than "
            "modified-live vaccines are indicated in pregnant bitches or "
            "colostrum-deprived puppies vaccinated before 6-8 weeks of age. The "
            "virus is highly resistant, persisting indoors for two months or "
            "more and outdoors potentially for years, so contaminated "
            "environments are disinfected with dilute bleach (1:30) or a "
            "peroxygen or accelerated-hydrogen-peroxide disinfectant, and "
            "confirmed or suspected cases are strictly isolated."
        ),
    },
    # ------------------------- FELINE UPPER RESPIRATORY COMPLEX --------------- #
    {
        "doc_id": "vet-kb-furd-001",
        "title": "Feline Upper Respiratory Complex - Aetiology",
        "disease": "Feline Upper Respiratory Complex",
        "species": ["cat"],
        "section": "aetiology",
        "source": "merck_feline_uri",
        "text": (
            "Feline respiratory disease complex is a multifactorial upper "
            "respiratory syndrome of cats. The principal component infections "
            "are feline viral rhinotracheitis (FVR, caused by feline "
            "herpesvirus type 1), feline calicivirus (FCV), Chlamydia felis and "
            "Mycoplasma felis, with Bordetella bronchiseptica and reoviruses as "
            "additional agents. The pathogens spread via aerosol droplets and "
            "fomites, so overcrowding, poor ventilation and stress in "
            "multi-cat environments promote outbreaks. Calicivirus is shed "
            "continually by infected cats, whereas infectious FVR virus is "
            "released intermittently, particularly during stress; the "
            "incubation period is 2-6 days for FVR and FCV and 5-10 days for "
            "chlamydial pneumonitis."
        ),
    },
    {
        "doc_id": "vet-kb-furd-002",
        "title": "Feline Upper Respiratory Complex - Clinical Signs and Diagnosis",
        "disease": "Feline Upper Respiratory Complex",
        "species": ["cat"],
        "section": "clinical_signs",
        "source": "merck_feline_uri",
        "text": (
            "Affected cats show fever, frequent sneezing, nasal discharge "
            "(serous to mucopurulent), conjunctivitis, rhinitis and "
            "salivation; the fever may reach 40.5 C (105 F). Herpesvirus tends "
            "to cause ocular signs including ulcerative keratitis, epiphora, "
            "chemosis, blepharospasm and conjunctival hyperaemia, while "
            "calicivirus is associated with ulcerative stomatitis and, in young "
            "kittens, a transient 'limping syndrome' with fever and joint pain. "
            "Signs may persist 5-10 days in milder cases and up to 6 weeks in "
            "severe cases. Diagnosis is usually presumptive from the typical "
            "signs; cytological examination of Giemsa-stained conjunctival "
            "scrapings helps identify chlamydiae and mycoplasmas, and PCR on "
            "ocular, nasal or pharyngeal secretions identifies specific agents."
        ),
    },
    {
        "doc_id": "vet-kb-furd-003",
        "title": "Feline Upper Respiratory Complex - Treatment",
        "disease": "Feline Upper Respiratory Complex",
        "species": ["cat"],
        "section": "treatment",
        "source": "merck_feline_uri",
        "text": (
            "Treatment of feline upper respiratory disease is largely "
            "symptomatic and supportive, but broad-spectrum antimicrobials are "
            "useful against secondary bacterial infection - for example "
            "amoxicillin with clavulanic acid, cephalosporins, "
            "trimethoprim-sulfa, fluoroquinolones, tetracyclines or "
            "chloramphenicol - with tetracyclines and fluoroquinolones being "
            "most effective against Chlamydia felis and Mycoplasma felis. "
            "Herpetic keratitis is treated with topical antiviral ointments "
            "containing idoxuridine, trifluridine or vidarabine every 4 hours, "
            "and oral lysine (250 mg 2-3 times daily) interferes with herpesvirus "
            "replication. Supportive measures include decongestant nose drops, "
            "fluid therapy, oxygen for severe dyspnoea, and assisted feeding, "
            "because inappetence from nasal congestion and oral ulcers is a "
            "major cause of deterioration."
        ),
    },
    {
        "doc_id": "vet-kb-furd-004",
        "title": "Feline Upper Respiratory Complex - Prevention and Vaccination",
        "disease": "Feline Upper Respiratory Complex",
        "species": ["cat"],
        "section": "prevention",
        "source": "merck_feline_uri",
        "text": (
            "Vaccination is central to preventing feline upper respiratory "
            "disease, although vaccines reduce severity rather than fully "
            "preventing infection. For intranasal FVR-FCV vaccine, cats older "
            "than 9 weeks should be vaccinated twice with a 3-week interval, "
            "kittens vaccinated at 3-4 week intervals until at least 12 weeks "
            "old, and adults revaccinated with a single dose every 1-3 years. "
            "Parenteral vaccines are available, often combined with feline "
            "panleukopenia. For Chlamydia felis, a single dose is recommended "
            "for cats over 12 weeks old, younger kittens revaccinated at 16 "
            "weeks, and all revaccinated annually. Control of environmental "
            "factors - reducing overcrowding, stress and exposure to sick cats, "
            "and improving ventilation and hygiene - is essential in catteries "
            "and shelters."
        ),
    },
    # ------------------------- FELINE HAEMOPLASMOSIS ------------------------- #
    {
        "doc_id": "vet-kb-hmpl-001",
        "title": "Feline Haemoplasmosis - Aetiology and Transmission",
        "disease": "Feline Haemoplasmosis",
        "species": ["cat"],
        "section": "aetiology",
        "source": "merck_haemoplasma",
        "text": (
            "Feline haemoplasmosis (feline infectious anaemia) is caused by "
            "haemotropic mycoplasmas that attach to the surface of red blood "
            "cells. Mycoplasma haemofelis (formerly the Ohio strain or large "
            "form of Haemobartonella felis) is the most pathogenic organism "
            "causing feline infectious anaemia; Candidatus Mycoplasma "
            "haemominutum (formerly the California strain or small form) is the "
            "most common haemoplasma in cat populations worldwide but is less "
            "pathogenic; and the pathogenicity of Candidatus Mycoplasma "
            "turicensis is not well understood. Direct transmission, possibly "
            "associated with fighting, is suspected and is supported by studies "
            "reporting haemoplasma DNA in saliva, on the gingiva and on claw "
            "beds; vertical (mother-to-offspring) transmission has also been "
            "documented."
        ),
    },
    {
        "doc_id": "vet-kb-hmpl-002",
        "title": "Feline Haemoplasmosis - Clinical Signs and Diagnosis",
        "disease": "Feline Haemoplasmosis",
        "species": ["cat"],
        "section": "clinical_signs",
        "source": "merck_haemoplasma",
        "text": (
            "Cats with acute haemoplasmosis show weakness, pallor of the mucous "
            "membranes, tachypnoea, tachycardia and occasionally collapse "
            "reflecting the underlying haemolytic anaemia; acutely ill cats may "
            "be febrile, while moribund cats may become hypothermic. Cardiac "
            "murmurs, splenomegaly and icterus may be present, and chronically "
            "affected cats show weakness, depression and weight loss or "
            "emaciation. Diagnosis is by identification of organisms on the "
            "surface of red cells in peripheral blood using light microscopy, "
            "but Mycoplasma haemofelis is visible less than 50% of the time in "
            "acutely infected cats, so PCR assays - which are considerably more "
            "sensitive and specific - are preferred. A regenerative anaemia "
            "with polychromasia, anisocytosis, nucleated red cells and a raised "
            "reticulocyte count is typical."
        ),
    },
    {
        "doc_id": "vet-kb-hmpl-003",
        "title": "Feline Haemoplasmosis - Treatment",
        "disease": "Feline Haemoplasmosis",
        "species": ["cat"],
        "section": "treatment",
        "source": "merck_haemoplasma",
        "text": (
            "The mainstay of treatment for feline haemoplasmosis is doxycycline "
            "at 10 mg/kg per day PO for a minimum of 2 weeks; administration of "
            "doxycycline hyclate should be followed by a bolus of several "
            "millilitres of water to prevent oesophageal injury. Either "
            "marbofloxacin or pradofloxacin is a suitable alternative to "
            "doxycycline. Supportive care includes oxygen and blood "
            "transfusions in severely anaemic cats, and immunosuppressive "
            "dosages of glucocorticoids may be added in cats that do not "
            "respond to antimicrobial therapy alone, because an immune-mediated "
            "component contributes to red-cell destruction. Treatment of "
            "PCR-positive healthy cats is currently not recommended, because no "
            "regimen has yet been shown to completely eliminate the organism."
        ),
    },
]


def build() -> list[dict]:
    """Expand source keys into full source objects and return chunk records."""
    records: list[dict] = []
    for chunk in CHUNKS:
        src_key = chunk["source"]
        if src_key not in SOURCES:
            raise KeyError(f"Chunk {chunk['doc_id']} references unknown source '{src_key}'")
        src = SOURCES[src_key]
        record = {
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "disease": chunk["disease"],
            "species": chunk["species"],
            "section": chunk["section"],
            "text": " ".join(chunk["text"].split()),
            "source": {
                "citation": src["citation"],
                "type": src["type"],
                "url": src["url"],
                "verification": src["verification"],
                "access_date": ACCESS_DATE,
            },
            "provenance": PROVENANCE,
        }
        records.append(record)
    return records


def build_sources_manifest(records: list[dict]) -> dict:
    """Build the provenance manifest (sources.json). Each source lists the
    doc_ids derived from it, so the manifest doubles as the dissertation's
    data-provenance appendix."""
    used_keys = {c["source"] for c in CHUNKS}
    # map source_key -> list of doc_ids
    key_to_docs: dict[str, list[str]] = {}
    for chunk in CHUNKS:
        key_to_docs.setdefault(chunk["source"], []).append(chunk["doc_id"])

    sources_out = []
    for key in sorted(used_keys):
        src = SOURCES[key]
        sources_out.append({
            "source_id": key,
            "citation_apa7": src["citation"],
            "type": src["type"],
            "url": src["url"],
            "access_date": ACCESS_DATE,
            "verification": src["verification"],
            "doc_ids": sorted(key_to_docs[key]),
        })
    return {
        "manifest_version": "1.0",
        "generated": ACCESS_DATE,
        "description": (
            "Provenance manifest for the veterinary RAG grounding corpus "
            "(data/rag_knowledge_base/kb_chunks.jsonl). Every source below was "
            "located and fetched during curation; no citation, DOI or URL was "
            "invented. verification=verified_fetched means the full text was "
            "retrieved and read during curation; verified_metadata_only means "
            "only existence/metadata was confirmed."
        ),
        "n_sources": len(sources_out),
        "n_chunks": len(records),
        "sources": sources_out,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "data" / "rag_knowledge_base"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kb_chunks.jsonl"
    records = build()
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = build_sources_manifest(records)
    manifest_path = out_dir / "sources.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    n_words = [len(r["text"].split()) for r in records]
    print(f"Wrote {len(records)} chunks to {out_path}")
    print(f"Wrote {manifest['n_sources']} sources to {manifest_path}")
    print(f"Word count per chunk: min={min(n_words)} max={max(n_words)} "
          f"mean={sum(n_words) // len(n_words)}")
    diseases = sorted({r["disease"] for r in records})
    print(f"Diseases covered ({len(diseases)}): {', '.join(diseases)}")
    print(f"Distinct sources used: {len({c['source'] for c in CHUNKS})}")


if __name__ == "__main__":
    main()
