# Project Protocol

## Research Question

Can self-supervised representations learned from routinely recorded maternal comorbidity and pregnancy-risk variables identify interpretable phenotypes that enrich for adverse neonatal and maternal delivery outcomes in U.S. Natality public-use records?

## Design

Retrospective national birth-record study using CDC/NCHS Natality public-use files.

## Data Source

Primary source: CDC/NCHS Natality public-use files from the Vital Statistics Online Data Portal.

Feasibility source: CDC WONDER Natality Expanded documentation and queries can be used for variable discovery and aggregate sanity checks, but individual-level modelling should use the downloadable public-use files.

## Candidate Study Years

- Development/training: 2016-2022.
- Model selection/development validation: 2023.
- Final temporal test: 2024.

If compute or storage becomes limiting, start with a one-year 2024 smoke test, then scale to a 3-year design before full 2016-2024 modelling.

## Population

All U.S. live births in selected Natality public-use years after excluding records with missing or implausible key variables needed for primary outcomes and model inputs.

Important boundary: Natality live-birth files do not represent all pregnancies. Stillbirths, miscarriages, and terminations require other files and should not be implied unless linked fetal-death data are explicitly added.

## Exposure/Input Concept

The input representation is not a single exposure. It is a maternal comorbidity and pregnancy-risk state derived from pre-delivery variables:

- maternal age, race/ethnicity, education, marital status, payment source;
- parity/live-birth order and prior cesarean/prior preterm birth where available;
- prepregnancy BMI and gestational weight gain;
- prepregnancy diabetes and gestational diabetes;
- prepregnancy hypertension, gestational hypertension, and eclampsia;
- smoking before and during pregnancy;
- infections present or treated during pregnancy;
- infertility treatment, fertility-enhancing drugs, ART;
- prenatal-care timing and visit count;
- plurality and gestational age-related context, with careful leakage checks depending on outcome.

## Primary Outcome Candidates

Primary neonatal risk-enrichment outcome:

- composite adverse neonatal outcome, defined from variables such as preterm birth, low birthweight, very low birthweight, low 5-minute Apgar, assisted ventilation, NICU admission or transfer, depending on confirmed file availability and harmonization.

Secondary maternal morbidity outcome:

- composite maternal morbidity using transfusion, ruptured uterus, unplanned hysterectomy, maternal ICU admission, and severe perineal laceration where available.

Outcome definitions must be finalized only after reading the year-specific user guides and checking variable harmonization across years.

## Modelling Strategy

### Self-supervised Encoder

Train a masked tabular transformer-style encoder on input variables without using outcome labels.

Candidate objectives:

- masked categorical reconstruction by cross-entropy;
- masked continuous reconstruction by scaled MSE or Huber loss;
- denoising of missingness indicators;
- optional contrastive consistency between two masked/noised views of the same birth record.

### Phenotype Discovery

Use the frozen or development-selected embedding to derive phenotype clusters.

Primary approach:

- fit scaler, embedding reducer if used, and clustering only in the training set;
- select cluster number using development data and clinical interpretability;
- assign development/test records to fixed centroids;
- describe phenotypes using standardized feature enrichment and outcome rates.

### Risk Models

Compare:

- classical clinical variables only;
- SSL embedding only;
- clinical variables plus SSL embedding;
- clinical variables plus SSL embedding plus phenotype label;
- gradient boosting and penalized logistic baselines.

## Validation

Avoid selecting hyperparameters on the final test set.

Recommended split:

- train: 2016-2022;
- dev: 2023;
- final test: 2024.

Report:

- event prevalence;
- AUROC and AUPRC with bootstrap confidence intervals;
- AUPRC relative to event-rate baseline;
- calibration intercept/slope, Brier score, ECE;
- subgroup performance by race/ethnicity, age, BMI, diabetes, hypertension, ART/infertility treatment, plurality and payment source where appropriate.

## Clinical Interpretation

The paper should claim risk enrichment and phenotype discovery, not causal effects or autonomous clinical diagnosis. Phenotypes should be interpreted as patterns consistent with maternal comorbidity burden and pregnancy-risk context.

## Target Journal Framing

Best near-term fit:

- BMC Medical Informatics and Decision Making;
- Artificial Intelligence in Medicine if the SSL method and validation are made stronger;
- npj Women's Health or npj Digital Medicine only if external validation, method novelty, and clinical interpretation become much stronger.
