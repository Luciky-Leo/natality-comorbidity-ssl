# Risks and Boundaries

## Main Scientific Risks

### Outcome Misclassification

Natality variables come from birth certificates and may underreport some maternal and neonatal conditions. The manuscript should describe this as surveillance-grade public data, not adjudicated EHR data.

### Live-Birth Restriction

Natality birth files include live births. They do not capture all pregnancies or fetal deaths unless linked fetal-death files are added. Avoid conclusions about total pregnancy loss burden.

### Leakage

Delivery method, labor characteristics, birthweight, gestational age and neonatal conditions can become leakage variables depending on the prediction target. Keep the primary model antepartum/pre-delivery. If intrapartum or delivery-context models are added, label them clearly as secondary.

### Causal Overclaiming

This project is not a causal effect study. It can identify risk phenotypes and calibrated enrichment patterns, but it cannot prove that a phenotype causes an outcome.

### Fairness Interpretation

Race/ethnicity, insurance/payment and education variables should be framed as social and structural context variables, not biological explanations.

### Comorbidity Detail

Natality can support endocrine, hypertensive, infectious and ART-related comorbidity phenotyping. It is weak for detailed cerebrovascular, gastrointestinal, urologic or renal emergency phenotyping. Those questions need MIMIC, HCUP, state inpatient data or hospital EHR.

## Statistical Risks

- Huge sample size can make trivial differences statistically significant.
- Rare outcomes require prevalence, absolute risk, AUPRC baseline and calibration, not only AUROC.
- Hyperparameter selection on a final test set would invalidate the validation claim.
- High-dimensional embeddings require regularization and sensitivity analyses with lower-dimensional PCA or embedding sizes.

## Recommended Conservative Claims

Use:

- "risk enrichment";
- "phenotype discovery";
- "calibrated public-data framework";
- "patterns consistent with comorbidity burden";
- "decision-support signal requiring prospective validation."

Avoid:

- "diagnosis";
- "causal effect";
- "clinical deployment-ready";
- "foundation model" unless the scale and pretraining justification are strong enough;
- "external clinical validation" unless an outcome-labelled external dataset is added.
