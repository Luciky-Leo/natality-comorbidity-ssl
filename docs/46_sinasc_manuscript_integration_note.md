# SINASC stress-test manuscript integration note

Date: 2026-05-28

## What was completed

An independent Brazil SINASC public-registry stress-test was completed using a separate harmonized matrix and a separately trained masked tabular SSL model.

- SSL training/development registry: SINASC 2023.
- Final temporal test registry: SINASC 2024.
- SSL training rows sampled from 2023: 500,000.
- Development/calibration rows sampled from 2023: 200,000.
- Full 2024 test rows: 2,389,325.
- Input features including missingness flags: 46.
- Final development reconstruction loss: 0.3720.
- Development-only selected phenotype count: k = 3.

## Main stress-test result

For the 2024 SINASC severe birth-status endpoint, the overlap-input LightGBM model remained the strongest pure predictor:

- LightGBM overlap inputs: AUPRC 0.1316, AUPRC/prevalence 3.48, top 1% enrichment 8.05.
- SSL embedding logistic: AUPRC 0.1070, AUPRC/prevalence 2.83, top 1% enrichment 6.06.
- SSL + phenotype logistic: AUPRC 0.1072, AUPRC/prevalence 2.84, top 1% enrichment 6.03.
- Phenotype only: AUPRC 0.0397, AUPRC/prevalence 1.05, top 1% enrichment 1.14.

For the broad birth-status endpoint:

- LightGBM overlap inputs: AUPRC 0.3784, AUPRC/prevalence 2.21, top 1% enrichment 4.93.
- SSL embedding logistic: AUPRC 0.3414, AUPRC/prevalence 1.99, top 1% enrichment 4.43.
- SSL + phenotype logistic: AUPRC 0.3417, AUPRC/prevalence 2.00, top 1% enrichment 4.40.

## Interpretation boundary

This should be written as independent registry workflow transport, not direct external validation of the U.S. Natality model. The trained U.S. encoder was not applied to SINASC because SINASC lacks several U.S. Natality feature families, including maternal BMI/weight gain, smoking, diabetes, hypertensive disorders, infections, infertility/ART, WIC/payment, current maternal morbidity variables, NICU, ventilation, and seizures.

The strongest defensible wording is:

> In an independent public-registry stress-test using Brazil SINASC, we rebuilt the harmonized matrix and trained a separate masked tabular SSL encoder under a 2023-to-2024 temporal split. SSL embeddings retained risk-enrichment signal for birth-status endpoints, although overlap-input LightGBM remained the strongest pure predictor. These results support transportability of the analytical workflow across public birth registries rather than transportability of a fixed U.S.-trained encoder.

## Recommended manuscript placement

- Main text if targeting high-impact digital medicine journals: add as a new "Independent registry stress-test" subsection after linked infant death transfer analysis.
- Figure option: add the SINASC summary as Figure S3 or a new Figure 7.
- Discussion: use it to soften the "no independent external validation" limitation, but keep a limitation that this is workflow-level registry transport, not hospital/EHR validation.

