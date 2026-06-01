# Linked Infant Death Adjusted Logistic Regression

Fixed SSL phenotypes were evaluated in descriptive logistic models for linked infant death endpoints. The maternal-registry model adjusted for available public-use maternal and pregnancy variables. The birth-status model additionally adjusted for gestational age and birthweight, which should be interpreted as an attenuation/sensitivity model rather than a deployable pre-delivery prediction model.

- linked cohort rows: 3,596,017
- output table: `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\linked_infant_death_adjusted_logistic_full2016_2022_mask035_d48_l2_cuda_full2023dev.csv`
- phenotype-only table: `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\linked_infant_death_adjusted_logistic_phenotype_terms_full2016_2022_mask035_d48_l2_cuda_full2023dev.csv`

## Phenotype Odds Ratios

| Outcome | Model | Term | OR | 95% CI | Events |
|---|---|---|---:|---:|---:|
| Infant death | maternal_registry_adjusted | phenotype_0_vs_2 | 3.413 | 3.219-3.617 | 19,743 |
| Infant death | maternal_registry_adjusted | phenotype_1_vs_2 | 1.077 | 1.036-1.119 | 19,743 |
| Infant death | birth_status_adjusted_descriptive | phenotype_0_vs_2 | 1.936 | 1.810-2.071 | 19,743 |
| Infant death | birth_status_adjusted_descriptive | phenotype_1_vs_2 | 1.235 | 1.186-1.286 | 19,743 |
| Neonatal death <28d | maternal_registry_adjusted | phenotype_0_vs_2 | 3.128 | 2.924-3.346 | 12,892 |
| Neonatal death <28d | maternal_registry_adjusted | phenotype_1_vs_2 | 0.760 | 0.724-0.797 | 12,892 |
| Neonatal death <28d | birth_status_adjusted_descriptive | phenotype_0_vs_2 | 1.606 | 1.483-1.740 | 12,892 |
| Neonatal death <28d | birth_status_adjusted_descriptive | phenotype_1_vs_2 | 0.980 | 0.931-1.032 | 12,892 |
| Early neonatal death <7d | maternal_registry_adjusted | phenotype_0_vs_2 | 3.105 | 2.884-3.343 | 10,046 |
| Early neonatal death <7d | maternal_registry_adjusted | phenotype_1_vs_2 | 0.636 | 0.602-0.672 | 10,046 |
| Early neonatal death <7d | birth_status_adjusted_descriptive | phenotype_0_vs_2 | 1.953 | 1.789-2.131 | 10,046 |
| Early neonatal death <7d | birth_status_adjusted_descriptive | phenotype_1_vs_2 | 1.249 | 1.176-1.326 | 10,046 |
| Postneonatal death 28d-1y | maternal_registry_adjusted | phenotype_0_vs_2 | 1.376 | 1.189-1.592 | 6,851 |
| Postneonatal death 28d-1y | maternal_registry_adjusted | phenotype_1_vs_2 | 1.524 | 1.429-1.626 | 6,851 |
| Postneonatal death 28d-1y | birth_status_adjusted_descriptive | phenotype_0_vs_2 | 1.219 | 1.057-1.405 | 6,851 |
| Postneonatal death 28d-1y | birth_status_adjusted_descriptive | phenotype_1_vs_2 | 1.765 | 1.656-1.882 | 6,851 |
