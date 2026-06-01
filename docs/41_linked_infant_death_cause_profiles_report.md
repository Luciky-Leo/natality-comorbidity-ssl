# Linked Infant Death Cause-Specific Phenotype Profiles

Underlying cause of death was parsed from UCOD ICD-10 positions 1368-1371 in the CDC/NCHS linked birth/infant death numerator records. Cause-specific enrichment is descriptive and grouped into broad ICD-10 categories.

- parsed infant death records: 19,743
- linked assignment rows: 3,596,017

## High-Risk Phenotype Cause-Specific Enrichment

| Cause group | Events | Rate % | Baseline % | Enrichment | Event capture % |
|---|---:|---:|---:|---:|---:|
| Perinatal conditions | 2,394 | 1.4540 | 0.2712 | 5.36 | 24.55 |
| Congenital anomalies | 417 | 0.2533 | 0.1103 | 2.30 | 10.51 |
| Infection/respiratory | 81 | 0.0492 | 0.0217 | 2.27 | 10.38 |
| Other causes | 116 | 0.0705 | 0.0332 | 2.12 | 9.71 |
| SIDS/ill-defined | 202 | 0.1227 | 0.0690 | 1.78 | 8.14 |
| External injury | 117 | 0.0711 | 0.0436 | 1.63 | 7.46 |

## Outputs

- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\linked_infant_death_cause_records_full2016_2022_mask035_d48_l2_cuda_full2023dev.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\linked_infant_death_cause_rates_full2016_2022_mask035_d48_l2_cuda_full2023dev.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\linked_infant_death_cause_highrisk_enrichment_full2016_2022_mask035_d48_l2_cuda_full2023dev.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\linked_infant_death_cause_profiles_metadata_full2016_2022_mask035_d48_l2_cuda_full2023dev.json`
