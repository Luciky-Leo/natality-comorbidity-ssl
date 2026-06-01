# Linked Infant Death Supervised Transfer Baseline

All-input LightGBM was trained for the registry-defined severe neonatal endpoint using the same 2016-2022 sampled training records as the main incremental analysis, Platt recalibrated in 2023, and transferred to the linked birth/infant death cohort without using infant-death labels.

- training rows: 3,500,000
- development rows: 3,605,081
- linked rows: 3,596,017
- fixed high-risk phenotype fraction: 4.5787%

## Infant death enrichment

| Rule | Selected % | Selected n | Infant death rate % | Enrichment | Death capture % |
|---|---:|---:|---:|---:|---:|
| phenotype_0 | 4.579 | 164,651 | 2.021 | 3.68 | 16.85 |
| all_input_lgbm_top_0.500pct | 0.500 | 17,980 | 42.503 | 77.42 | 38.71 |
| all_input_lgbm_top_1.000pct | 1.000 | 35,960 | 22.511 | 41.00 | 41.00 |
| all_input_lgbm_top_2.000pct | 2.000 | 71,920 | 12.044 | 21.94 | 43.87 |
| all_input_lgbm_top_4.579pct | 4.579 | 164,651 | 5.886 | 10.72 | 49.09 |
| all_input_lgbm_top_5.000pct | 5.000 | 179,801 | 5.466 | 9.96 | 49.78 |
| all_input_lgbm_top_10.000pct | 10.000 | 359,602 | 3.112 | 5.67 | 56.68 |

## Outputs

- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\linked_infant_death_supervised_transfer_baseline_full2016_2022_mask035_d48_l2_cuda_full2023dev.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\linked_infant_death_supervised_transfer_baseline_metadata_full2016_2022_mask035_d48_l2_cuda_full2023dev.json`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\objects\linked_infant_death_supervised_transfer_scores_full2016_2022_mask035_d48_l2_cuda_full2023dev.parquet`
