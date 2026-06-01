# Incremental SSL LightGBM Report

This analysis tests whether full-scale SSL embeddings add predictive value beyond leakage-controlled Natality input variables under the same 2016-2022 train, 2023 calibration, and 2024 test split.

| Endpoint | Feature set | AUPRC | AUROC | AUPRC/Prev. | Top 1% enrichment | ECE |
|---|---|---:|---:|---:|---:|---:|
| outcome_maternal_morbidity_core | all_inputs | 0.0234 | 0.6905 | 3.03 | 7.04 | 0.00016 |
| outcome_maternal_morbidity_core | ssl_embeddings | 0.0170 | 0.6584 | 2.21 | 4.77 | 0.00016 |
| outcome_maternal_morbidity_core | all_inputs_plus_ssl | 0.0230 | 0.6901 | 2.98 | 6.99 | 0.00018 |
| outcome_severe_neonatal_no_nicu | all_inputs | 0.3009 | 0.7613 | 6.47 | 15.20 | 0.00127 |
| outcome_severe_neonatal_no_nicu | ssl_embeddings | 0.2184 | 0.7145 | 4.70 | 11.58 | 0.00188 |
| outcome_severe_neonatal_no_nicu | all_inputs_plus_ssl | 0.3008 | 0.7612 | 6.47 | 15.16 | 0.00123 |

## Outputs

- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\incremental_ssl_lightgbm_metrics_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\incremental_ssl_lightgbm_topk_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\incremental_ssl_lightgbm_bootstrap_ci_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\incremental_ssl_lightgbm_metadata_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.json`

## Boundary

If `all_inputs_plus_ssl` does not improve over `all_inputs`, the manuscript should not claim broad predictive superiority of SSL. The defensible claim would shift toward scalable representation learning, phenotype discovery, and risk enrichment rather than replacement of strong supervised baselines.
