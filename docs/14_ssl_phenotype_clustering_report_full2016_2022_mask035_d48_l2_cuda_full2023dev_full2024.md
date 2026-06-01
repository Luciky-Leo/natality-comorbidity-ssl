# SSL Phenotype Clustering Report

Phenotype discovery was fit on 2023 development embeddings only. The selected centroids were then applied to 2024 test embeddings.

## Cluster Selection

- selected k: 3
- PCA components used for clustering: 16
- PCA explained variance ratio total: 0.9722

## 2024 Phenotype Outcome Rates

| Phenotype | n | Maternal core morbidity % | Severe neonatal % |
|---:|---:|---:|---:|
| 0 | 1,677,612 | 0.815 | 4.815 |
| 1 | 1,788,300 | 0.681 | 4.145 |
| 2 | 172,524 | 1.271 | 8.249 |

## Phenotype Stability on 2023 Development Resamples

- resampling scheme: 10 iterations, 80% of 2023 development records per iteration
- stability sample cap: 500000
- median ARI vs primary labels: 0.997
- median NMI vs primary labels: 0.991
- median minimum cluster proportion: 0.046

## Bootstrap 95% CI for 2024 Phenotype Outcome Rates

| Endpoint | Phenotype | Event rate % | 95% CI % |
|---|---:|---:|---:|
| Maternal core morbidity | 0 | 0.815 | 0.801-0.826 |
| Maternal core morbidity | 1 | 0.681 | 0.671-0.695 |
| Maternal core morbidity | 2 | 1.271 | 1.213-1.331 |
| Severe neonatal | 0 | 4.815 | 4.783-4.844 |
| Severe neonatal | 1 | 4.145 | 4.116-4.168 |
| Severe neonatal | 2 | 8.249 | 8.138-8.408 |

## Platt-Calibrated 2024 Risk Metrics

| Endpoint | Feature set | Events/test | Prev. | AUROC | AUPRC | AUPRC/Prev. | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| outcome_maternal_morbidity_core | ssl_embedding | 28,042 | 0.00771 | 0.6583 | 0.0168 | 2.19 | 0.00762 | 0.00015 |
| outcome_maternal_morbidity_core | ssl_plus_phenotype | 28,042 | 0.00771 | 0.6588 | 0.0167 | 2.17 | 0.00762 | 0.00016 |
| outcome_maternal_morbidity_core | phenotype | 28,042 | 0.00771 | 0.5354 | 0.0085 | 1.11 | 0.00765 | 0.00130 |
| outcome_severe_neonatal_no_nicu | ssl_embedding | 169,135 | 0.04649 | 0.7093 | 0.1872 | 4.03 | 0.04132 | 0.00221 |
| outcome_severe_neonatal_no_nicu | ssl_plus_phenotype | 169,135 | 0.04649 | 0.7098 | 0.1859 | 4.00 | 0.04133 | 0.00211 |
| outcome_severe_neonatal_no_nicu | phenotype | 169,135 | 0.04649 | 0.5364 | 0.0518 | 1.12 | 0.04425 | 0.00423 |

## Bootstrap 95% CI for Platt-Calibrated AUPRC

| Endpoint | Feature set | AUPRC | 95% CI |
|---|---|---:|---:|
| outcome_maternal_morbidity_core | ssl_embedding | 0.0168 | 0.0165-0.0173 |
| outcome_maternal_morbidity_core | ssl_plus_phenotype | 0.0167 | 0.0164-0.0172 |
| outcome_maternal_morbidity_core | phenotype | 0.0085 | 0.0084-0.0086 |
| outcome_severe_neonatal_no_nicu | ssl_embedding | 0.1872 | 0.1853-0.1891 |
| outcome_severe_neonatal_no_nicu | ssl_plus_phenotype | 0.1859 | 0.1840-0.1879 |
| outcome_severe_neonatal_no_nicu | phenotype | 0.0518 | 0.0516-0.0522 |

## Top 1% Enrichment

| Endpoint | Feature set | Event rate | Enrichment over prevalence | Event capture % |
|---|---|---:|---:|---:|
| outcome_maternal_morbidity_core | ssl_embedding | 0.0361 | 4.69 | 4.69 |
| outcome_maternal_morbidity_core | ssl_plus_phenotype | 0.0351 | 4.55 | 4.55 |
| outcome_maternal_morbidity_core | phenotype | 0.0183 | 2.38 | 2.38 |
| outcome_severe_neonatal_no_nicu | ssl_plus_phenotype | 0.4745 | 10.21 | 10.21 |
| outcome_severe_neonatal_no_nicu | ssl_embedding | 0.4731 | 10.18 | 10.18 |
| outcome_severe_neonatal_no_nicu | phenotype | 0.0972 | 2.09 | 2.09 |

## Outputs

- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\ssl_phenotype_cluster_selection_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\ssl_phenotype_outcome_rates_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\ssl_phenotype_stability_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\ssl_phenotype_risk_metrics_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\ssl_phenotype_risk_metric_bootstrap_ci_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\ssl_phenotype_top_risk_enrichment_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\ssl_phenotype_outcome_rate_bootstrap_ci_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\ssl_phenotype_prototypes_2023_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\objects\ssl_phenotype_dev_assignments_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.parquet`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\objects\ssl_phenotype_test_assignments_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.parquet`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\figures\ssl_phenotype_cluster_selection_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.png`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\figures\ssl_phenotype_stability_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.png`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\figures\ssl_phenotype_2024_outcome_rates_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.png`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\figures\ssl_phenotype_2024_outcome_rate_ci_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.png`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\figures\ssl_phenotype_2024_auprc_ci_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.png`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\figures\ssl_phenotype_2023_pca_scatter_full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024.png`

## Boundary

The current embeddings cover 3,605,081 development records and the full 3,638,436-record 2024 temporal test year. The 2024 labels were used only for final evaluation, not for cluster selection.
