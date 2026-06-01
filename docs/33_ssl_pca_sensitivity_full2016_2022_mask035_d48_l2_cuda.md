# SSL PCA Sensitivity Report

PCA was fit on 2023 development SSL embeddings only. The 2024 SSL test embeddings were transformed using the fixed 2023 scaler/PCA and evaluated once.

## Key Metrics

| Endpoint | Feature set | AUPRC | AUROC | Top 1% enrichment | ECE |
|---|---|---:|---:|---:|---:|
| Maternal core morbidity | PCA 5 | 0.0127 | 0.6199 | 2.92 | 0.00071 |
| Maternal core morbidity | PCA 10 | 0.0126 | 0.6202 | 3.32 | 0.00069 |
| Maternal core morbidity | PCA 20 | 0.0148 | 0.6409 | 3.80 | 0.00071 |
| Maternal core morbidity | PCA 32 | 0.0162 | 0.6566 | 4.12 | 0.00070 |
| Maternal core morbidity | Full 48 | 0.0167 | 0.6645 | 4.65 | 0.00065 |
| Severe neonatal outcome | PCA 5 | 0.0815 | 0.6494 | 2.76 | 0.00474 |
| Severe neonatal outcome | PCA 10 | 0.0841 | 0.6500 | 3.12 | 0.00411 |
| Severe neonatal outcome | PCA 20 | 0.1060 | 0.6754 | 4.72 | 0.00135 |
| Severe neonatal outcome | PCA 32 | 0.1429 | 0.6934 | 7.42 | 0.00128 |
| Severe neonatal outcome | Full 48 | 0.1834 | 0.7097 | 9.81 | 0.00204 |

## Interpretation

If PCA-compressed embeddings retain similar performance to the full 48-dimensional embedding, the SSL signal is less likely to be an artifact of high-dimensional overfitting.

## Output

- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\ssl_pca_sensitivity_full2016_2022_mask035_d48_l2_cuda.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\figures\ssl_pca_sensitivity_full2016_2022_mask035_d48_l2_cuda.png`
