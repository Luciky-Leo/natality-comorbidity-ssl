# SSL Ablation Grid Report

This focused reconstruction ablation varies masking rate, embedding width, and encoder depth while preserving the 2016-2022 pretrain / 2023 development boundary.

- train sample per configuration: 20,000 records/year x 7 years
- development sample per configuration: 100,000
- epochs per configuration: 1

## Final Development Reconstruction Loss

| Tag | Mask | d_model | Layers | Dev loss | Dev cat | Dev num |
|---|---:|---:|---:|---:|---:|---:|
| abl_mask035_d48_l2 | 0.35 | 48 | 2 | 0.8181 | 0.6880 | 0.9022 |
| abl_mask020_d48_l1 | 0.20 | 48 | 1 | 0.8186 | 0.6895 | 0.9022 |
| abl_mask010_d48_l2 | 0.10 | 48 | 2 | 0.8191 | 0.6885 | 0.9035 |
| abl_mask020_d48_l2 | 0.20 | 48 | 2 | 0.8196 | 0.6886 | 0.9044 |
| abl_mask020_d32_l2 | 0.20 | 32 | 2 | 0.8226 | 0.6871 | 0.9103 |

## Boundary

This is a reconstruction-level ablation. The selected SSL configuration should still be propagated through embedding export, phenotype clustering, calibration, and final 2024 risk enrichment before manuscript claims are upgraded.

## Output

- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\ssl_ablation_grid.csv`
