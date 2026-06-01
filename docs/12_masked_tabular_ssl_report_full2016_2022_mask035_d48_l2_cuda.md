# Streaming Masked Tabular SSL Report

Preprocessing and training stream through 2016-2022 row groups without loading all rows into memory.

## Configuration

- device: cuda
- train rows seen by preprocessor: 26,345,765
- row groups per year: all
- rows per row group: all
- development rows: 500,000
- categorical features: 44
- numeric features: 68
- d_model: 48
- layers: 2
- heads: 4
- mask rate: 0.35
- epochs: 1

## Final Development Loss

- dev total loss: 0.185863
- dev categorical reconstruction loss: 0.301314
- dev numeric reconstruction loss: 0.111160

## Outputs

- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\objects\masked_tabular_ssl_encoder_full2016_2022_mask035_d48_l2_cuda.pt`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\objects\masked_tabular_ssl_preprocessor_full2016_2022_mask035_d48_l2_cuda.json`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\masked_tabular_ssl_history_full2016_2022_mask035_d48_l2_cuda.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\objects\masked_tabular_ssl_dev_embeddings_full2016_2022_mask035_d48_l2_cuda.parquet`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\objects\masked_tabular_ssl_test_embeddings_full2016_2022_mask035_d48_l2_cuda.parquet`

## Interpretation Boundary

A true full-scale run should leave row-group and row-count debug options unset. CPU-only full-scale pretraining is expected to be substantially slower than sampled prototype runs.
