# SINASC independent registry SSL stress-test

## Boundary

This analysis trains and evaluates a separate masked tabular SSL model within Brazil SINASC public-use birth records. It is an independent registry workflow stress-test, not direct external validation of the U.S. Natality encoder.

## Data split

- 2023 sampled SSL training rows: 500,000
- 2023 development/calibration rows: 200,000
- 2024 full temporal test rows: 2,389,325

## SSL training

- Device: cpu
- Input features including missingness flags: 46
- Categorical features: 13
- Numeric features: 33
- Final development reconstruction loss: 0.3720
- Selected phenotype count k: 3

## Main 2024 stress-test result

- Best severe birth-status AUPRC model: sinasc_overlap_inputs_lightgbm (AUPRC 0.1316, AUPRC/prevalence 3.48, top 1% enrichment 8.05).
- Highest-risk phenotype for severe birth-status outcome: P0, n=1,001,823, severe event rate=4.20%.

## Interpretation

SINASC supports an independent stress-test of the workflow using overlapping registry variables and birth-status endpoints. The result should be described as transport of the analytical framework to an independent public registry. It should not be described as transport of the trained U.S. model, because the SINASC input schema lacks several U.S. Natality feature families.

## Outputs

- Metrics: `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\sinasc_ssl_stress_test_metrics_sinasc_2023train500k_dev200k_2024full_ssl.csv`
- Phenotype rates: `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\sinasc_ssl_phenotype_rates_sinasc_2023train500k_dev200k_2024full_ssl.csv`
- Cluster selection: `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\sinasc_ssl_cluster_selection_sinasc_2023train500k_dev200k_2024full_ssl.csv`
- Figure: `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\figures\sinasc_ssl_stress_test_summary_sinasc_2023train500k_dev200k_2024full_ssl.png`
- Metadata: `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\sinasc_ssl_stress_test_metadata_sinasc_2023train500k_dev200k_2024full_ssl.json`
