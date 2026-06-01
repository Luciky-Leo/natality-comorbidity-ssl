# Multi-Year Analytic Cohort Report

Generated analytic cohorts for 2016-2024 CDC/NCHS Natality public-use files.

## Cohort Files

| Year | Split | Rows | Columns | Size MB |
|---:|---|---:|---:|---:|
| 2016 | train | 3,956,112 | 136 | 83.9 |
| 2017 | train | 3,864,754 | 136 | 81.9 |
| 2018 | train | 3,801,534 | 136 | 81.4 |
| 2019 | train | 3,757,582 | 136 | 80.8 |
| 2020 | train | 3,619,826 | 136 | 77.6 |
| 2021 | train | 3,669,928 | 136 | 77.7 |
| 2022 | train | 3,676,029 | 136 | 78.7 |
| 2023 | development | 3,605,081 | 136 | 76.9 |
| 2024 | test | 3,638,436 | 136 | 77.5 |

## Split Summary

| Split | Years | Rows |
|---|---|---:|
| development | 2023 | 3,605,081 |
| test | 2024 | 3,638,436 |
| train | 2016,2017,2018,2019,2020,2021,2022 | 26,345,765 |

## Primary Endpoint Prevalence

| Year | Split | Maternal core morbidity % | Severe neonatal no NICU % |
|---:|---|---:|---:|
| 2016 | train | 0.5131 | 3.7235 |
| 2017 | train | 0.5494 | 3.8088 |
| 2018 | train | 0.5681 | 3.8474 |
| 2019 | train | 0.6061 | 4.0206 |
| 2020 | train | 0.6219 | 4.1135 |
| 2021 | train | 0.6897 | 4.3417 |
| 2022 | train | 0.6838 | 4.3554 |
| 2023 | development | 0.7591 | 4.5481 |
| 2024 | test | 0.7707 | 4.6486 |

## Output Tables

- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\natality_2016_2024_analytic_manifest.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\data\processed\natality_2016_2024_split_manifest.csv`
- `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\natality_2016_2024_analytic_endpoint_prevalence.csv`

## Interpretation

- The planned temporal split is now explicit: 2016-2022 training, 2023 development/model selection, and 2024 final test.
- The 2024 final test set must not be used for SSL hyperparameter selection or phenotype cluster-number selection.
- The increasing maternal morbidity prevalence over calendar years should be described and handled through temporal validation rather than random splitting.
