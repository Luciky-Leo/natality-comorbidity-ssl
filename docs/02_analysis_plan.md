# Analysis Plan

## Stage 0: One-Year Smoke Test

Goal: prove that the fixed-width parser, variable dictionary and outcome definitions are correct before building AI models.

Deliverables:

- parsed 2024 sample or full-year table;
- variable dictionary extracted from the user guide;
- missingness table;
- outcome prevalence table;
- baseline descriptive Table 1.

## Stage 1: Multi-Year Cohort Build

Use 2016-2024 public-use files if variables are harmonized.

Required outputs:

- `data/processed/cohort.parquet`;
- `data/processed/splits.parquet`;
- `results/tables/source_data_table1.csv`;
- `results/tables/outcome_prevalence_by_year.csv`.

## Stage 2: Masked Tabular SSL Encoder

Candidate model:

- categorical embeddings for categorical variables;
- linear projections for continuous variables;
- transformer encoder over feature tokens;
- pooled record embedding;
- masked reconstruction heads for categorical and continuous variables.

Masking strategy:

- randomly mask categorical tokens;
- randomly mask continuous values;
- add missingness-aware masks;
- optionally create two masked views for contrastive consistency.

Primary embedding sizes to compare on development data only:

- 16, 32, 64.

Primary hyperparameters to select on development data only:

- mask rate;
- embedding dimension;
- number of transformer layers;
- dropout/weight decay;
- reconstruction-only versus reconstruction plus contrastive objective.

## Stage 3: Phenotype Clustering

Fit clustering only on training embeddings.

Candidate workflow:

- standardize embeddings using training-set scaler;
- optionally reduce dimensions with training-set PCA;
- fit k-means or Gaussian mixture models;
- select k using development-set stability, silhouette, phenotype size, and clinical interpretability;
- assign test records to fixed training centroids.

Report:

- phenotype sizes;
- standardized feature enrichment;
- outcome rates by phenotype;
- adjusted association between phenotype and outcomes;
- phenotype prototype heatmap.

## Stage 4: Risk Enrichment Models

Primary comparison:

- clinical variables only;
- SSL embeddings only;
- clinical variables plus SSL embeddings;
- clinical variables plus SSL embeddings plus phenotype.

Model classes:

- penalized logistic regression;
- XGBoost/LightGBM if available;
- explainable boosting machine if package support is available;
- calibrated logistic risk score as the interpretable final model.

Primary metric:

- AUPRC, because the target is risk enrichment under imbalanced adverse outcomes.

Secondary metrics:

- AUROC;
- Brier score;
- calibration intercept and slope;
- expected calibration error;
- decision-curve net benefit.

## Stage 5: Calibration

Calibration must be trained on development data or through nested/out-of-fold training only.

Candidate recalibration:

- Platt scaling;
- isotonic regression if sample size and event count are adequate.

Report both raw and recalibrated scores.

## Stage 6: Explainability

Explain at three levels:

- global feature importance or SHAP on final risk model;
- phenotype-level feature enrichment;
- embedding contribution by feature family through ablation or grouped permutation importance.

## Stage 7: Manuscript Figures

Planned main figures:

1. Study design and AI method workflow.
2. Cohort construction, missingness, and outcome prevalence.
3. SSL representation and maternal comorbidity phenotype map.
4. Model performance, AUPRC enrichment over prevalence baseline, and calibration.
5. Phenotype clinical interpretation and subgroup/fairness analysis.

Planned supplement:

- variable harmonization;
- model-selection grid using development data only;
- bootstrap stability;
- sensitivity outcomes;
- leakage-control analysis;
- full source-data tables.
