# Additional file 2. Reporting checklist mapping

This mapping is intended to accompany the manuscript as a reviewer-facing guide. It is not a substitute for any journal-specific web form requested during submission.

## STROBE mapping

| STROBE domain | Manuscript location |
|---|---|
| Title/abstract | Title; Abstract |
| Background/rationale | Background |
| Objectives | Final paragraph of Background |
| Study design | Methods: Data source and study design; Figure 1 |
| Setting | Methods: CDC/NCHS Natality 2016-2024; linked birth/infant death; SINASC |
| Participants | Methods: Data source and study design; Results: Calendar-time cohort |
| Variables | Methods: Input variables; Outcomes; Linked endpoint; SINASC stress-test |
| Data sources/measurement | Methods: public-use files, field maps, registry-defined endpoints |
| Bias | Methods and Discussion: leakage control, temporal split, coding/missingness limitations |
| Study size | Results: Calendar-time cohort and analytic endpoints |
| Quantitative variables | Methods: Input variables; SSL encoder; phenotype discovery |
| Statistical methods | Methods: supervised baselines, calibration, decision curves, stability, sensitivity |
| Descriptive data | Results: cohort sizes, endpoint prevalence, phenotype profiles |
| Outcome data | Results: primary endpoints, linked infant death, SINASC outcomes |
| Main results | Results; Figures 2-7 |
| Other analyses | Results: calibration, PCA sensitivity, subgroup robustness, linked attenuation models |
| Limitations | Discussion limitations paragraph |
| Interpretation/generalizability | Discussion; Conclusions |
| Funding | Declarations: Funding |

## TRIPOD+AI mapping

| TRIPOD+AI domain | Manuscript location |
|---|---|
| Source of data | Methods: Data source and study design; Availability of data and materials |
| Prediction problem and intended use | Background; Methods: Clinical utility; Discussion |
| Eligibility and setting | Methods: Data source and study design; SINASC public-registry stress-test |
| Outcome definition | Methods: Outcomes; linked infant death endpoint; SINASC endpoint definition |
| Predictors and leakage control | Methods: Input variables; SINASC section; Figure 1 |
| Missing data | Methods: Input variables; SSL encoder preprocessing |
| Sample size/events | Results: Calendar-time cohort and analytic endpoints |
| Model specification | Methods: masked tabular SSL encoder; supervised baselines; calibration |
| Training/development/test split | Methods: Data source and study design; Figure 1 |
| Hyperparameter/model selection | Methods: SSL encoder; phenotype discovery; stability and sensitivity analyses |
| Calibration | Methods: downstream models and calibration; Results: calibration |
| Performance measures | Methods: AUPRC, AUROC, Brier, ECE, top-risk enrichment, decision curves |
| Internal/temporal validation | Methods and Results: 2024 temporal testing |
| External/transport evaluation | Methods and Results: linked severe-endpoint transfer and SINASC stress-test |
| Model interpretation | Methods and Results: phenotype profiles, cause-specific infant death, PCA sensitivity |
| Clinical utility | Methods and Results: top-risk enrichment, number needed to evaluate, decision curves |
| Reproducibility | Methods: software and reproducibility; Additional file 1; Code availability |
| Limitations | Discussion limitations paragraph |
