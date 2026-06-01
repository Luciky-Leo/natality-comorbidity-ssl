# Natality Comorbidity SSL

Reproducibility materials for:

**Full-scale self-supervised maternal comorbidity phenotyping and cross-registry stress-testing in public birth records**

This repository contains source code, manuscript source files, source-data tables, and selected model artifacts for a public-registry study using CDC/NCHS Natality public-use records and Brazil OpenDataSUS/SINASC public-use records.

## Scope

The study uses leakage-controlled registry-defined maternal, obstetric, anthropometric, prenatal-care, cardiometabolic, infectious, infertility/ART, and missingness variables to train a masked tabular self-supervised encoder. The learned embeddings are used for phenotype discovery, calibrated maternal and neonatal risk enrichment, linked infant death transfer analysis, and SINASC registry stress-testing.

The repository is intended to support manuscript review and post-publication reproducibility. It does **not** redistribute individual-level CDC/NCHS or SINASC records.

## Data sources

Raw public-use files should be obtained from the official sources:

- CDC/NCHS Vital Statistics Online Data Access: https://www.cdc.gov/nchs/data_access/Vitalstatsonline.htm
- CDC/NCHS Natality public-use file documentation
- CDC/NCHS linked birth/infant death public-use file documentation
- Brazil OpenDataSUS/SINASC public-use files and structure dictionary: https://opendatasus.saude.gov.br/

The scripts in this repository expect locally downloaded public-use records and regenerate the analytic matrices, embeddings, source-data tables, and figures.

## Repository layout

- `config/`: field manifests, endpoint definitions, and project configuration.
- `scripts/`: data parsing, SSL training, embedding export, downstream models, external registry stress-test, and figure-building scripts.
- `source_data/`: manuscript source-data tables derived from public-use records.
- `manuscript/`: BMC Medical Informatics and Decision Making LaTeX source, figures, bibliography, and check PDF.
- `supplement/`: additional-file archive and reporting-checklist mapping used for journal submission.
- `docs/`: selected protocol and reproducibility reports.

## Reproducibility notes

The analysis was implemented in Python using pandas, pyarrow, scikit-learn, LightGBM, PyTorch, NumPy, and matplotlib. The main random seed was `20260525`.

Raw individual-level public-use records are intentionally excluded. Large intermediate matrices and full embedding files are also excluded from the repository; they can be regenerated from official public-use sources using the included scripts.

## License

Code and reusable materials in this repository are released under the MIT License. Third-party public-use data remain governed by their source providers' terms and documentation.

## Citation

Please cite the manuscript and the archived Zenodo release when available. The Zenodo DOI should be generated from a GitHub release after the final repository version is pushed.
