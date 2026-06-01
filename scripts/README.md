# Script Plan

Scripts should be added in this order.

## 00_download

Download official user guides and public-use data files from CDC/NCHS. Keep the raw files unchanged.

## 01_prepare

Parse fixed-width files, extract variable dictionaries, harmonize year-specific variables, and create the analytic cohort.

## 02_ssl

Train masked tabular SSL encoders and save frozen embeddings. Hyperparameters must be selected on the development split only.

## 03_models

Train calibrated risk-enrichment models, phenotype clustering, bootstrap confidence intervals, subgroup/fairness analyses, and sensitivity outcomes.

## 04_figures

Generate manuscript-ready figures and source-data tables from saved model outputs.
