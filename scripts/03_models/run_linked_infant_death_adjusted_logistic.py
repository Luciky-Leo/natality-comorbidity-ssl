#!/usr/bin/env python
"""Descriptive adjusted logistic models for linked infant death phenotype transfer."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COHORT = PROJECT_ROOT / "data" / "processed" / "linked_infant_death_2023_cohort.parquet"
DEFAULT_ASSIGNMENTS = PROJECT_ROOT / "results" / "objects" / "linked_infant_death_phenotype_assignments_full2016_2022_mask035_d48_l2_cuda.parquet"
DEFAULT_ZIP = PROJECT_ROOT / "data" / "raw_linked_birth_infant_death" / "2024PE2023CO.zip"
DEFAULT_FIELDS = PROJECT_ROOT / "config" / "nat2024_smoke_fields.csv"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "39_linked_infant_death_adjusted_logistic_report.md"

DENOM_NAME_PREFIX = "VS2023LINK.Public.USDENPUB"

OUTCOMES = [
    "outcome_infant_death",
    "outcome_neonatal_death_lt28d",
    "outcome_early_neonatal_death_lt7d",
    "outcome_postneonatal_death_28d_1y",
]

OUTCOME_LABELS = {
    "outcome_infant_death": "Infant death",
    "outcome_neonatal_death_lt28d": "Neonatal death <28d",
    "outcome_early_neonatal_death_lt7d": "Early neonatal death <7d",
    "outcome_postneonatal_death_28d_1y": "Postneonatal death 28d-1y",
}

MATERNAL_NUMERIC = [
    ("maternal_age_per5y", "input_MAGER", 5.0),
    ("bmi_per5", "input_BMI", 5.0),
    ("prenatal_visits_per5", "input_PREVIS", 5.0),
    ("weight_gain_per10", "input_WTGAIN", 10.0),
]

MATERNAL_BINARY = [
    ("prepregnancy_diabetes", "input_RF_PDIAB"),
    ("gestational_diabetes", "input_RF_GDIAB"),
    ("prepregnancy_hypertension", "input_RF_PHYPE"),
    ("gestational_hypertension", "input_RF_GHYPE"),
    ("eclampsia", "input_RF_EHYPE"),
    ("previous_preterm_birth", "input_RF_PPTERM"),
    ("infertility_treatment", "input_RF_INFTR"),
    ("fertility_drug", "input_RF_FEDRG"),
    ("assisted_reproductive_technology", "input_RF_ARTEC"),
    ("previous_cesarean", "input_RF_CESAR"),
    ("smoking", "input_CIG_REC"),
    ("chlamydia", "input_IP_CHLAM"),
    ("hepatitis_b", "input_IP_HEPB"),
    ("hepatitis_c", "input_IP_HEPC"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--linked-zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--fields", type=Path, default=DEFAULT_FIELDS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-tag", default="full2016_2022_mask035_d48_l2_cuda")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


def load_fields(path: Path) -> dict[str, dict[str, object]]:
    out = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            row["start"] = int(row["start"])
            row["end"] = int(row["end"])
            row["missing_set"] = {value for value in row["missing_values"].split("|") if value}
            out[row["name"]] = row
    return out


def raw_slice(raw: bytes, start: int, end: int) -> str:
    return raw[start - 1 : end].decode("latin1").strip()


def raw_value(raw: bytes, field: dict[str, object]) -> str:
    return raw_slice(raw, int(field["start"]), int(field["end"]))


def parse_int(value: str, missing: bool) -> int | None:
    if missing:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def is_missing(value: str, field: dict[str, object]) -> bool:
    return value == "" or value in field["missing_set"]


def find_member(names: list[str], prefix: str) -> str:
    matches = [name for name in names if name.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one member with prefix {prefix}, got {matches}")
    return matches[0]


def extract_birth_status(zip_path: Path, fields_path: Path, chunk_size: int) -> pd.DataFrame:
    fields = load_fields(fields_path)
    rows: list[dict[str, object]] = []
    chunks: list[pd.DataFrame] = []
    record_id = 0
    with zipfile.ZipFile(zip_path) as zf:
        member = find_member(zf.namelist(), DENOM_NAME_PREFIX)
        with zf.open(member, "r") as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                re_status = raw_slice(raw, 104, 104)
                if re_status.isdigit() and int(re_status) >= 4:
                    continue
                record_id += 1
                gest_value = raw_value(raw, fields["COMBGEST"])
                # Linked birth/infant death denominator files use BRTHWGT
                # (positions 512-515) rather than the Natality DBWT name.
                bw_value = raw_slice(raw, 512, 515)
                rows.append(
                    {
                        "record_id": record_id,
                        "adjust_gest_weeks_combined": parse_int(gest_value, is_missing(gest_value, fields["COMBGEST"])),
                        "adjust_birthweight_g": parse_int(bw_value, bw_value in {"", "9999"}),
                    }
                )
                if len(rows) >= chunk_size:
                    chunks.append(pd.DataFrame.from_records(rows))
                    rows.clear()
    if rows:
        chunks.append(pd.DataFrame.from_records(rows))
    return pd.concat(chunks, ignore_index=True)


def yes(series: pd.Series) -> pd.Series:
    return series.fillna("N").astype(str).eq("Y").astype("float64")


def numeric_scaled(frame: pd.DataFrame, source: str, scale: float) -> tuple[pd.Series, pd.Series]:
    raw = pd.to_numeric(frame[source], errors="coerce")
    missing = raw.isna().astype("float64")
    median = raw.median()
    if pd.isna(median):
        median = 0.0
    return raw.fillna(median).astype("float64") / scale, missing


def build_design(frame: pd.DataFrame, include_birth_status: bool) -> tuple[np.ndarray, list[str]]:
    features: list[pd.Series] = []
    names: list[str] = []

    for phenotype in [0, 1]:
        features.append(frame["phenotype"].eq(phenotype).astype("float64"))
        names.append(f"phenotype_{phenotype}_vs_2")

    for name, source, scale in MATERNAL_NUMERIC:
        value, missing = numeric_scaled(frame, source, scale)
        features.append(value)
        names.append(name)
        if missing.any():
            features.append(missing)
            names.append(f"{name}_missing")

    for name, source in MATERNAL_BINARY:
        features.append(yes(frame[source]))
        names.append(name)

    plurality = pd.to_numeric(frame["input_DPLURAL"], errors="coerce")
    features.append((plurality > 1).fillna(False).astype("float64"))
    names.append("multiple_gestation")
    features.append(frame["input_SEX"].fillna("").astype(str).eq("M").astype("float64"))
    names.append("male_infant")

    if include_birth_status:
        gest = pd.to_numeric(frame["adjust_gest_weeks_combined"], errors="coerce")
        bw = pd.to_numeric(frame["adjust_birthweight_g"], errors="coerce")
        gest_median = gest.median()
        if pd.isna(gest_median):
            gest_median = 0.0
        bw_median = bw.median()
        if pd.isna(bw_median):
            bw_median = 0.0
        features.append(gest.fillna(gest_median).astype("float64"))
        names.append("gestational_age_weeks")
        features.append(gest.isna().astype("float64"))
        names.append("gestational_age_missing")
        features.append((bw.fillna(bw_median).astype("float64") / 500.0))
        names.append("birthweight_per500g")
        features.append(bw.isna().astype("float64"))
        names.append("birthweight_missing")

    design = pd.concat(features, axis=1)
    design.columns = names
    design = design.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return design.to_numpy(dtype=np.float64, copy=False), names


def coefficient_covariance(x: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    x_intercept = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    weights = probabilities * (1.0 - probabilities)
    hessian = np.zeros((x_intercept.shape[1], x_intercept.shape[1]), dtype=np.float64)
    batch = 250_000
    for start in range(0, len(x_intercept), batch):
        stop = min(start + batch, len(x_intercept))
        xb = x_intercept[start:stop]
        wb = weights[start:stop]
        hessian += xb.T @ (xb * wb[:, None])
    return np.linalg.pinv(hessian)


def fit_one(
    frame: pd.DataFrame,
    outcome: str,
    model_name: str,
    include_birth_status: bool,
    max_iter: int,
) -> pd.DataFrame:
    x, names = build_design(frame, include_birth_status=include_birth_status)
    y = frame[outcome].to_numpy(dtype=np.int8)
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=max_iter)
    model.fit(x, y)
    probabilities = model.predict_proba(x)[:, 1]
    cov = coefficient_covariance(x, probabilities)
    coefficients = np.concatenate([model.intercept_, model.coef_.ravel()])
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    terms = ["intercept", *names]
    rows = []
    for term, beta, stderr in zip(terms, coefficients, se, strict=True):
        rows.append(
            {
                "outcome": outcome,
                "outcome_label": OUTCOME_LABELS[outcome],
                "model": model_name,
                "term": term,
                "coefficient": float(beta),
                "standard_error": float(stderr),
                "odds_ratio": float(np.exp(beta)),
                "ci_low": float(np.exp(beta - 1.96 * stderr)),
                "ci_high": float(np.exp(beta + 1.96 * stderr)),
                "n": int(len(y)),
                "events": int(y.sum()),
                "prevalence": float(y.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_tag}" if args.output_tag else ""

    covariate_cols = [
        "record_id",
        *OUTCOMES,
        *[source for _, source, _ in MATERNAL_NUMERIC],
        *[source for _, source in MATERNAL_BINARY],
        "input_DPLURAL",
        "input_SEX",
    ]
    cohort = pd.read_parquet(args.cohort, columns=covariate_cols)
    assignments = pd.read_parquet(args.assignments, columns=["record_id", "phenotype"])
    birth_status = extract_birth_status(args.linked_zip, args.fields, args.chunk_size)
    frame = cohort.merge(assignments, on="record_id", how="inner").merge(birth_status, on="record_id", how="left")
    if len(frame) != len(cohort):
        raise RuntimeError(f"merge changed row count: cohort={len(cohort):,}, merged={len(frame):,}")

    all_rows = []
    for outcome in OUTCOMES:
        all_rows.append(
            fit_one(
                frame,
                outcome=outcome,
                model_name="maternal_registry_adjusted",
                include_birth_status=False,
                max_iter=args.max_iter,
            )
        )
        all_rows.append(
            fit_one(
                frame,
                outcome=outcome,
                model_name="birth_status_adjusted_descriptive",
                include_birth_status=True,
                max_iter=args.max_iter,
            )
        )
    results = pd.concat(all_rows, ignore_index=True)
    table_path = args.tables / f"linked_infant_death_adjusted_logistic{suffix}.csv"
    phenotype_path = args.tables / f"linked_infant_death_adjusted_logistic_phenotype_terms{suffix}.csv"
    metadata_path = args.tables / f"linked_infant_death_adjusted_logistic_metadata{suffix}.json"
    results.to_csv(table_path, index=False)
    phenotype = results[results["term"].str.startswith("phenotype_")].copy()
    phenotype.to_csv(phenotype_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "cohort": str(args.cohort),
                "assignments": str(args.assignments),
                "linked_zip": str(args.linked_zip),
                "rows": int(len(frame)),
                "models": ["maternal_registry_adjusted", "birth_status_adjusted_descriptive"],
                "birth_status_note": "Birthweight and gestational age are descriptive attenuation variables, not pre-delivery predictors.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Linked Infant Death Adjusted Logistic Regression",
        "",
        "Fixed SSL phenotypes were evaluated in descriptive logistic models for linked infant death endpoints. The maternal-registry model adjusted for available public-use maternal and pregnancy variables. The birth-status model additionally adjusted for gestational age and birthweight, which should be interpreted as an attenuation/sensitivity model rather than a deployable pre-delivery prediction model.",
        "",
        f"- linked cohort rows: {len(frame):,}",
        f"- output table: `{table_path}`",
        f"- phenotype-only table: `{phenotype_path}`",
        "",
        "## Phenotype Odds Ratios",
        "",
        "| Outcome | Model | Term | OR | 95% CI | Events |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in phenotype.to_dict("records"):
        lines.append(
            "| {outcome_label} | {model} | {term} | {orv:.3f} | {lo:.3f}-{hi:.3f} | {events:,} |".format(
                outcome_label=row["outcome_label"],
                model=row["model"],
                term=row["term"],
                orv=float(row["odds_ratio"]),
                lo=float(row["ci_low"]),
                hi=float(row["ci_high"]),
                events=int(row["events"]),
            )
        )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {table_path}")
    print(f"wrote {phenotype_path}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
