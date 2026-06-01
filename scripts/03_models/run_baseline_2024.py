#!/usr/bin/env python
"""Run 2024 baseline models before self-supervised learning.

The goal is feasibility and signal auditing, not final model selection. The
script uses a train/calibration/test split, evaluates raw and Platt-recalibrated
probabilities, and reports AUPRC relative to the event-rate baseline.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COHORT = PROJECT_ROOT / "data" / "processed" / "nat2024_analytic_cohort.parquet"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "08_baseline_2024_report.md"

PRIMARY_ENDPOINTS = [
    "outcome_maternal_morbidity_core",
    "outcome_severe_neonatal_no_nicu",
]

SECONDARY_ENDPOINTS = [
    "outcome_maternal_morbidity_extended",
    "outcome_severe_neonatal_plus_nicu",
    "outcome_broad_neonatal_composite",
]

COMORBIDITY_FEATURE_BASES = [
    "MAGER",
    "MAGER9",
    "BMI",
    "BMI_R",
    "WTGAIN",
    "WTGAIN_REC",
    "CIG0_R",
    "CIG1_R",
    "CIG2_R",
    "CIG3_R",
    "CIG_REC",
    "RF_PDIAB",
    "RF_GDIAB",
    "RF_PHYPE",
    "RF_GHYPE",
    "RF_EHYPE",
    "RF_PPTERM",
    "RF_INFTR",
    "RF_FEDRG",
    "RF_ARTEC",
    "RF_CESAR",
    "RF_CESARN",
    "NO_RISKS",
    "IP_GON",
    "IP_SYPH",
    "IP_CHLAM",
    "IP_HEPB",
    "IP_HEPC",
    "NO_INFEC",
    "DPLURAL",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--endpoints", nargs="*", default=PRIMARY_ENDPOINTS)
    parser.add_argument(
        "--include-secondary",
        action="store_true",
        help="Also model secondary/sensitivity endpoints.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["sgd_logistic_l2", "lightgbm"],
        choices=["sgd_logistic_l2", "lightgbm"],
    )
    parser.add_argument(
        "--feature-sets",
        nargs="*",
        default=["comorbidity_only", "all_inputs"],
        choices=["comorbidity_only", "all_inputs"],
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--cal-frac", type=float, default=0.20)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--lgbm-estimators", type=int, default=250)
    return parser.parse_args()


def onehot_encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=True)


def make_feature_sets(columns: list[str]) -> dict[str, list[str]]:
    all_inputs = [
        column
        for column in columns
        if column.startswith("input_") or column.startswith("missing_input_")
    ]
    base_names = {f"input_{name}" for name in COMORBIDITY_FEATURE_BASES}
    missing_names = {f"missing_input_{name}" for name in COMORBIDITY_FEATURE_BASES}
    comorbidity = [
        column for column in all_inputs if column in base_names or column in missing_names
    ]
    return {
        "all_inputs": all_inputs,
        "comorbidity_only": comorbidity,
    }


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    parquet_file = pq.ParquetFile(args.cohort)
    columns = parquet_file.schema.names
    endpoints = list(args.endpoints)
    if args.include_secondary:
        endpoints.extend(SECONDARY_ENDPOINTS)
    endpoints = list(dict.fromkeys(endpoints))
    feature_sets = make_feature_sets(columns)
    selected_features = sorted({item for name in args.feature_sets for item in feature_sets[name]})
    selected_columns = ["record_id"] + selected_features + endpoints
    table = pq.read_table(args.cohort, columns=selected_columns)
    data = table.to_pandas()
    if args.max_rows is not None and args.max_rows < len(data):
        data = data.sample(n=args.max_rows, random_state=args.seed).sort_index()

    for column in selected_features:
        if pd.api.types.is_object_dtype(data[column]) or isinstance(
            data[column].dtype, pd.StringDtype
        ):
            data[column] = data[column].astype("category")
        elif pd.api.types.is_bool_dtype(data[column]):
            data[column] = data[column].astype("int8")

    return data, feature_sets, endpoints


def split_indices(y: pd.Series, seed: int, train_frac: float, cal_frac: float, test_frac: float):
    total = train_frac + cal_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError("train/cal/test fractions must sum to 1")
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=1 - train_frac,
        random_state=seed,
        stratify=y,
    )
    relative_test = test_frac / (cal_frac + test_frac)
    cal_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test,
        random_state=seed + 1,
        stratify=y.iloc[temp_idx],
    )
    return train_idx, cal_idx, test_idx


def column_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = []
    categorical = []
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(
            frame[column]
        ):
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def make_logistic_pipeline(frame: pd.DataFrame) -> Pipeline:
    numeric_cols, categorical_cols = column_types(frame)
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                        ),
                        ("onehot", onehot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=1.0)
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=50,
        tol=1e-4,
        random_state=20260525,
        n_jobs=-1,
        early_stopping=False,
        class_weight="balanced",
        average=True,
    )
    return Pipeline([("preprocess", preprocessor), ("model", classifier)])


def prepare_lgbm_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_bool_dtype(out[column]):
            out[column] = out[column].astype("int8")
        elif pd.api.types.is_object_dtype(out[column]) or isinstance(
            out[column].dtype, pd.StringDtype
        ):
            out[column] = out[column].astype("category")
    return out


def make_lgbm(seed: int, n_estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=1000,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )


def clip_prob(prob: np.ndarray) -> np.ndarray:
    return np.clip(prob, 1e-6, 1 - 1e-6)


def fit_platt(y_cal: np.ndarray, p_cal: np.ndarray) -> LogisticRegression:
    logits = np.log(clip_prob(p_cal) / (1 - clip_prob(p_cal))).reshape(-1, 1)
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(logits, y_cal)
    return model


def apply_platt(model: LogisticRegression, prob: np.ndarray) -> np.ndarray:
    logits = np.log(clip_prob(prob) / (1 - clip_prob(prob))).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def calibration_slope_intercept(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    logits = np.log(clip_prob(prob) / (1 - clip_prob(prob))).reshape(-1, 1)
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    model.fit(logits, y_true)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = calibration_bins(y_true, prob, n_bins=n_bins)
    return float((bins["weight"] * (bins["event_rate"] - bins["mean_pred"]).abs()).sum())


def calibration_bins(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    order = np.argsort(prob)
    splits = np.array_split(order, n_bins)
    rows = []
    n = len(y_true)
    for i, idx in enumerate(splits, start=1):
        if len(idx) == 0:
            continue
        y_bin = y_true[idx]
        p_bin = prob[idx]
        rows.append(
            {
                "bin": i,
                "n": int(len(idx)),
                "weight": float(len(idx) / n),
                "mean_pred": float(np.mean(p_bin)),
                "event_rate": float(np.mean(y_bin)),
                "pred_min": float(np.min(p_bin)),
                "pred_max": float(np.max(p_bin)),
            }
        )
    return pd.DataFrame(rows)


def top_risk_enrichment(y_true: np.ndarray, prob: np.ndarray, fractions=(0.01, 0.05, 0.10)):
    prevalence = float(np.mean(y_true))
    order = np.argsort(-prob)
    total_events = int(np.sum(y_true))
    rows = []
    for fraction in fractions:
        k = max(1, int(round(len(y_true) * fraction)))
        idx = order[:k]
        event_rate = float(np.mean(y_true[idx]))
        events_captured = int(np.sum(y_true[idx]))
        rows.append(
            {
                "top_fraction": fraction,
                "top_n": k,
                "event_rate": event_rate,
                "enrichment_over_prevalence": event_rate / prevalence
                if prevalence > 0
                else np.nan,
                "events_captured": events_captured,
                "event_capture_pct": 100 * events_captured / total_events
                if total_events > 0
                else np.nan,
            }
        )
    return rows


def metric_row(
    endpoint: str,
    feature_set: str,
    model_name: str,
    probability_type: str,
    y_true: np.ndarray,
    prob: np.ndarray,
) -> dict[str, object]:
    prevalence = float(np.mean(y_true))
    intercept, slope = calibration_slope_intercept(y_true, prob)
    auprc = float(average_precision_score(y_true, prob))
    return {
        "endpoint": endpoint,
        "feature_set": feature_set,
        "model": model_name,
        "probability_type": probability_type,
        "n_test": int(len(y_true)),
        "events_test": int(np.sum(y_true)),
        "prevalence_test": prevalence,
        "auroc": float(roc_auc_score(y_true, prob)),
        "auprc": auprc,
        "auprc_over_prevalence": auprc / prevalence if prevalence > 0 else np.nan,
        "brier": float(brier_score_loss(y_true, prob)),
        "ece_10": expected_calibration_error(y_true, prob, n_bins=10),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def plot_metrics(metrics: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    raw = metrics[metrics["probability_type"] == "platt"]
    labels = (
        raw["endpoint"].str.replace("outcome_", "", regex=False)
        + "\n"
        + raw["feature_set"]
        + "\n"
        + raw["model"]
    )

    plt.figure(figsize=(max(8, 0.42 * len(raw)), 5))
    plt.bar(np.arange(len(raw)), raw["auprc"], color="#3568A8")
    plt.scatter(np.arange(len(raw)), raw["prevalence_test"], color="#D95F02", zorder=3)
    plt.xticks(np.arange(len(raw)), labels, rotation=75, ha="right", fontsize=8)
    plt.ylabel("AUPRC")
    plt.title("2024 baseline AUPRC vs event-rate baseline")
    plt.tight_layout()
    plt.savefig(figure_dir / "nat2024_baseline_auprc.png", dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)

    data, feature_sets, endpoints = load_data(args)
    metadata_rows = []
    metrics_rows = []
    calibration_rows = []
    enrichment_rows = []
    importance_rows = []

    for endpoint in endpoints:
        endpoint_data = data[data[endpoint].notna()].copy()
        y = endpoint_data[endpoint].astype("int8")
        if y.nunique() < 2:
            print(f"skip {endpoint}: only one class")
            continue
        train_idx, cal_idx, test_idx = split_indices(
            y, args.seed, args.train_frac, args.cal_frac, args.test_frac
        )
        y_train = y.iloc[train_idx].to_numpy()
        y_cal = y.iloc[cal_idx].to_numpy()
        y_test = y.iloc[test_idx].to_numpy()

        print(
            f"{endpoint}: train={len(train_idx):,}, cal={len(cal_idx):,}, test={len(test_idx):,}, test_events={int(y_test.sum()):,}",
            flush=True,
        )

        for feature_set_name in args.feature_sets:
            columns = feature_sets[feature_set_name]
            X = endpoint_data[columns]

            for model_name in args.models:
                print(f"fit {endpoint} | {feature_set_name} | {model_name}", flush=True)
                if model_name == "sgd_logistic_l2":
                    model = make_logistic_pipeline(X.iloc[train_idx])
                    model.fit(X.iloc[train_idx], y_train)
                    p_cal_raw = model.predict_proba(X.iloc[cal_idx])[:, 1]
                    p_test_raw = model.predict_proba(X.iloc[test_idx])[:, 1]
                elif model_name == "lightgbm":
                    X_lgbm = prepare_lgbm_frame(X)
                    model = make_lgbm(args.seed, args.lgbm_estimators)
                    model.fit(
                        X_lgbm.iloc[train_idx],
                        y_train,
                        eval_set=[(X_lgbm.iloc[cal_idx], y_cal)],
                        eval_metric="auc",
                        callbacks=[lgb.early_stopping(30, verbose=False)],
                    )
                    p_cal_raw = model.predict_proba(X_lgbm.iloc[cal_idx])[:, 1]
                    p_test_raw = model.predict_proba(X_lgbm.iloc[test_idx])[:, 1]
                    booster = model.booster_
                    gains = booster.feature_importance(importance_type="gain")
                    splits = booster.feature_importance(importance_type="split")
                    for feature, gain, split in zip(columns, gains, splits):
                        importance_rows.append(
                            {
                                "endpoint": endpoint,
                                "feature_set": feature_set_name,
                                "model": model_name,
                                "feature": feature,
                                "importance_gain": float(gain),
                                "importance_split": int(split),
                            }
                        )
                else:
                    raise ValueError(model_name)

                platt = fit_platt(y_cal, p_cal_raw)
                p_test_platt = apply_platt(platt, p_test_raw)

                for probability_type, probability in [
                    ("raw", p_test_raw),
                    ("platt", p_test_platt),
                ]:
                    metrics_rows.append(
                        metric_row(
                            endpoint,
                            feature_set_name,
                            model_name,
                            probability_type,
                            y_test,
                            probability,
                        )
                    )
                    bins = calibration_bins(y_test, probability, n_bins=10)
                    bins.insert(0, "probability_type", probability_type)
                    bins.insert(0, "model", model_name)
                    bins.insert(0, "feature_set", feature_set_name)
                    bins.insert(0, "endpoint", endpoint)
                    calibration_rows.extend(bins.to_dict("records"))

                    for row in top_risk_enrichment(y_test, probability):
                        row.update(
                            {
                                "endpoint": endpoint,
                                "feature_set": feature_set_name,
                                "model": model_name,
                                "probability_type": probability_type,
                            }
                        )
                        enrichment_rows.append(row)

                metadata_rows.append(
                    {
                        "endpoint": endpoint,
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "train_n": int(len(train_idx)),
                        "cal_n": int(len(cal_idx)),
                        "test_n": int(len(test_idx)),
                        "train_events": int(y_train.sum()),
                        "cal_events": int(y_cal.sum()),
                        "test_events": int(y_test.sum()),
                        "n_features": int(len(columns)),
                    }
                )

    metrics = pd.DataFrame(metrics_rows)
    calibration = pd.DataFrame(calibration_rows)
    enrichment = pd.DataFrame(enrichment_rows)
    metadata = pd.DataFrame(metadata_rows)
    importance = pd.DataFrame(importance_rows)

    metrics_path = args.tables / "nat2024_baseline_metrics.csv"
    calibration_path = args.tables / "nat2024_baseline_calibration_bins.csv"
    enrichment_path = args.tables / "nat2024_baseline_top_risk_enrichment.csv"
    metadata_path = args.tables / "nat2024_baseline_model_metadata.csv"
    importance_path = args.tables / "nat2024_baseline_lightgbm_importance.csv"

    metrics.to_csv(metrics_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    enrichment.to_csv(enrichment_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    if not importance.empty:
        importance.to_csv(importance_path, index=False)

    plot_metrics(metrics, args.figures)

    platt_metrics = metrics[metrics["probability_type"] == "platt"].sort_values(
        ["endpoint", "auprc"], ascending=[True, False]
    )
    report_lines = [
        "# 2024 Baseline Model Report",
        "",
        f"Cohort: `{args.cohort}`",
        f"Rows loaded: {len(data):,}",
        f"Max rows setting: {args.max_rows}",
        "",
        "## Split Design",
        "",
        f"- train fraction: {args.train_frac}",
        f"- calibration fraction: {args.cal_frac}",
        f"- test fraction: {args.test_frac}",
        "- raw model scores and Platt-recalibrated probabilities are both reported.",
        "",
        "## Platt-Calibrated Test Metrics",
        "",
        "| Endpoint | Feature set | Model | Events/test | Prev. | AUROC | AUPRC | AUPRC/Prev. | Brier | ECE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in platt_metrics.to_dict("records"):
        report_lines.append(
            "| {endpoint} | {feature_set} | {model} | {events_test:,} | {prevalence_test:.5f} | {auroc:.4f} | {auprc:.4f} | {auprc_over_prevalence:.2f} | {brier:.5f} | {ece_10:.5f} |".format(
                **row
            )
        )
    report_lines.extend(
        [
            "",
            "## Output Tables",
            "",
            f"- `{metrics_path}`",
            f"- `{calibration_path}`",
            f"- `{enrichment_path}`",
            f"- `{metadata_path}`",
            f"- `{importance_path}`",
            "",
            "## Output Figure",
            "",
            f"- `{args.figures / 'nat2024_baseline_auprc.png'}`",
            f"- `{args.figures / 'nat2024_baseline_top_risk_enrichment.png'}`",
            f"- `{args.figures / 'nat2024_baseline_lightgbm_importance.png'}`",
            "",
            "## Interpretation Boundary",
            "",
            "These are single-year internal baseline results. They are useful for signal auditing, but final manuscript validation should use the planned 2016-2022 train, 2023 development, and 2024 test split.",
        ]
    )
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    run_metadata = {
        "cohort": str(args.cohort),
        "rows_loaded": int(len(data)),
        "endpoints": endpoints,
        "models": args.models,
        "feature_sets": args.feature_sets,
        "seed": args.seed,
        "max_rows": args.max_rows,
    }
    (args.tables / "nat2024_baseline_run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote {metrics_path}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
