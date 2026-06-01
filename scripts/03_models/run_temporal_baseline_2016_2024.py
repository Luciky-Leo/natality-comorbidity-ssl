#!/usr/bin/env python
"""Temporal baseline validation using 2016-2022 train, 2023 dev, 2024 test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "processed" / "natality_2016_2024_split_manifest.csv"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "11_temporal_baseline_report.md"

PRIMARY_ENDPOINTS = [
    "outcome_maternal_morbidity_core",
    "outcome_severe_neonatal_no_nicu",
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
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--endpoints", nargs="*", default=PRIMARY_ENDPOINTS)
    parser.add_argument(
        "--feature-sets",
        nargs="*",
        default=["comorbidity_only", "all_inputs"],
        choices=["comorbidity_only", "all_inputs"],
    )
    parser.add_argument("--max-train-per-year", type=int, default=500_000)
    parser.add_argument("--max-dev-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--lgbm-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260525)
    return parser.parse_args()


def feature_sets_from_schema(path: Path) -> dict[str, list[str]]:
    names = pq.ParquetFile(path).schema.names
    all_inputs = [
        column
        for column in names
        if column.startswith("input_") or column.startswith("missing_input_")
    ]
    base_names = {f"input_{name}" for name in COMORBIDITY_FEATURE_BASES}
    missing_names = {f"missing_input_{name}" for name in COMORBIDITY_FEATURE_BASES}
    comorbidity = [
        column for column in all_inputs if column in base_names or column in missing_names
    ]
    return {"all_inputs": all_inputs, "comorbidity_only": comorbidity}


def load_split(
    manifest: pd.DataFrame,
    split: str,
    columns: list[str],
    seed: int,
    max_per_year: int | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    frames = []
    for row in manifest[manifest["split"] == split].to_dict("records"):
        path = Path(row["path"])
        frame = pq.read_table(path, columns=columns).to_pandas()
        if max_per_year is not None and len(frame) > max_per_year:
            frame = frame.sample(n=max_per_year, random_state=seed + int(row["year"]))
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if max_rows is not None and len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=seed)
    return data


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


def fit_platt(y_dev: np.ndarray, p_dev: np.ndarray) -> LogisticRegression:
    logits = np.log(clip_prob(p_dev) / (1 - clip_prob(p_dev))).reshape(-1, 1)
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(logits, y_dev)
    return model


def apply_platt(model: LogisticRegression, prob: np.ndarray) -> np.ndarray:
    logits = np.log(clip_prob(prob) / (1 - clip_prob(prob))).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def calibration_bins(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    order = np.argsort(prob)
    splits = np.array_split(order, n_bins)
    rows = []
    n = len(y_true)
    for i, idx in enumerate(splits, start=1):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "bin": i,
                "n": int(len(idx)),
                "weight": float(len(idx) / n),
                "mean_pred": float(np.mean(prob[idx])),
                "event_rate": float(np.mean(y_true[idx])),
            }
        )
    return pd.DataFrame(rows)


def ece(y_true: np.ndarray, prob: np.ndarray) -> float:
    bins = calibration_bins(y_true, prob)
    return float((bins["weight"] * (bins["event_rate"] - bins["mean_pred"]).abs()).sum())


def top_risk_rows(y_true: np.ndarray, prob: np.ndarray):
    prevalence = float(np.mean(y_true))
    total_events = int(np.sum(y_true))
    order = np.argsort(-prob)
    rows = []
    for fraction in [0.01, 0.05, 0.10]:
        k = max(1, int(round(len(y_true) * fraction)))
        idx = order[:k]
        events = int(np.sum(y_true[idx]))
        event_rate = float(np.mean(y_true[idx]))
        rows.append(
            {
                "top_fraction": fraction,
                "top_n": k,
                "event_rate": event_rate,
                "enrichment_over_prevalence": event_rate / prevalence
                if prevalence > 0
                else np.nan,
                "events_captured": events,
                "event_capture_pct": 100 * events / total_events
                if total_events
                else np.nan,
            }
        )
    return rows


def metric_row(endpoint: str, feature_set: str, probability_type: str, y, prob):
    prevalence = float(np.mean(y))
    auprc = float(average_precision_score(y, prob))
    return {
        "endpoint": endpoint,
        "feature_set": feature_set,
        "model": "lightgbm",
        "probability_type": probability_type,
        "n_test": int(len(y)),
        "events_test": int(np.sum(y)),
        "prevalence_test": prevalence,
        "auroc": float(roc_auc_score(y, prob)),
        "auprc": auprc,
        "auprc_over_prevalence": auprc / prevalence if prevalence > 0 else np.nan,
        "brier": float(brier_score_loss(y, prob)),
        "ece_10": ece(y, prob),
    }


def main() -> None:
    args = parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.split_manifest)
    first_path = Path(manifest.iloc[0]["path"])
    feature_sets = feature_sets_from_schema(first_path)

    needed_features = sorted(
        {feature for name in args.feature_sets for feature in feature_sets[name]}
    )
    columns = needed_features + args.endpoints

    print("load train")
    train = load_split(
        manifest,
        "train",
        columns,
        seed=args.seed,
        max_per_year=args.max_train_per_year,
    )
    print(f"train rows {len(train):,}")
    print("load dev")
    dev = load_split(
        manifest,
        "development",
        columns,
        seed=args.seed,
        max_rows=args.max_dev_rows,
    )
    print(f"dev rows {len(dev):,}")
    print("load test")
    test = load_split(manifest, "test", columns, seed=args.seed, max_rows=args.max_test_rows)
    print(f"test rows {len(test):,}")

    metrics_rows = []
    enrichment_rows = []
    calibration_rows = []
    importance_rows = []
    metadata_rows = []

    for endpoint in args.endpoints:
        train_ep = train[train[endpoint].notna()].copy()
        dev_ep = dev[dev[endpoint].notna()].copy()
        test_ep = test[test[endpoint].notna()].copy()
        y_train = train_ep[endpoint].astype("int8").to_numpy()
        y_dev = dev_ep[endpoint].astype("int8").to_numpy()
        y_test = test_ep[endpoint].astype("int8").to_numpy()
        print(
            f"{endpoint}: train={len(y_train):,} events={int(y_train.sum()):,}; dev={len(y_dev):,} events={int(y_dev.sum()):,}; test={len(y_test):,} events={int(y_test.sum()):,}",
            flush=True,
        )

        for feature_set in args.feature_sets:
            features = feature_sets[feature_set]
            X_train = prepare_lgbm_frame(train_ep[features])
            X_dev = prepare_lgbm_frame(dev_ep[features])
            X_test = prepare_lgbm_frame(test_ep[features])
            model = make_lgbm(args.seed, args.lgbm_estimators)
            print(f"fit {endpoint} | {feature_set}", flush=True)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_dev, y_dev)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            p_dev_raw = model.predict_proba(X_dev)[:, 1]
            p_test_raw = model.predict_proba(X_test)[:, 1]
            platt = fit_platt(y_dev, p_dev_raw)
            p_test_platt = apply_platt(platt, p_test_raw)

            for prob_type, prob in [("raw", p_test_raw), ("platt", p_test_platt)]:
                metrics_rows.append(metric_row(endpoint, feature_set, prob_type, y_test, prob))
                bins = calibration_bins(y_test, prob)
                bins.insert(0, "probability_type", prob_type)
                bins.insert(0, "feature_set", feature_set)
                bins.insert(0, "endpoint", endpoint)
                calibration_rows.extend(bins.to_dict("records"))
                for row in top_risk_rows(y_test, prob):
                    row.update(
                        {
                            "endpoint": endpoint,
                            "feature_set": feature_set,
                            "model": "lightgbm",
                            "probability_type": prob_type,
                        }
                    )
                    enrichment_rows.append(row)

            booster = model.booster_
            gains = booster.feature_importance(importance_type="gain")
            splits = booster.feature_importance(importance_type="split")
            for feature, gain, split_count in zip(features, gains, splits):
                importance_rows.append(
                    {
                        "endpoint": endpoint,
                        "feature_set": feature_set,
                        "model": "lightgbm",
                        "feature": feature,
                        "importance_gain": float(gain),
                        "importance_split": int(split_count),
                    }
                )
            metadata_rows.append(
                {
                    "endpoint": endpoint,
                    "feature_set": feature_set,
                    "train_n": int(len(y_train)),
                    "dev_n": int(len(y_dev)),
                    "test_n": int(len(y_test)),
                    "train_events": int(y_train.sum()),
                    "dev_events": int(y_dev.sum()),
                    "test_events": int(y_test.sum()),
                    "n_features": int(len(features)),
                    "max_train_per_year": args.max_train_per_year,
                    "max_dev_rows": args.max_dev_rows,
                    "max_test_rows": args.max_test_rows,
                }
            )

    metrics = pd.DataFrame(metrics_rows)
    enrichment = pd.DataFrame(enrichment_rows)
    calibration = pd.DataFrame(calibration_rows)
    importance = pd.DataFrame(importance_rows)
    metadata = pd.DataFrame(metadata_rows)

    metrics_path = args.tables / "temporal_baseline_2016_2024_metrics.csv"
    enrichment_path = args.tables / "temporal_baseline_2016_2024_top_risk_enrichment.csv"
    calibration_path = args.tables / "temporal_baseline_2016_2024_calibration_bins.csv"
    importance_path = args.tables / "temporal_baseline_2016_2024_lightgbm_importance.csv"
    metadata_path = args.tables / "temporal_baseline_2016_2024_metadata.csv"
    metrics.to_csv(metrics_path, index=False)
    enrichment.to_csv(enrichment_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    importance.to_csv(importance_path, index=False)
    metadata.to_csv(metadata_path, index=False)

    platt = metrics[metrics["probability_type"] == "platt"].sort_values(
        ["endpoint", "auprc"], ascending=[True, False]
    )
    report_lines = [
        "# Temporal Baseline Report",
        "",
        "Training split: 2016-2022.",
        "Development/calibration split: 2023.",
        "Final test split: 2024.",
        "",
        f"Max train rows per year: {args.max_train_per_year}",
        f"Max dev rows: {args.max_dev_rows}",
        f"Max test rows: {args.max_test_rows}",
        "",
        "## Platt-Calibrated 2024 Test Metrics",
        "",
        "| Endpoint | Feature set | Events/test | Prev. | AUROC | AUPRC | AUPRC/Prev. | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in platt.to_dict("records"):
        report_lines.append(
            "| {endpoint} | {feature_set} | {events_test:,} | {prevalence_test:.5f} | {auroc:.4f} | {auprc:.4f} | {auprc_over_prevalence:.2f} | {brier:.5f} | {ece_10:.5f} |".format(
                **row
            )
        )
    report_lines.extend(
        [
            "",
            "## Output Tables",
            "",
            f"- `{metrics_path}`",
            f"- `{enrichment_path}`",
            f"- `{calibration_path}`",
            f"- `{importance_path}`",
            f"- `{metadata_path}`",
            "",
            "## Interpretation Boundary",
            "",
            "This temporal baseline uses sampled training rows by default. It is stronger than single-year random splitting because 2024 is untouched, but final baseline estimates should be rerun with larger or full training data if compute permits.",
        ]
    )
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    run_meta = {
        "split_manifest": str(args.split_manifest),
        "endpoints": args.endpoints,
        "feature_sets": args.feature_sets,
        "max_train_per_year": args.max_train_per_year,
        "max_dev_rows": args.max_dev_rows,
        "max_test_rows": args.max_test_rows,
        "lgbm_estimators": args.lgbm_estimators,
        "seed": args.seed,
    }
    (args.tables / "temporal_baseline_2016_2024_run_metadata.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )
    print(f"wrote {metrics_path}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
