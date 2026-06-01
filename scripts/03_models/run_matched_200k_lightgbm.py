#!/usr/bin/env python
"""Run LightGBM on the exact SSL development/test record IDs."""

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
DEFAULT_DEV_EMB = PROJECT_ROOT / "results" / "objects" / "masked_tabular_ssl_dev_embeddings_200k.parquet"
DEFAULT_TEST_EMB = PROJECT_ROOT / "results" / "objects" / "masked_tabular_ssl_test_embeddings_200k.parquet"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "19_matched_200k_lightgbm_report.md"

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
    parser.add_argument("--dev-embeddings", type=Path, default=DEFAULT_DEV_EMB)
    parser.add_argument("--test-embeddings", type=Path, default=DEFAULT_TEST_EMB)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-train-per-year", type=int, default=500_000)
    parser.add_argument("--lgbm-estimators", type=int, default=300)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--output-tag", default="200k")
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


def load_train(
    manifest: pd.DataFrame,
    columns: list[str],
    seed: int,
    max_per_year: int,
) -> pd.DataFrame:
    frames = []
    for row in manifest[manifest["split"] == "train"].to_dict("records"):
        frame = pq.read_table(Path(row["path"]), columns=columns).to_pandas()
        if len(frame) > max_per_year:
            frame = frame.sample(n=max_per_year, random_state=seed + int(row["year"]))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_exact_records(
    manifest: pd.DataFrame,
    split: str,
    columns: list[str],
    id_source: Path,
) -> pd.DataFrame:
    wanted = pd.read_parquet(id_source, columns=["record_id"])
    wanted_ids = set(wanted["record_id"].astype("int64").tolist())
    path = Path(manifest.loc[manifest["split"] == split, "path"].iloc[0])
    frame = pq.read_table(path, columns=["record_id"] + columns).to_pandas()
    frame = frame[frame["record_id"].isin(wanted_ids)].copy()
    order = pd.DataFrame({"record_id": list(wanted["record_id"].astype("int64"))})
    frame = order.merge(frame, on="record_id", how="left")
    if frame[columns].isna().all(axis=1).any():
        missing = int(frame[columns].isna().all(axis=1).sum())
        raise RuntimeError(f"missing exact records after merge: {missing}")
    return frame.drop(columns=["record_id"])


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


def ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    order = np.argsort(prob)
    bins = np.array_split(order, n_bins)
    total = 0.0
    for idx in bins:
        total += len(idx) / len(y_true) * abs(float(np.mean(y_true[idx])) - float(np.mean(prob[idx])))
    return total


def top_risk_rows(y_true: np.ndarray, prob: np.ndarray) -> list[dict[str, float]]:
    prevalence = float(np.mean(y_true))
    total_events = int(np.sum(y_true))
    order = np.argsort(-prob)
    rows = []
    for fraction in [0.01, 0.05, 0.10]:
        k = max(1, int(round(len(y_true) * fraction)))
        idx = order[:k]
        event_rate = float(np.mean(y_true[idx]))
        events = int(np.sum(y_true[idx]))
        rows.append(
            {
                "top_fraction": fraction,
                "top_n": k,
                "event_rate": event_rate,
                "enrichment_over_prevalence": event_rate / prevalence if prevalence else np.nan,
                "events_captured": events,
                "event_capture_pct": 100 * events / total_events if total_events else np.nan,
            }
        )
    return rows


def metric_snapshot(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    prevalence = float(np.mean(y_true))
    auprc = float(average_precision_score(y_true, prob))
    top1 = top_risk_rows(y_true, prob)[0]
    return {
        "auroc": float(roc_auc_score(y_true, prob)),
        "auprc": auprc,
        "auprc_over_prevalence": auprc / prevalence if prevalence else np.nan,
        "brier": float(brier_score_loss(y_true, prob)),
        "ece_10": ece(y_true, prob),
        "top1_event_rate": float(top1["event_rate"]),
        "top1_enrichment_over_prevalence": float(top1["enrichment_over_prevalence"]),
    }


def bootstrap_ci_rows(
    y_true: np.ndarray,
    prob: np.ndarray,
    point: dict[str, float],
    seed: int,
    n_bootstrap: int,
    base: dict[str, str],
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    # Full-year 2024 bootstrap samples can contain millions of rows and many
    # unique probabilities. Recomputing AUROC/ECE on every resample is memory
    # intensive and is not needed for the manuscript figures, which use AUPRC
    # and top-risk enrichment uncertainty.
    if n > 1_000_000:
        metrics_to_bootstrap = [
            "auprc",
            "auprc_over_prevalence",
            "top1_event_rate",
            "top1_enrichment_over_prevalence",
        ]
    else:
        metrics_to_bootstrap = list(point)
    values = {metric: [] for metric in metrics_to_bootstrap}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_boot = y_true[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        prob_boot = prob[idx]
        prevalence = float(np.mean(y_boot))
        if "auprc" in values or "auprc_over_prevalence" in values:
            auprc = float(average_precision_score(y_boot, prob_boot))
            values["auprc"].append(auprc)
            values["auprc_over_prevalence"].append(auprc / prevalence if prevalence else np.nan)
        if "top1_event_rate" in values or "top1_enrichment_over_prevalence" in values:
            top1 = top_risk_rows(y_boot, prob_boot)[0]
            values["top1_event_rate"].append(float(top1["event_rate"]))
            values["top1_enrichment_over_prevalence"].append(float(top1["enrichment_over_prevalence"]))
        remaining = [metric for metric in values if metric not in {"auprc", "auprc_over_prevalence", "top1_event_rate", "top1_enrichment_over_prevalence"}]
        if remaining:
            snapshot = metric_snapshot(y_boot, prob_boot)
            for metric in remaining:
                values[metric].append(snapshot[metric])

    rows: list[dict[str, float | int | str]] = []
    for metric, metric_values in values.items():
        arr = np.asarray(metric_values, dtype="float64")
        if len(arr) == 0:
            ci_low = np.nan
            ci_high = np.nan
        else:
            ci_low = float(np.percentile(arr, 2.5))
            ci_high = float(np.percentile(arr, 97.5))
        row = dict(base)
        row.update(
            {
                "metric": metric,
                "point": float(point[metric]),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_bootstrap_valid": int(len(arr)),
                "n_bootstrap_requested": int(n_bootstrap),
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    if args.output_tag and args.report == DEFAULT_REPORT:
        args.report = args.report.with_name(f"{args.report.stem}_{args.output_tag}{args.report.suffix}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.split_manifest)
    feature_sets = feature_sets_from_schema(Path(manifest.iloc[0]["path"]))
    all_columns = sorted(set(sum(feature_sets.values(), []) + PRIMARY_ENDPOINTS))

    print("load train")
    train = load_train(manifest, all_columns, args.seed, args.max_train_per_year)
    print(f"train rows: {len(train):,}")
    print("load matched development")
    dev = load_exact_records(manifest, "development", all_columns, args.dev_embeddings)
    print(f"development rows: {len(dev):,}")
    print("load matched test")
    test = load_exact_records(manifest, "test", all_columns, args.test_embeddings)
    print(f"test rows: {len(test):,}")

    metric_rows = []
    enrichment_rows = []
    bootstrap_rows = []
    for endpoint_idx, endpoint in enumerate(PRIMARY_ENDPOINTS):
        y_train = train[endpoint].astype("int8").to_numpy()
        y_dev = dev[endpoint].astype("int8").to_numpy()
        y_test = test[endpoint].astype("int8").to_numpy()
        for feature_idx, (feature_set, features) in enumerate(feature_sets.items()):
            print(f"fit {endpoint} {feature_set}", flush=True)
            x_train = prepare_lgbm_frame(train[features])
            x_dev = prepare_lgbm_frame(dev[features])
            x_test = prepare_lgbm_frame(test[features])
            model = make_lgbm(args.seed, args.lgbm_estimators)
            model.fit(x_train, y_train, categorical_feature="auto")
            p_dev_raw = model.predict_proba(x_dev)[:, 1]
            p_test_raw = model.predict_proba(x_test)[:, 1]
            platt = fit_platt(y_dev, p_dev_raw)
            p_test = apply_platt(platt, p_test_raw)
            prevalence = float(np.mean(y_test))
            snapshot = metric_snapshot(y_test, p_test)
            metric_rows.append(
                {
                    "endpoint": endpoint,
                    "feature_set": feature_set,
                    "model": "matched_lightgbm",
                    "probability_type": "platt",
                    "n_test": int(len(y_test)),
                    "events_test": int(np.sum(y_test)),
                    "prevalence_test": prevalence,
                    "auroc": snapshot["auroc"],
                    "auprc": snapshot["auprc"],
                    "auprc_over_prevalence": snapshot["auprc_over_prevalence"],
                    "brier": snapshot["brier"],
                    "ece_10": snapshot["ece_10"],
                }
            )
            for row in top_risk_rows(y_test, p_test):
                row.update(
                    {
                        "endpoint": endpoint,
                        "feature_set": feature_set,
                        "model": "matched_lightgbm",
                        "probability_type": "platt",
                    }
                )
                enrichment_rows.append(row)
            bootstrap_rows.extend(
                bootstrap_ci_rows(
                    y_test,
                    p_test,
                    snapshot,
                    args.seed + endpoint_idx * 1000 + feature_idx * 100,
                    args.bootstrap_iterations,
                    {
                        "endpoint": endpoint,
                        "feature_set": feature_set,
                        "model": "matched_lightgbm",
                        "probability_type": "platt",
                    },
                )
            )

    metrics = pd.DataFrame(metric_rows)
    enrichment = pd.DataFrame(enrichment_rows)
    bootstrap_ci = pd.DataFrame(bootstrap_rows)
    metrics_path = args.tables / f"matched_lightgbm_metrics{suffix}.csv"
    enrichment_path = args.tables / f"matched_lightgbm_top_risk_enrichment{suffix}.csv"
    bootstrap_ci_path = args.tables / f"matched_lightgbm_bootstrap_ci{suffix}.csv"
    metadata_path = args.tables / f"matched_lightgbm_metadata{suffix}.json"
    metrics.to_csv(metrics_path, index=False)
    enrichment.to_csv(enrichment_path, index=False)
    bootstrap_ci.to_csv(bootstrap_ci_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "dev_embeddings": str(args.dev_embeddings),
                "test_embeddings": str(args.test_embeddings),
                "train_rows": int(len(train)),
                "dev_rows": int(len(dev)),
                "test_rows": int(len(test)),
                "max_train_per_year": int(args.max_train_per_year),
                "lgbm_estimators": int(args.lgbm_estimators),
                "bootstrap_iterations": int(args.bootstrap_iterations),
                "output_tag": args.output_tag,
                "seed": int(args.seed),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "# Matched LightGBM Report",
        "",
        "This analysis evaluates supervised LightGBM on the same 2024 test record IDs used by the expanded SSL analysis. Training still uses sampled 2016-2022 records and calibration uses the matched 2023 SSL development records.",
        "",
        "## Platt-Calibrated Test Metrics",
        "",
        "| Endpoint | Feature set | n test | AUPRC | AUROC | Top 1% enrichment | ECE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    top1 = enrichment[np.isclose(enrichment["top_fraction"], 0.01)]
    for row in metrics.merge(
        top1[["endpoint", "feature_set", "enrichment_over_prevalence"]],
        on=["endpoint", "feature_set"],
        how="left",
    ).to_dict("records"):
        report_lines.append(
            "| {endpoint} | {feature_set} | {n_test:,} | {auprc:.4f} | {auroc:.4f} | {enrichment_over_prevalence:.2f} | {ece_10:.5f} |".format(
                **row
            )
        )
    report_lines.extend(
        [
            "",
            "## Output",
            "",
            f"- `{metrics_path}`",
            f"- `{enrichment_path}`",
            f"- `{bootstrap_ci_path}`",
            f"- `{metadata_path}`",
        ]
    )
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"wrote {metrics_path}")
    print(f"wrote {enrichment_path}")
    print(f"wrote {bootstrap_ci_path}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
