#!/usr/bin/env python
"""Evaluate whether SSL embeddings add value beyond supervised Natality inputs."""

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
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "31_incremental_ssl_lightgbm_report.md"

PRIMARY_ENDPOINTS = [
    "outcome_maternal_morbidity_core",
    "outcome_severe_neonatal_no_nicu",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--train-embeddings", type=Path, required=True)
    parser.add_argument("--dev-embeddings", type=Path, required=True)
    parser.add_argument("--test-embeddings", type=Path, required=True)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-tag", default="full2016_2022")
    parser.add_argument("--lgbm-estimators", type=int, default=300)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260525)
    return parser.parse_args()


def input_columns_from_schema(path: Path) -> list[str]:
    names = pq.ParquetFile(path).schema.names
    return [
        column
        for column in names
        if column.startswith("input_") or column.startswith("missing_input_")
    ]


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("ssl_emb_")]


def load_train_exact(
    manifest: pd.DataFrame,
    train_embeddings: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    frames = []
    order = train_embeddings[["source_year", "record_id"]].copy()
    order["source_year"] = order["source_year"].astype("int32")
    order["record_id"] = order["record_id"].astype("int64")
    for year, group in order.groupby("source_year", sort=True):
        path = Path(manifest.loc[manifest["year"].eq(int(year)), "path"].iloc[0])
        wanted_ids = set(group["record_id"].tolist())
        frame = pq.read_table(path, columns=["record_id"] + columns).to_pandas()
        frame["record_id"] = frame["record_id"].astype("int64")
        frame = frame[frame["record_id"].isin(wanted_ids)].copy()
        frame.insert(0, "source_year", int(year))
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    merged = order.merge(raw, on=["source_year", "record_id"], how="left")
    if merged[columns].isna().all(axis=1).any():
        missing = int(merged[columns].isna().all(axis=1).sum())
        raise RuntimeError(f"missing train records after merge: {missing}")
    return merged


def load_single_year_exact(
    manifest: pd.DataFrame,
    split: str,
    embeddings: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    order = embeddings[["record_id"]].copy()
    order["record_id"] = order["record_id"].astype("int64")
    path = Path(manifest.loc[manifest["split"].eq(split), "path"].iloc[0])
    wanted_ids = set(order["record_id"].tolist())
    frame = pq.read_table(path, columns=["record_id"] + columns).to_pandas()
    frame["record_id"] = frame["record_id"].astype("int64")
    frame = frame[frame["record_id"].isin(wanted_ids)].copy()
    merged = order.merge(frame, on="record_id", how="left")
    if merged[columns].isna().all(axis=1).any():
        missing = int(merged[columns].isna().all(axis=1).sum())
        raise RuntimeError(f"missing {split} records after merge: {missing}")
    return merged


def prepare_lgbm_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_bool_dtype(out[column]):
            out[column] = out[column].astype("int8")
        elif pd.api.types.is_object_dtype(out[column]) or isinstance(out[column].dtype, pd.StringDtype):
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
    model = LogisticRegression(max_iter=1000)
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
        if len(idx) == 0:
            continue
        total += len(idx) / len(y_true) * abs(float(np.mean(y_true[idx])) - float(np.mean(prob[idx])))
    return float(total)


def top_risk_rows(y_true: np.ndarray, prob: np.ndarray) -> list[dict[str, float]]:
    prevalence = float(np.mean(y_true))
    total_events = int(np.sum(y_true))
    order = np.argsort(-prob)
    rows = []
    for fraction in [0.005, 0.01, 0.02, 0.05, 0.10]:
        k = max(1, int(round(len(y_true) * fraction)))
        idx = order[:k]
        events = int(np.sum(y_true[idx]))
        event_rate = float(events / k)
        rows.append(
            {
                "top_fraction": fraction,
                "top_n": int(k),
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
    top1 = [row for row in top_risk_rows(y_true, prob) if np.isclose(row["top_fraction"], 0.01)][0]
    return {
        "auroc": float(roc_auc_score(y_true, prob)),
        "auprc": auprc,
        "auprc_over_prevalence": auprc / prevalence if prevalence else np.nan,
        "brier": float(brier_score_loss(y_true, prob)),
        "ece_10": ece(y_true, prob),
        "top1_event_rate": float(top1["event_rate"]),
        "top1_enrichment_over_prevalence": float(top1["enrichment_over_prevalence"]),
    }


def bootstrap_rows(
    y_true: np.ndarray,
    prob: np.ndarray,
    point: dict[str, float],
    base: dict[str, str],
    seed: int,
    iterations: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in point}
    n = len(y_true)
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        y_boot = y_true[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        snap = metric_snapshot(y_boot, prob[idx])
        for metric, value in snap.items():
            values[metric].append(value)
    rows = []
    for metric, vals in values.items():
        arr = np.asarray(vals, dtype="float64")
        row = dict(base)
        row.update(
            {
                "metric": metric,
                "point": float(point[metric]),
                "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
                "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
                "n_bootstrap_valid": int(len(arr)),
                "n_bootstrap_requested": int(iterations),
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    if args.output_tag and args.report == DEFAULT_REPORT:
        args.report = args.report.with_name(f"{args.report.stem}_{args.output_tag}{args.report.suffix}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_tag}" if args.output_tag else ""

    manifest = pd.read_csv(args.split_manifest)
    input_cols = input_columns_from_schema(Path(manifest.iloc[0]["path"]))
    endpoint_cols = PRIMARY_ENDPOINTS
    raw_cols = input_cols + endpoint_cols

    print("load embeddings", flush=True)
    train_emb = pd.read_parquet(args.train_embeddings)
    dev_emb = pd.read_parquet(args.dev_embeddings)
    test_emb = pd.read_parquet(args.test_embeddings)
    emb_cols = embedding_columns(train_emb)

    print("load exact input records", flush=True)
    train_raw = load_train_exact(manifest, train_emb, raw_cols)
    dev_raw = load_single_year_exact(manifest, "development", dev_emb, raw_cols)
    test_raw = load_single_year_exact(manifest, "test", test_emb, raw_cols)
    train = pd.concat([train_raw.reset_index(drop=True), train_emb[emb_cols].reset_index(drop=True)], axis=1)
    dev = pd.concat([dev_raw.reset_index(drop=True), dev_emb[emb_cols].reset_index(drop=True)], axis=1)
    test = pd.concat([test_raw.reset_index(drop=True), test_emb[emb_cols].reset_index(drop=True)], axis=1)

    feature_sets = {
        "all_inputs": input_cols,
        "ssl_embeddings": emb_cols,
        "all_inputs_plus_ssl": input_cols + emb_cols,
    }

    metric_rows = []
    enrichment_rows = []
    bootstrap_ci_rows = []
    for endpoint_idx, endpoint in enumerate(PRIMARY_ENDPOINTS):
        y_train = train[endpoint].astype("int8").to_numpy()
        y_dev = dev[endpoint].astype("int8").to_numpy()
        y_test = test[endpoint].astype("int8").to_numpy()
        for feature_idx, (feature_set, features) in enumerate(feature_sets.items()):
            print(f"fit {endpoint} {feature_set}", flush=True)
            model = make_lgbm(args.seed + endpoint_idx * 10 + feature_idx, args.lgbm_estimators)
            model.fit(
                prepare_lgbm_frame(train[features]),
                y_train,
                categorical_feature="auto",
            )
            p_dev_raw = model.predict_proba(prepare_lgbm_frame(dev[features]))[:, 1]
            p_test_raw = model.predict_proba(prepare_lgbm_frame(test[features]))[:, 1]
            platt = fit_platt(y_dev, p_dev_raw)
            p_test = apply_platt(platt, p_test_raw)
            prevalence = float(np.mean(y_test))
            snapshot = metric_snapshot(y_test, p_test)
            metric_rows.append(
                {
                    "endpoint": endpoint,
                    "feature_set": feature_set,
                    "model": "lightgbm",
                    "probability_type": "platt",
                    "n_train": int(len(train)),
                    "n_dev": int(len(dev)),
                    "n_test": int(len(test)),
                    "events_test": int(np.sum(y_test)),
                    "prevalence_test": prevalence,
                    **snapshot,
                }
            )
            for row in top_risk_rows(y_test, p_test):
                row.update(
                    {
                        "endpoint": endpoint,
                        "feature_set": feature_set,
                        "model": "lightgbm",
                        "probability_type": "platt",
                    }
                )
                enrichment_rows.append(row)
            bootstrap_ci_rows.extend(
                bootstrap_rows(
                    y_test,
                    p_test,
                    snapshot,
                    {
                        "endpoint": endpoint,
                        "feature_set": feature_set,
                        "model": "lightgbm",
                        "probability_type": "platt",
                    },
                    args.seed + endpoint_idx * 1000 + feature_idx * 100,
                    args.bootstrap_iterations,
                )
            )

    metrics = pd.DataFrame(metric_rows)
    enrichment = pd.DataFrame(enrichment_rows)
    bootstrap_ci = pd.DataFrame(bootstrap_ci_rows)
    metrics_path = args.tables / f"incremental_ssl_lightgbm_metrics{suffix}.csv"
    enrichment_path = args.tables / f"incremental_ssl_lightgbm_topk{suffix}.csv"
    bootstrap_path = args.tables / f"incremental_ssl_lightgbm_bootstrap_ci{suffix}.csv"
    metadata_path = args.tables / f"incremental_ssl_lightgbm_metadata{suffix}.json"
    metrics.to_csv(metrics_path, index=False)
    enrichment.to_csv(enrichment_path, index=False)
    bootstrap_ci.to_csv(bootstrap_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "train_embeddings": str(args.train_embeddings),
                "dev_embeddings": str(args.dev_embeddings),
                "test_embeddings": str(args.test_embeddings),
                "embedding_columns": len(emb_cols),
                "input_columns": len(input_cols),
                "train_rows": int(len(train)),
                "dev_rows": int(len(dev)),
                "test_rows": int(len(test)),
                "lgbm_estimators": int(args.lgbm_estimators),
                "bootstrap_iterations": int(args.bootstrap_iterations),
                "seed": int(args.seed),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    top1 = enrichment[np.isclose(enrichment["top_fraction"], 0.01)][
        ["endpoint", "feature_set", "enrichment_over_prevalence"]
    ]
    lines = [
        "# Incremental SSL LightGBM Report",
        "",
        "This analysis tests whether full-scale SSL embeddings add predictive value beyond leakage-controlled Natality input variables under the same 2016-2022 train, 2023 calibration, and 2024 test split.",
        "",
        "| Endpoint | Feature set | AUPRC | AUROC | AUPRC/Prev. | Top 1% enrichment | ECE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.merge(top1, on=["endpoint", "feature_set"], how="left").to_dict("records"):
        lines.append(
            "| {endpoint} | {feature_set} | {auprc:.4f} | {auroc:.4f} | {auprc_over_prevalence:.2f} | {enrichment_over_prevalence:.2f} | {ece_10:.5f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{metrics_path}`",
            f"- `{enrichment_path}`",
            f"- `{bootstrap_path}`",
            f"- `{metadata_path}`",
            "",
            "## Boundary",
            "",
            "If `all_inputs_plus_ssl` does not improve over `all_inputs`, the manuscript should not claim broad predictive superiority of SSL. The defensible claim would shift toward scalable representation learning, phenotype discovery, and risk enrichment rather than replacement of strong supervised baselines.",
        ]
    )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {metrics_path}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
