#!/usr/bin/env python
"""Cluster SSL embeddings into phenotypes and evaluate 2024 test transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    brier_score_loss,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEV_EMB = PROJECT_ROOT / "results" / "objects" / "masked_tabular_ssl_dev_embeddings.parquet"
DEFAULT_TEST_EMB = PROJECT_ROOT / "results" / "objects" / "masked_tabular_ssl_test_embeddings.parquet"
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "processed" / "natality_2016_2024_split_manifest.csv"
DEFAULT_OBJECT_DIR = PROJECT_ROOT / "results" / "objects"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "14_ssl_phenotype_clustering_report.md"

ENDPOINTS = [
    "outcome_maternal_morbidity_core",
    "outcome_severe_neonatal_no_nicu",
]

PROTOTYPE_FEATURES = [
    "input_MAGER",
    "input_BMI",
    "input_PREVIS",
    "input_WTGAIN",
    "input_DPLURAL",
    "input_RF_PDIAB",
    "input_RF_GDIAB",
    "input_RF_PHYPE",
    "input_RF_GHYPE",
    "input_RF_EHYPE",
    "input_RF_PPTERM",
    "input_RF_INFTR",
    "input_RF_FEDRG",
    "input_RF_ARTEC",
    "input_RF_CESAR",
    "input_CIG_REC",
    "input_IP_CHLAM",
    "input_IP_HEPB",
    "input_IP_HEPC",
    "input_NO_RISKS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-embeddings", type=Path, default=DEFAULT_DEV_EMB)
    parser.add_argument("--test-embeddings", type=Path, default=DEFAULT_TEST_EMB)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--objects", type=Path, default=DEFAULT_OBJECT_DIR)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--silhouette-sample", type=int, default=10000)
    parser.add_argument("--min-cluster-prop", type=float, default=0.05)
    parser.add_argument("--kmeans-n-init", type=int, default=20)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--stability-iterations", type=int, default=20)
    parser.add_argument("--stability-fraction", type=float, default=0.80)
    parser.add_argument("--stability-max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260525)
    return parser.parse_args()


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("ssl_emb_")]


def select_k(
    x_dev: np.ndarray,
    k_values: list[int],
    seed: int,
    silhouette_sample: int,
    min_cluster_prop: float,
    n_init: int,
) -> tuple[int, pd.DataFrame, dict[int, KMeans]]:
    rows = []
    models = {}
    rng = np.random.default_rng(seed)
    if len(x_dev) > silhouette_sample:
        silhouette_idx = rng.choice(len(x_dev), size=silhouette_sample, replace=False)
    else:
        silhouette_idx = np.arange(len(x_dev))

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
        labels = model.fit_predict(x_dev)
        models[k] = model
        counts = np.bincount(labels, minlength=k)
        min_prop = counts.min() / counts.sum()
        sil = silhouette_score(x_dev[silhouette_idx], labels[silhouette_idx])
        db = davies_bouldin_score(x_dev, labels)
        ch = calinski_harabasz_score(x_dev, labels)
        rows.append(
            {
                "k": k,
                "silhouette": sil,
                "davies_bouldin": db,
                "calinski_harabasz": ch,
                "min_cluster_prop": min_prop,
                "cluster_sizes": ";".join(str(int(item)) for item in counts),
                "passes_min_size": bool(min_prop >= min_cluster_prop),
            }
        )

    table = pd.DataFrame(rows)
    candidates = table[table["passes_min_size"]].copy()
    if candidates.empty:
        candidates = table.copy()
    best_k = int(candidates.sort_values(["silhouette", "min_cluster_prop"], ascending=False).iloc[0]["k"])
    return best_k, table, models


def assign_to_centroids(x: np.ndarray, model: KMeans) -> np.ndarray:
    distances = model.transform(x)
    return np.argmin(distances, axis=1)


def cluster_stability_rows(
    x_dev_raw: np.ndarray,
    reference_labels: np.ndarray,
    k: int,
    pca_components: int,
    seed: int,
    iterations: int,
    fraction: float,
    max_rows: int,
    n_init: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    n = len(x_dev_raw)
    sample_n = max(k * 50, int(round(n * fraction)))
    sample_n = min(sample_n, n)
    if max_rows > 0:
        sample_n = min(sample_n, max_rows)
    for iteration in range(1, iterations + 1):
        idx = rng.choice(n, size=sample_n, replace=False)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_dev_raw[idx])
        n_components = min(pca_components, x_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=seed + iteration)
        x_reduced = pca.fit_transform(x_scaled)
        model = KMeans(n_clusters=k, random_state=seed + iteration, n_init=n_init)
        labels = model.fit_predict(x_reduced)
        counts = np.bincount(labels, minlength=k)
        rows.append(
            {
                "iteration": iteration,
                "sample_n": int(sample_n),
                "sample_fraction": float(sample_n / n),
                "k": int(k),
                "ari_vs_primary": float(adjusted_rand_score(reference_labels[idx], labels)),
                "nmi_vs_primary": float(normalized_mutual_info_score(reference_labels[idx], labels)),
                "min_cluster_prop": float(counts.min() / counts.sum()),
                "cluster_sizes": ";".join(str(int(item)) for item in counts),
                "pca_explained_variance_total": float(pca.explained_variance_ratio_.sum()),
            }
        )
    return pd.DataFrame(rows)


def phenotype_rates(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for phenotype, group in frame.groupby("phenotype", sort=True):
        row = {"split": split, "phenotype": int(phenotype), "n": int(len(group))}
        for endpoint in ENDPOINTS:
            row[f"{endpoint}_positive_n"] = int(group[endpoint].sum())
            row[f"{endpoint}_rate"] = float(group[endpoint].mean())
        rows.append(row)
    return pd.DataFrame(rows)


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


def calibration_bins(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    order = np.argsort(prob)
    splits = np.array_split(order, n_bins)
    rows = []
    n = len(y_true)
    for i, idx in enumerate(splits, start=1):
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
                "enrichment_over_prevalence": event_rate / prevalence
                if prevalence > 0
                else np.nan,
                "events_captured": events,
                "event_capture_pct": 100 * events / total_events if total_events else np.nan,
            }
        )
    return rows


def risk_metric_snapshot(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    prevalence = float(np.mean(y_true))
    auprc = float(average_precision_score(y_true, prob))
    top1 = top_risk_rows(y_true, prob)[0]
    return {
        "auroc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "auprc": auprc,
        "auprc_over_prevalence": auprc / prevalence if prevalence > 0 else np.nan,
        "brier": float(brier_score_loss(y_true, prob)),
        "ece_10": ece(y_true, prob),
        "top1_event_rate": float(top1["event_rate"]),
        "top1_enrichment_over_prevalence": float(top1["enrichment_over_prevalence"]),
        "top1_event_capture_pct": float(top1["event_capture_pct"]),
    }


def percentile_rows(
    values_by_metric: dict[str, list[float]],
    point_values: dict[str, float],
    base_row: dict[str, object],
    requested: int,
) -> list[dict[str, object]]:
    rows = []
    for metric, values in values_by_metric.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            ci_low = np.nan
            ci_high = np.nan
        else:
            ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
        row = dict(base_row)
        row.update(
            {
                "metric": metric,
                "point": float(point_values.get(metric, np.nan)),
                "ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
                "ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
                "n_bootstrap_valid": int(len(arr)),
                "n_bootstrap_requested": int(requested),
            }
        )
        rows.append(row)
    return rows


def bootstrap_risk_metric_rows(
    y_true: np.ndarray,
    prob: np.ndarray,
    base_row: dict[str, object],
    seed: int,
    n_bootstrap: int,
) -> list[dict[str, object]]:
    point = risk_metric_snapshot(y_true, prob)
    values = {metric: [] for metric in point}
    rng = np.random.default_rng(seed)
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        snapshot = risk_metric_snapshot(y_true[idx], prob[idx])
        for metric, value in snapshot.items():
            values[metric].append(value)
    return percentile_rows(values, point, base_row, n_bootstrap)


def bootstrap_phenotype_rate_rows(
    test_assignments: pd.DataFrame,
    seed: int,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for phenotype, group in test_assignments.groupby("phenotype", sort=True):
        for endpoint in ENDPOINTS:
            y = group[endpoint].astype("int8").to_numpy()
            boot = []
            n = len(y)
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                boot.append(float(np.mean(y[idx])))
            ci_low, ci_high = np.percentile(np.asarray(boot), [2.5, 97.5])
            rows.append(
                {
                    "split": "test",
                    "phenotype": int(phenotype),
                    "endpoint": endpoint,
                    "n": int(n),
                    "events": int(np.sum(y)),
                    "event_rate": float(np.mean(y)),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "n_bootstrap_requested": int(n_bootstrap),
                }
            )
    return pd.DataFrame(rows)


def make_design(frame: pd.DataFrame, emb_cols: list[str], feature_set: str) -> pd.DataFrame:
    if feature_set == "ssl_embedding":
        return frame[emb_cols].copy()
    if feature_set == "phenotype":
        return pd.get_dummies(frame["phenotype"].astype("category"), prefix="phenotype")
    if feature_set == "ssl_plus_phenotype":
        phen = pd.get_dummies(frame["phenotype"].astype("category"), prefix="phenotype")
        return pd.concat([frame[emb_cols].reset_index(drop=True), phen.reset_index(drop=True)], axis=1)
    raise ValueError(feature_set)


def run_risk_models(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    emb_cols: list[str],
    seed: int,
    n_bootstrap: int,
):
    metric_rows = []
    enrichment_rows = []
    calibration_rows = []
    bootstrap_rows = []
    for endpoint_idx, endpoint in enumerate(ENDPOINTS):
        y = dev[endpoint].astype("int8")
        train_idx, cal_idx = train_test_split(
            np.arange(len(dev)),
            test_size=0.4,
            random_state=seed,
            stratify=y,
        )
        y_train = y.iloc[train_idx].to_numpy()
        y_cal = y.iloc[cal_idx].to_numpy()
        y_test = test[endpoint].astype("int8").to_numpy()

        for feature_idx, feature_set in enumerate(["ssl_embedding", "phenotype", "ssl_plus_phenotype"]):
            x_dev = make_design(dev, emb_cols, feature_set)
            x_test = make_design(test, emb_cols, feature_set)
            x_train = x_dev.iloc[train_idx]
            x_cal = x_dev.iloc[cal_idx]

            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "logistic",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            solver="lbfgs",
                        ),
                    ),
                ]
            )
            model.fit(x_train, y_train)
            p_cal_raw = model.predict_proba(x_cal)[:, 1]
            p_test_raw = model.predict_proba(x_test)[:, 1]
            platt = fit_platt(y_cal, p_cal_raw)
            p_test_platt = apply_platt(platt, p_test_raw)

            for probability_idx, (probability_type, prob) in enumerate([("raw", p_test_raw), ("platt", p_test_platt)]):
                prevalence = float(np.mean(y_test))
                auprc = float(average_precision_score(y_test, prob))
                snapshot = risk_metric_snapshot(y_test, prob)
                metric_rows.append(
                    {
                        "endpoint": endpoint,
                        "feature_set": feature_set,
                        "model": "logistic",
                        "probability_type": probability_type,
                        "n_test": int(len(y_test)),
                        "events_test": int(np.sum(y_test)),
                        "prevalence_test": prevalence,
                        "auroc": snapshot["auroc"],
                        "auprc": auprc,
                        "auprc_over_prevalence": auprc / prevalence if prevalence > 0 else np.nan,
                        "brier": snapshot["brier"],
                        "ece_10": snapshot["ece_10"],
                    }
                )
                if probability_type == "platt" and n_bootstrap > 0:
                    bootstrap_seed = seed + endpoint_idx * 1000 + feature_idx * 100 + probability_idx
                    bootstrap_rows.extend(
                        bootstrap_risk_metric_rows(
                            y_test,
                            prob,
                            {
                                "endpoint": endpoint,
                                "feature_set": feature_set,
                                "model": "logistic",
                                "probability_type": probability_type,
                            },
                            bootstrap_seed,
                            n_bootstrap,
                        )
                    )
                bins = calibration_bins(y_test, prob)
                bins.insert(0, "probability_type", probability_type)
                bins.insert(0, "feature_set", feature_set)
                bins.insert(0, "endpoint", endpoint)
                calibration_rows.extend(bins.to_dict("records"))
                for row in top_risk_rows(y_test, prob):
                    row.update(
                        {
                            "endpoint": endpoint,
                            "feature_set": feature_set,
                            "model": "logistic",
                            "probability_type": probability_type,
                        }
                    )
                    enrichment_rows.append(row)
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(enrichment_rows),
        pd.DataFrame(calibration_rows),
        pd.DataFrame(bootstrap_rows),
    )


def load_feature_prototypes(split_manifest: Path, dev_assignments: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(split_manifest)
    dev_path = Path(manifest.loc[manifest["split"] == "development", "path"].iloc[0])
    cols = ["record_id"] + PROTOTYPE_FEATURES
    raw = pq.read_table(dev_path, columns=cols).to_pandas()
    merged = dev_assignments[["record_id", "phenotype"]].merge(raw, on="record_id", how="left")
    rows = []
    for phenotype, group in merged.groupby("phenotype", sort=True):
        row = {"phenotype": int(phenotype), "n": int(len(group))}
        for feature in PROTOTYPE_FEATURES:
            values = group[feature]
            if pd.api.types.is_numeric_dtype(values):
                row[f"{feature}_mean"] = float(pd.to_numeric(values, errors="coerce").mean())
            else:
                as_text = values.astype("string")
                if set(as_text.dropna().unique()).issubset({"Y", "N", "U", "X"}):
                    row[f"{feature}_Y_rate"] = float((as_text == "Y").mean())
                    row[f"{feature}_U_or_missing_rate"] = float((as_text.isna() | (as_text == "U")).mean())
                else:
                    row[f"{feature}_mode"] = str(as_text.mode(dropna=True).iloc[0]) if not as_text.mode(dropna=True).empty else ""
        rows.append(row)
    return pd.DataFrame(rows)


def plot_cluster_selection(selection: pd.DataFrame, output: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(6.2, 4.0))
    ax1.plot(selection["k"], selection["silhouette"], marker="o", label="Silhouette")
    ax1.set_xlabel("Number of clusters")
    ax1.set_ylabel("Silhouette")
    ax2 = ax1.twinx()
    ax2.plot(selection["k"], selection["min_cluster_prop"], marker="s", color="#D95F02", label="Min cluster proportion")
    ax2.set_ylabel("Min cluster proportion")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="best")
    ax1.set_title("SSL Phenotype Cluster Selection on 2023")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_outcome_rates(rates: pd.DataFrame, output: Path) -> None:
    plot_data = rates[rates["split"] == "test"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharex=True)
    for ax, endpoint, title in [
        (axes[0], "outcome_maternal_morbidity_core_rate", "Maternal core morbidity"),
        (axes[1], "outcome_severe_neonatal_no_nicu_rate", "Severe neonatal"),
    ]:
        ax.bar(plot_data["phenotype"].astype(str), plot_data[endpoint], color="#3A6EA5")
        ax.set_title(title)
        ax.set_xlabel("Phenotype")
        ax.set_ylabel("2024 event rate")
    fig.suptitle("2024 Outcome Rates by SSL Phenotype", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_outcome_rate_ci(rate_ci: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharex=True)
    panels = [
        (axes[0], "outcome_maternal_morbidity_core", "Maternal core morbidity"),
        (axes[1], "outcome_severe_neonatal_no_nicu", "Severe neonatal"),
    ]
    for ax, endpoint, title in panels:
        plot_data = rate_ci[rate_ci["endpoint"] == endpoint].sort_values("phenotype")
        x = np.arange(len(plot_data))
        y = plot_data["event_rate"].to_numpy() * 100
        yerr = np.vstack(
            [
                y - plot_data["ci_low"].to_numpy() * 100,
                plot_data["ci_high"].to_numpy() * 100 - y,
            ]
        )
        ax.errorbar(x, y, yerr=yerr, fmt="o", color="#3A6EA5", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_data["phenotype"].astype(str))
        ax.set_title(title)
        ax.set_xlabel("Phenotype")
        ax.set_ylabel("2024 event rate, %")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_auprc_ci(metric_ci: pd.DataFrame, output: Path) -> None:
    plot_data = metric_ci[
        (metric_ci["metric"] == "auprc") & (metric_ci["probability_type"] == "platt")
    ].copy()
    endpoint_titles = {
        "outcome_maternal_morbidity_core": "Maternal core morbidity",
        "outcome_severe_neonatal_no_nicu": "Severe neonatal",
    }
    feature_order = ["phenotype", "ssl_embedding", "ssl_plus_phenotype"]
    feature_labels = {
        "phenotype": "Phenotype",
        "ssl_embedding": "SSL embedding",
        "ssl_plus_phenotype": "SSL + phenotype",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for ax, endpoint in zip(axes, ENDPOINTS):
        panel = plot_data[plot_data["endpoint"] == endpoint].set_index("feature_set").loc[feature_order].reset_index()
        x = np.arange(len(panel))
        y = panel["point"].to_numpy()
        yerr = np.vstack([y - panel["ci_low"].to_numpy(), panel["ci_high"].to_numpy() - y])
        ax.errorbar(x, y, yerr=yerr, fmt="o", color="#D95F02", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([feature_labels[item] for item in panel["feature_set"]], rotation=20, ha="right")
        ax.set_title(endpoint_titles[endpoint])
        ax.set_ylabel("AUPRC")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_stability(stability: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("ari_vs_primary", "ARI vs primary"),
        ("nmi_vs_primary", "NMI vs primary"),
        ("min_cluster_prop", "Min cluster proportion"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, (column, title) in zip(axes, metrics):
        ax.boxplot(stability[column], widths=0.45, showfliers=True)
        ax.scatter(
            np.ones(len(stability)),
            stability[column],
            s=16,
            alpha=0.55,
            color="#3A6EA5",
        )
        ax.set_xticks([1])
        ax.set_xticklabels(["resamples"])
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pca_scatter(dev_assignments: pd.DataFrame, output: Path, seed: int) -> None:
    sample = dev_assignments.sample(n=min(10000, len(dev_assignments)), random_state=seed)
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    scatter = ax.scatter(
        sample["pc1"],
        sample["pc2"],
        c=sample["phenotype"],
        cmap="tab10",
        s=8,
        alpha=0.65,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("2023 SSL Embedding Phenotypes")
    legend = ax.legend(*scatter.legend_elements(), title="Phenotype", frameon=False, loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.objects.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    if args.output_tag and args.report == DEFAULT_REPORT:
        args.report = args.report.with_name(f"{args.report.stem}{suffix}{args.report.suffix}")

    dev = pd.read_parquet(args.dev_embeddings)
    test = pd.read_parquet(args.test_embeddings)
    emb_cols = embedding_columns(dev)
    x_dev_raw = dev[emb_cols].to_numpy(dtype=np.float32)
    x_test_raw = test[emb_cols].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_dev_scaled = scaler.fit_transform(x_dev_raw)
    x_test_scaled = scaler.transform(x_test_raw)

    n_components = min(args.pca_components, x_dev_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=args.seed)
    x_dev = pca.fit_transform(x_dev_scaled)
    x_test = pca.transform(x_test_scaled)

    best_k, selection, models = select_k(
        x_dev,
        list(range(args.k_min, args.k_max + 1)),
        args.seed,
        args.silhouette_sample,
        args.min_cluster_prop,
        args.kmeans_n_init,
    )
    model = models[best_k]
    dev_labels = model.labels_
    test_labels = assign_to_centroids(x_test, model)
    stability = cluster_stability_rows(
        x_dev_raw,
        dev_labels,
        best_k,
        n_components,
        args.seed + 300,
        args.stability_iterations,
        args.stability_fraction,
        args.stability_max_rows,
        max(1, min(args.kmeans_n_init, 10)),
    )

    dev_assign = dev[["source_year", "record_id"] + ENDPOINTS].copy()
    test_assign = test[["source_year", "record_id"] + ENDPOINTS].copy()
    dev_assign["phenotype"] = dev_labels
    test_assign["phenotype"] = test_labels
    dev_assign["pc1"] = x_dev[:, 0]
    dev_assign["pc2"] = x_dev[:, 1] if x_dev.shape[1] > 1 else 0.0
    test_assign["pc1"] = x_test[:, 0]
    test_assign["pc2"] = x_test[:, 1] if x_test.shape[1] > 1 else 0.0

    dev_model_frame = pd.concat([dev_assign.reset_index(drop=True), dev[emb_cols].reset_index(drop=True)], axis=1)
    test_model_frame = pd.concat([test_assign.reset_index(drop=True), test[emb_cols].reset_index(drop=True)], axis=1)

    rates = pd.concat(
        [
            phenotype_rates(dev_assign, "development"),
            phenotype_rates(test_assign, "test"),
        ],
        ignore_index=True,
    )
    metrics, enrichment, calibration, metric_ci = run_risk_models(
        dev_model_frame,
        test_model_frame,
        emb_cols,
        args.seed,
        args.bootstrap_iterations,
    )
    rate_ci = bootstrap_phenotype_rate_rows(test_assign, args.seed + 77, args.bootstrap_iterations)
    prototypes = load_feature_prototypes(args.split_manifest, dev_assign)

    selection_path = args.tables / f"ssl_phenotype_cluster_selection{suffix}.csv"
    rates_path = args.tables / f"ssl_phenotype_outcome_rates{suffix}.csv"
    stability_path = args.tables / f"ssl_phenotype_stability{suffix}.csv"
    metrics_path = args.tables / f"ssl_phenotype_risk_metrics{suffix}.csv"
    metric_ci_path = args.tables / f"ssl_phenotype_risk_metric_bootstrap_ci{suffix}.csv"
    enrichment_path = args.tables / f"ssl_phenotype_top_risk_enrichment{suffix}.csv"
    calibration_path = args.tables / f"ssl_phenotype_calibration_bins{suffix}.csv"
    rate_ci_path = args.tables / f"ssl_phenotype_outcome_rate_bootstrap_ci{suffix}.csv"
    prototypes_path = args.tables / f"ssl_phenotype_prototypes_2023{suffix}.csv"
    dev_assignment_path = args.objects / f"ssl_phenotype_dev_assignments{suffix}.parquet"
    test_assignment_path = args.objects / f"ssl_phenotype_test_assignments{suffix}.parquet"
    model_path = args.objects / f"ssl_phenotype_model{suffix}.npz"

    selection.to_csv(selection_path, index=False)
    rates.to_csv(rates_path, index=False)
    stability.to_csv(stability_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    metric_ci.to_csv(metric_ci_path, index=False)
    enrichment.to_csv(enrichment_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    rate_ci.to_csv(rate_ci_path, index=False)
    prototypes.to_csv(prototypes_path, index=False)
    dev_assign.to_parquet(dev_assignment_path, index=False)
    test_assign.to_parquet(test_assignment_path, index=False)
    np.savez(
        model_path,
        best_k=best_k,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        centroids=model.cluster_centers_,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )

    cluster_fig = args.figures / f"ssl_phenotype_cluster_selection{suffix}.png"
    stability_fig = args.figures / f"ssl_phenotype_stability{suffix}.png"
    outcome_fig = args.figures / f"ssl_phenotype_2024_outcome_rates{suffix}.png"
    outcome_ci_fig = args.figures / f"ssl_phenotype_2024_outcome_rate_ci{suffix}.png"
    auprc_ci_fig = args.figures / f"ssl_phenotype_2024_auprc_ci{suffix}.png"
    scatter_fig = args.figures / f"ssl_phenotype_2023_pca_scatter{suffix}.png"
    plot_cluster_selection(selection, cluster_fig)
    plot_cluster_stability(stability, stability_fig)
    plot_outcome_rates(rates, outcome_fig)
    plot_outcome_rate_ci(rate_ci, outcome_ci_fig)
    plot_auprc_ci(metric_ci, auprc_ci_fig)
    plot_pca_scatter(dev_assign, scatter_fig, args.seed)

    platt_metrics = metrics[metrics["probability_type"] == "platt"].sort_values(
        ["endpoint", "auprc"], ascending=[True, False]
    )
    top1 = enrichment[
        (enrichment["probability_type"] == "platt")
        & (enrichment["top_fraction"] == 0.01)
    ].sort_values(["endpoint", "enrichment_over_prevalence"], ascending=[True, False])

    report_lines = [
        "# SSL Phenotype Clustering Report",
        "",
        "Phenotype discovery was fit on 2023 development embeddings only. The selected centroids were then applied to 2024 test embeddings.",
        "",
        "## Cluster Selection",
        "",
        f"- selected k: {best_k}",
        f"- PCA components used for clustering: {n_components}",
        f"- PCA explained variance ratio total: {float(pca.explained_variance_ratio_.sum()):.4f}",
        "",
        "## 2024 Phenotype Outcome Rates",
        "",
        "| Phenotype | n | Maternal core morbidity % | Severe neonatal % |",
        "|---:|---:|---:|---:|",
    ]
    test_rates = rates[rates["split"] == "test"].sort_values("phenotype")
    for row in test_rates.to_dict("records"):
        report_lines.append(
            "| {phenotype} | {n:,} | {maternal:.3f} | {neonatal:.3f} |".format(
                phenotype=row["phenotype"],
                n=row["n"],
                maternal=100 * row["outcome_maternal_morbidity_core_rate"],
                neonatal=100 * row["outcome_severe_neonatal_no_nicu_rate"],
            )
        )

    report_lines.extend(
        [
            "",
            "## Phenotype Stability on 2023 Development Resamples",
            "",
            f"- resampling scheme: {args.stability_iterations} iterations, {args.stability_fraction:.0%} of 2023 development records per iteration",
            f"- stability sample cap: {args.stability_max_rows if args.stability_max_rows > 0 else 'none'}",
            f"- median ARI vs primary labels: {stability['ari_vs_primary'].median():.3f}",
            f"- median NMI vs primary labels: {stability['nmi_vs_primary'].median():.3f}",
            f"- median minimum cluster proportion: {stability['min_cluster_prop'].median():.3f}",
        ]
    )

    report_lines.extend(
        [
            "",
            "## Bootstrap 95% CI for 2024 Phenotype Outcome Rates",
            "",
            "| Endpoint | Phenotype | Event rate % | 95% CI % |",
            "|---|---:|---:|---:|",
        ]
    )
    endpoint_labels = {
        "outcome_maternal_morbidity_core": "Maternal core morbidity",
        "outcome_severe_neonatal_no_nicu": "Severe neonatal",
    }
    for row in rate_ci.sort_values(["endpoint", "phenotype"]).to_dict("records"):
        report_lines.append(
            "| {endpoint} | {phenotype} | {point:.3f} | {low:.3f}-{high:.3f} |".format(
                endpoint=endpoint_labels[row["endpoint"]],
                phenotype=row["phenotype"],
                point=100 * row["event_rate"],
                low=100 * row["ci_low"],
                high=100 * row["ci_high"],
            )
        )

    report_lines.extend(
        [
            "",
            "## Platt-Calibrated 2024 Risk Metrics",
            "",
            "| Endpoint | Feature set | Events/test | Prev. | AUROC | AUPRC | AUPRC/Prev. | Brier | ECE |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in platt_metrics.to_dict("records"):
        report_lines.append(
            "| {endpoint} | {feature_set} | {events_test:,} | {prevalence_test:.5f} | {auroc:.4f} | {auprc:.4f} | {auprc_over_prevalence:.2f} | {brier:.5f} | {ece_10:.5f} |".format(
                **row
            )
        )

    report_lines.extend(
        [
            "",
            "## Bootstrap 95% CI for Platt-Calibrated AUPRC",
            "",
            "| Endpoint | Feature set | AUPRC | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    auprc_ci = metric_ci[
        (metric_ci["metric"] == "auprc") & (metric_ci["probability_type"] == "platt")
    ].sort_values(["endpoint", "point"], ascending=[True, False])
    for row in auprc_ci.to_dict("records"):
        report_lines.append(
            "| {endpoint} | {feature_set} | {point:.4f} | {ci_low:.4f}-{ci_high:.4f} |".format(
                **row
            )
        )

    report_lines.extend(
        [
            "",
            "## Top 1% Enrichment",
            "",
            "| Endpoint | Feature set | Event rate | Enrichment over prevalence | Event capture % |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in top1.to_dict("records"):
        report_lines.append(
            "| {endpoint} | {feature_set} | {event_rate:.4f} | {enrichment_over_prevalence:.2f} | {event_capture_pct:.2f} |".format(
                **row
            )
        )

    report_lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{selection_path}`",
            f"- `{rates_path}`",
            f"- `{stability_path}`",
            f"- `{metrics_path}`",
            f"- `{metric_ci_path}`",
            f"- `{enrichment_path}`",
            f"- `{rate_ci_path}`",
            f"- `{prototypes_path}`",
            f"- `{dev_assignment_path}`",
            f"- `{test_assignment_path}`",
            f"- `{cluster_fig}`",
            f"- `{stability_fig}`",
            f"- `{outcome_fig}`",
            f"- `{outcome_ci_fig}`",
            f"- `{auprc_ci_fig}`",
            f"- `{scatter_fig}`",
            "",
            "## Boundary",
            "",
        ]
    )
    if len(test) >= 3_000_000:
        report_lines.append(
            f"The current embeddings cover {len(dev):,} development records and the full {len(test):,}-record 2024 temporal test year. The 2024 labels were used only for final evaluation, not for cluster selection."
        )
    else:
        report_lines.append(
            f"This is still a sampled SSL prototype because the current embeddings cover {len(dev):,} development records and {len(test):,} test records. The 2024 labels were used only for final evaluation, not for cluster selection."
        )
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (args.tables / f"ssl_phenotype_run_metadata{suffix}.json").write_text(
        json.dumps(
            {
                "dev_embeddings": str(args.dev_embeddings),
                "test_embeddings": str(args.test_embeddings),
                "output_tag": args.output_tag,
                "best_k": best_k,
                "embedding_columns": len(emb_cols),
                "dev_rows": int(len(dev)),
                "test_rows": int(len(test)),
                "pca_components": n_components,
                "pca_explained_variance_total": float(pca.explained_variance_ratio_.sum()),
                "bootstrap_iterations": int(args.bootstrap_iterations),
                "stability_iterations": int(args.stability_iterations),
                "stability_fraction": float(args.stability_fraction),
                "stability_max_rows": int(args.stability_max_rows),
                "kmeans_n_init": int(args.kmeans_n_init),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"selected k={best_k}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
