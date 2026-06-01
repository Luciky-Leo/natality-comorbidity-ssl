#!/usr/bin/env python
"""Sensitivity analysis for PCA-compressed SSL embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEV_EMB = PROJECT_ROOT / "results" / "objects" / "masked_tabular_ssl_dev_embeddings_200k.parquet"
DEFAULT_TEST_EMB = PROJECT_ROOT / "results" / "objects" / "masked_tabular_ssl_test_embeddings_200k.parquet"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "16_ssl_pca_sensitivity_report.md"

ENDPOINTS = [
    "outcome_maternal_morbidity_core",
    "outcome_severe_neonatal_no_nicu",
]

ENDPOINT_LABELS = {
    "outcome_maternal_morbidity_core": "Maternal core morbidity",
    "outcome_severe_neonatal_no_nicu": "Severe neonatal outcome",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-embeddings", type=Path, default=DEFAULT_DEV_EMB)
    parser.add_argument("--test-embeddings", type=Path, default=DEFAULT_TEST_EMB)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--components", nargs="*", type=int, default=[5, 10, 20, 32])
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument(
        "--output-tag",
        default="200k",
        help="Suffix for output files, for example full2016_2022_mask035_d48_l2_cuda.",
    )
    return parser.parse_args()


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("ssl_emb_")]


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


def ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    order = np.argsort(prob)
    bins = np.array_split(order, n_bins)
    total = 0.0
    for idx in bins:
        total += len(idx) / len(y_true) * abs(float(np.mean(y_true[idx])) - float(np.mean(prob[idx])))
    return total


def top_risk(y_true: np.ndarray, prob: np.ndarray, fraction: float = 0.01) -> dict[str, float]:
    prevalence = float(np.mean(y_true))
    order = np.argsort(-prob)
    k = max(1, int(round(len(y_true) * fraction)))
    idx = order[:k]
    event_rate = float(np.mean(y_true[idx]))
    events = int(np.sum(y_true[idx]))
    total_events = int(np.sum(y_true))
    return {
        "top_fraction": fraction,
        "top_n": k,
        "top1_event_rate": event_rate,
        "top1_enrichment_over_prevalence": event_rate / prevalence if prevalence else np.nan,
        "top1_events_captured": events,
        "top1_event_capture_pct": 100 * events / total_events if total_events else np.nan,
    }


def model_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    prevalence = float(np.mean(y_true))
    auprc = float(average_precision_score(y_true, prob))
    out = {
        "n_test": int(len(y_true)),
        "events_test": int(np.sum(y_true)),
        "prevalence_test": prevalence,
        "auroc": float(roc_auc_score(y_true, prob)),
        "auprc": auprc,
        "auprc_over_prevalence": auprc / prevalence if prevalence else np.nan,
        "brier": float(brier_score_loss(y_true, prob)),
        "ece_10": ece(y_true, prob),
    }
    out.update(top_risk(y_true, prob))
    return out


def run_models(dev_features: np.ndarray, test_features: np.ndarray, dev: pd.DataFrame, test: pd.DataFrame, label: str, seed: int) -> list[dict[str, object]]:
    rows = []
    for endpoint in ENDPOINTS:
        y_dev = dev[endpoint].astype("int8").to_numpy()
        y_test = test[endpoint].astype("int8").to_numpy()
        train_idx, cal_idx = train_test_split(
            np.arange(len(dev)),
            test_size=0.4,
            random_state=seed,
            stratify=y_dev,
        )
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
        model.fit(dev_features[train_idx], y_dev[train_idx])
        p_cal = model.predict_proba(dev_features[cal_idx])[:, 1]
        p_test_raw = model.predict_proba(test_features)[:, 1]
        platt = fit_platt(y_dev[cal_idx], p_cal)
        p_test = apply_platt(platt, p_test_raw)
        row = {
            "endpoint": endpoint,
            "endpoint_label": ENDPOINT_LABELS[endpoint],
            "feature_set": label,
        }
        row.update(model_metrics(y_test, p_test))
        rows.append(row)
    return rows


def plot_sensitivity(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=False)
    for ax, endpoint in zip(axes, ENDPOINTS):
        panel = summary[summary["endpoint"] == endpoint].copy()
        x = np.arange(len(panel))
        ax.plot(x, panel["auprc"], marker="o", color="#D95F02")
        ax.set_xticks(x)
        ax.set_xticklabels(panel["feature_set"], rotation=25, ha="right")
        ax.set_title(ENDPOINT_LABELS[endpoint])
        ax.set_ylabel("AUPRC")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, output: Path, table_path: Path, figure_path: Path) -> None:
    lines = [
        "# SSL PCA Sensitivity Report",
        "",
        "PCA was fit on 2023 development SSL embeddings only. The 2024 SSL test embeddings were transformed using the fixed 2023 scaler/PCA and evaluated once.",
        "",
        "## Key Metrics",
        "",
        "| Endpoint | Feature set | AUPRC | AUROC | Top 1% enrichment | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            "| {endpoint_label} | {feature_set} | {auprc:.4f} | {auroc:.4f} | {top1_enrichment_over_prevalence:.2f} | {ece_10:.5f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If PCA-compressed embeddings retain similar performance to the full 48-dimensional embedding, the SSL signal is less likely to be an artifact of high-dimensional overfitting.",
            "",
            "## Output",
            "",
            f"- `{table_path}`",
            f"- `{figure_path}`",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    dev = pd.read_parquet(args.dev_embeddings)
    test = pd.read_parquet(args.test_embeddings)
    emb_cols = embedding_columns(dev)
    x_dev = dev[emb_cols].to_numpy(dtype=np.float32)
    x_test = test[emb_cols].to_numpy(dtype=np.float32)

    rows = []
    scaler = StandardScaler()
    x_dev_scaled = scaler.fit_transform(x_dev)
    x_test_scaled = scaler.transform(x_test)
    for n_components in args.components:
        pca = PCA(n_components=min(n_components, x_dev_scaled.shape[1]), random_state=args.seed)
        dev_pca = pca.fit_transform(x_dev_scaled)
        test_pca = pca.transform(x_test_scaled)
        label = f"PCA {dev_pca.shape[1]}"
        for row in run_models(dev_pca, test_pca, dev, test, label, args.seed):
            row["pca_components"] = int(dev_pca.shape[1])
            row["pca_explained_variance_total"] = float(pca.explained_variance_ratio_.sum())
            rows.append(row)

    for row in run_models(x_dev, x_test, dev, test, "Full 48", args.seed):
        row["pca_components"] = np.nan
        row["pca_explained_variance_total"] = np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    feature_order = {f"PCA {item}": i for i, item in enumerate(args.components)}
    feature_order["Full 48"] = len(feature_order)
    summary["feature_order"] = summary["feature_set"].map(feature_order)
    summary = summary.sort_values(["endpoint", "feature_order"]).reset_index(drop=True)

    table_path = args.tables / f"ssl_pca_sensitivity_{args.output_tag}.csv"
    figure_path = args.figures / f"ssl_pca_sensitivity_{args.output_tag}.png"
    summary.to_csv(table_path, index=False)
    plot_sensitivity(summary, figure_path)
    write_report(summary, args.report, table_path, figure_path)
    print(f"wrote {table_path}")
    print(f"wrote {figure_path}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
