#!/usr/bin/env python
"""Build manuscript-ready summary tables and figures across baseline and SSL models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
DEFAULT_DOCS_DIR = PROJECT_ROOT / "docs"

ENDPOINT_LABELS = {
    "outcome_maternal_morbidity_core": "Maternal core morbidity",
    "outcome_severe_neonatal_no_nicu": "Severe neonatal outcome",
}

MODEL_LABELS = {
    "temporal_lightgbm_comorbidity_only": "LightGBM comorbidity",
    "temporal_lightgbm_all_inputs": "LightGBM all inputs",
    "ssl_embedding": "SSL embedding",
    "ssl_plus_phenotype": "SSL + phenotype",
    "phenotype": "Phenotype only",
}

MODEL_ORDER = [
    "temporal_lightgbm_comorbidity_only",
    "temporal_lightgbm_all_inputs",
    "ssl_embedding",
    "ssl_plus_phenotype",
    "phenotype",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--ssl-tag", default="200k")
    return parser.parse_args()


def load_baseline(tables: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(tables / "temporal_baseline_2016_2024_metrics.csv")
    enrichment = pd.read_csv(tables / "temporal_baseline_2016_2024_top_risk_enrichment.csv")
    metrics = metrics[metrics["probability_type"] == "platt"].copy()
    metrics["model_key"] = "temporal_lightgbm_" + metrics["feature_set"].astype(str)
    metrics["analysis_scope"] = "2016-2022 train; 2023 calibration; 2024 1,000,000-record test sample"
    enrichment = enrichment[
        (enrichment["probability_type"] == "platt")
        & (np.isclose(enrichment["top_fraction"], 0.01))
    ].copy()
    enrichment["model_key"] = "temporal_lightgbm_" + enrichment["feature_set"].astype(str)
    return metrics, enrichment


def load_ssl(tables: Path, tag: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    suffix = f"_{tag}" if tag else ""
    metrics = pd.read_csv(tables / f"ssl_phenotype_risk_metrics{suffix}.csv")
    enrichment = pd.read_csv(tables / f"ssl_phenotype_top_risk_enrichment{suffix}.csv")
    ci = pd.read_csv(tables / f"ssl_phenotype_risk_metric_bootstrap_ci{suffix}.csv")
    metrics = metrics[metrics["probability_type"] == "platt"].copy()
    metrics["model_key"] = metrics["feature_set"].astype(str)
    metrics["analysis_scope"] = f"2016-2022 SSL pretraining; 2023 development; 2024 {metrics['n_test'].iloc[0]:,}-record SSL test sample"
    enrichment = enrichment[
        (enrichment["probability_type"] == "platt")
        & (np.isclose(enrichment["top_fraction"], 0.01))
    ].copy()
    enrichment["model_key"] = enrichment["feature_set"].astype(str)
    ci = ci[ci["probability_type"] == "platt"].copy()
    ci["model_key"] = ci["feature_set"].astype(str)
    return metrics, enrichment, ci


def build_summary_table(args: argparse.Namespace) -> pd.DataFrame:
    baseline_metrics, baseline_enrichment = load_baseline(args.tables)
    ssl_metrics, ssl_enrichment, ssl_ci = load_ssl(args.tables, args.ssl_tag)

    metrics = pd.concat([baseline_metrics, ssl_metrics], ignore_index=True)
    enrichment = pd.concat([baseline_enrichment, ssl_enrichment], ignore_index=True)
    top = enrichment[
        [
            "endpoint",
            "model_key",
            "event_rate",
            "enrichment_over_prevalence",
            "events_captured",
            "event_capture_pct",
        ]
    ].rename(
        columns={
            "event_rate": "top1_event_rate",
            "enrichment_over_prevalence": "top1_enrichment_over_prevalence",
            "events_captured": "top1_events_captured",
            "event_capture_pct": "top1_event_capture_pct",
        }
    )
    summary = metrics.merge(top, on=["endpoint", "model_key"], how="left")
    summary["endpoint_label"] = summary["endpoint"].map(ENDPOINT_LABELS)
    summary["model_label"] = summary["model_key"].map(MODEL_LABELS)
    summary["model_order"] = summary["model_key"].map({key: i for i, key in enumerate(MODEL_ORDER)})

    ci_wide = (
        ssl_ci[ssl_ci["metric"].isin(["auprc", "auroc", "top1_enrichment_over_prevalence"])]
        .pivot_table(
            index=["endpoint", "model_key"],
            columns="metric",
            values=["ci_low", "ci_high"],
            aggfunc="first",
        )
    )
    ci_wide.columns = [f"{metric}_{bound}" for bound, metric in ci_wide.columns]
    ci_wide = ci_wide.reset_index()
    summary = summary.merge(ci_wide, on=["endpoint", "model_key"], how="left")
    summary = summary.sort_values(["endpoint", "model_order"]).reset_index(drop=True)

    columns = [
        "endpoint",
        "endpoint_label",
        "model_key",
        "model_label",
        "analysis_scope",
        "n_test",
        "events_test",
        "prevalence_test",
        "auroc",
        "auroc_ci_low",
        "auroc_ci_high",
        "auprc",
        "auprc_ci_low",
        "auprc_ci_high",
        "auprc_over_prevalence",
        "brier",
        "ece_10",
        "top1_event_rate",
        "top1_enrichment_over_prevalence",
        "top1_enrichment_over_prevalence_ci_low",
        "top1_enrichment_over_prevalence_ci_high",
        "top1_events_captured",
        "top1_event_capture_pct",
    ]
    return summary[columns]


def plot_model_summary(summary: pd.DataFrame, output: Path) -> None:
    colors = {
        "temporal_lightgbm_comorbidity_only": "#4C78A8",
        "temporal_lightgbm_all_inputs": "#1F4E79",
        "ssl_embedding": "#F58518",
        "ssl_plus_phenotype": "#D95F02",
        "phenotype": "#767676",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    panels = [
        (axes[0, 0], "outcome_maternal_morbidity_core", "auprc", "AUPRC"),
        (axes[0, 1], "outcome_severe_neonatal_no_nicu", "auprc", "AUPRC"),
        (axes[1, 0], "outcome_maternal_morbidity_core", "top1_enrichment_over_prevalence", "Top 1% enrichment"),
        (axes[1, 1], "outcome_severe_neonatal_no_nicu", "top1_enrichment_over_prevalence", "Top 1% enrichment"),
    ]
    for ax, endpoint, metric, ylabel in panels:
        panel = summary[summary["endpoint"] == endpoint].copy()
        x = np.arange(len(panel))
        ax.bar(
            x,
            panel[metric],
            color=[colors[key] for key in panel["model_key"]],
            width=0.72,
        )
        if metric == "auprc":
            low_col = "auprc_ci_low"
            high_col = "auprc_ci_high"
        else:
            low_col = "top1_enrichment_over_prevalence_ci_low"
            high_col = "top1_enrichment_over_prevalence_ci_high"
        has_ci = panel[low_col].notna() & panel[high_col].notna()
        if has_ci.any():
            ci_panel = panel[has_ci]
            ci_x = x[has_ci.to_numpy()]
            y = ci_panel[metric].to_numpy()
            yerr = np.vstack(
                [
                    y - ci_panel[low_col].to_numpy(),
                    ci_panel[high_col].to_numpy() - y,
                ]
            )
            ax.errorbar(ci_x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.0, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(panel["model_label"], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ENDPOINT_LABELS[endpoint])
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, output: Path, figure_path: Path) -> None:
    lines = [
        "# Manuscript-Ready Model Summary",
        "",
        "This summary combines the sampled temporal LightGBM baseline with the expanded SSL phenotype analysis. The rows should be interpreted as model-evidence summaries rather than a perfectly matched head-to-head comparison, because the temporal LightGBM baseline used a 1,000,000-record 2024 test sample whereas the expanded SSL analysis used a 200,000-record 2024 test sample.",
        "",
        "## Main Interpretation",
        "",
        "- The strongest pure prediction baseline remains all-input temporal LightGBM.",
        "- SSL embeddings provide moderate risk enrichment and support stable phenotype discovery.",
        "- Adding phenotype labels to SSL embeddings yields little additional discrimination, but improves the clinical interpretability of the latent representation.",
        "- Phenotype-only models should be framed as explanatory summaries, not standalone risk predictors.",
        "",
        "## Key 2024 Test Metrics",
        "",
        "| Endpoint | Model | n test | AUPRC | AUROC | Top 1% enrichment | ECE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            "| {endpoint_label} | {model_label} | {n_test:,} | {auprc:.4f} | {auroc:.4f} | {top1:.2f} | {ece:.5f} |".format(
                endpoint_label=row["endpoint_label"],
                model_label=row["model_label"],
                n_test=int(row["n_test"]),
                auprc=row["auprc"],
                auroc=row["auroc"],
                top1=row["top1_enrichment_over_prevalence"],
                ece=row["ece_10"],
            )
        )
    lines.extend(
        [
            "",
            "## Output",
            "",
            f"- Summary table: `{DEFAULT_TABLE_DIR / 'manuscript_model_summary.csv'}`",
            f"- Figure: `{figure_path}`",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    args.docs.mkdir(parents=True, exist_ok=True)
    summary = build_summary_table(args)
    table_path = args.tables / "manuscript_model_summary.csv"
    figure_path = args.figures / "manuscript_model_summary.png"
    report_path = args.docs / "15_manuscript_model_summary.md"
    summary.to_csv(table_path, index=False)
    plot_model_summary(summary, figure_path)
    write_report(summary, report_path, figure_path)
    print(f"wrote {table_path}")
    print(f"wrote {figure_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
