#!/usr/bin/env python
"""Build manuscript summary using matched 200k LightGBM and SSL results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

ENDPOINTS = [
    "outcome_maternal_morbidity_core",
    "outcome_severe_neonatal_no_nicu",
]

ENDPOINT_LABELS = {
    "outcome_maternal_morbidity_core": "Maternal core morbidity",
    "outcome_severe_neonatal_no_nicu": "Severe neonatal outcome",
}

MODEL_LABELS = {
    "lightgbm_comorbidity_only": "LightGBM comorbidity",
    "lightgbm_all_inputs": "LightGBM all inputs",
    "ssl_embedding": "SSL embedding",
    "ssl_plus_phenotype": "SSL + phenotype",
    "phenotype": "Phenotype only",
}

MODEL_ORDER = [
    "lightgbm_comorbidity_only",
    "lightgbm_all_inputs",
    "ssl_embedding",
    "ssl_plus_phenotype",
    "phenotype",
]

COLORS = {
    "lightgbm_comorbidity_only": "#4C78A8",
    "lightgbm_all_inputs": "#1F4E79",
    "ssl_embedding": "#F58518",
    "ssl_plus_phenotype": "#D95F02",
    "phenotype": "#767676",
}


def load_matched_lightgbm() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(TABLE_DIR / "matched_200k_lightgbm_metrics.csv")
    enrichment = pd.read_csv(TABLE_DIR / "matched_200k_lightgbm_top_risk_enrichment.csv")
    ci_path = TABLE_DIR / "matched_200k_lightgbm_bootstrap_ci.csv"
    ci = pd.read_csv(ci_path) if ci_path.exists() else pd.DataFrame()
    metrics["model_key"] = "lightgbm_" + metrics["feature_set"].astype(str)
    metrics["analysis_scope"] = "Matched 200,000-record 2024 SSL test sample"
    enrichment = enrichment[np.isclose(enrichment["top_fraction"], 0.01)].copy()
    enrichment["model_key"] = "lightgbm_" + enrichment["feature_set"].astype(str)
    if not ci.empty:
        ci = ci[ci["probability_type"] == "platt"].copy()
        ci["model_key"] = "lightgbm_" + ci["feature_set"].astype(str)
    return metrics, enrichment, ci


def load_ssl() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(TABLE_DIR / "ssl_phenotype_risk_metrics_200k.csv")
    enrichment = pd.read_csv(TABLE_DIR / "ssl_phenotype_top_risk_enrichment_200k.csv")
    ci = pd.read_csv(TABLE_DIR / "ssl_phenotype_risk_metric_bootstrap_ci_200k.csv")
    metrics = metrics[metrics["probability_type"] == "platt"].copy()
    metrics["model_key"] = metrics["feature_set"].astype(str)
    metrics["analysis_scope"] = "Matched 200,000-record 2024 SSL test sample"
    enrichment = enrichment[
        (enrichment["probability_type"] == "platt")
        & (np.isclose(enrichment["top_fraction"], 0.01))
    ].copy()
    enrichment["model_key"] = enrichment["feature_set"].astype(str)
    ci = ci[ci["probability_type"] == "platt"].copy()
    ci["model_key"] = ci["feature_set"].astype(str)
    return metrics, enrichment, ci


def build_summary() -> pd.DataFrame:
    lgbm_metrics, lgbm_top, lgbm_ci = load_matched_lightgbm()
    ssl_metrics, ssl_top, ssl_ci = load_ssl()
    metrics = pd.concat([lgbm_metrics, ssl_metrics], ignore_index=True)
    top = pd.concat([lgbm_top, ssl_top], ignore_index=True)
    top = top[
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

    all_ci = pd.concat([lgbm_ci, ssl_ci], ignore_index=True)
    ci_wide = (
        all_ci[all_ci["metric"].isin(["auprc", "auroc", "top1_enrichment_over_prevalence"])]
        .pivot_table(
            index=["endpoint", "model_key"],
            columns="metric",
            values=["ci_low", "ci_high"],
            aggfunc="first",
        )
    )
    ci_wide.columns = [f"{metric}_{bound}" for bound, metric in ci_wide.columns]
    summary = summary.merge(ci_wide.reset_index(), on=["endpoint", "model_key"], how="left")
    summary = summary.sort_values(["endpoint", "model_order"]).reset_index(drop=True)
    return summary


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
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
        ax.bar(x, panel[metric], color=[COLORS[key] for key in panel["model_key"]], width=0.72)
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
            yerr = np.vstack([y - ci_panel[low_col].to_numpy(), ci_panel[high_col].to_numpy() - y])
            ax.errorbar(ci_x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.0, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(panel["model_label"], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ENDPOINT_LABELS[endpoint])
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, output: Path, table_path: Path, figure_path: Path) -> None:
    lgbm_ci_path = TABLE_DIR / "matched_200k_lightgbm_bootstrap_ci.csv"
    ssl_ci_path = TABLE_DIR / "ssl_phenotype_risk_metric_bootstrap_ci_200k.csv"
    lines = [
        "# Matched 200k Manuscript Model Summary",
        "",
        "This summary compares supervised LightGBM and SSL-derived models on the same 200,000-record 2024 test sample used by the expanded SSL analysis. It is the preferred Figure 2 source because it removes the unequal-test-sample-size concern.",
        "",
        "## Main Interpretation",
        "",
        "- All-input LightGBM remains the strongest pure predictor in the matched 200k test set.",
        "- SSL embeddings retain clinically meaningful risk enrichment, especially for severe neonatal outcome.",
        "- Phenotype labels contribute more to interpretation than to discrimination.",
        "",
        "## Key Metrics",
        "",
        "| Endpoint | Model | AUPRC | AUROC | Top 1% enrichment | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            "| {endpoint_label} | {model_label} | {auprc:.4f} | {auroc:.4f} | {top1_enrichment_over_prevalence:.2f} | {ece_10:.5f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Output",
            "",
            f"- `{table_path}`",
            f"- `{figure_path}`",
            f"- `{lgbm_ci_path}`",
            f"- `{ssl_ci_path}`",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    table_path = TABLE_DIR / "manuscript_model_summary_matched200k.csv"
    figure_path = FIGURE_DIR / "manuscript_model_summary_matched200k.png"
    report_path = DOCS_DIR / "20_manuscript_model_summary_matched200k.md"
    summary.to_csv(table_path, index=False)
    plot_summary(summary, figure_path)
    write_report(summary, report_path, table_path, figure_path)
    print(f"wrote {table_path}")
    print(f"wrote {figure_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
