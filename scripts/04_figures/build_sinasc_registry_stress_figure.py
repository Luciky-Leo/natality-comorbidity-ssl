#!/usr/bin/env python
"""Build manuscript Figure 7 for the SINASC registry stress-test."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
SUBMISSION_FIG_DIR = PROJECT_ROOT / "submission_package" / "springer_nature_latex" / "figures"
SUBMISSION_SOURCE_DIR = PROJECT_ROOT / "submission_package" / "springer_nature_latex" / "source_data"

TAG = "sinasc_2023train500k_dev200k_2024full_ssl"

BLUE = "#4E79A7"
ORANGE = "#F28E2B"
GREEN = "#59A14F"
PURPLE = "#B07AA1"
GRAY = "#767676"
LIGHT_GRAY = "#E6E6E6"
DARK = "#202020"
GRID = "#E7E7E7"

MODEL_LABEL = {
    "sinasc_overlap_inputs_lightgbm": "Overlap inputs\nLightGBM",
    "ssl_embedding_logit": "SSL\nembedding",
    "ssl_plus_phenotype_logit": "SSL +\nphenotype",
    "phenotype_only_logit": "Phenotype\nonly",
}

MODEL_COLOR = {
    "sinasc_overlap_inputs_lightgbm": BLUE,
    "ssl_embedding_logit": ORANGE,
    "ssl_plus_phenotype_logit": GREEN,
    "phenotype_only_logit": GRAY,
}

ENDPOINT_LABEL = {
    "outcome_sinasc_severe_birth_status": "Severe birth-status",
    "outcome_sinasc_broad_birth_status": "Broad birth-status",
}

BURDEN_LABEL = {
    "outcome_sinasc_broad_birth_status": "Broad birth-status",
    "outcome_preterm_lt37": "Preterm <37 wk",
    "outcome_low_birthweight_lt2500g": "Low birthweight <2500 g",
    "outcome_sinasc_severe_birth_status": "Severe birth-status",
    "outcome_very_preterm_lt32": "Very preterm <32 wk",
    "outcome_very_low_birthweight_lt1500g": "Very low birthweight <1500 g",
    "outcome_congenital_anomaly": "Congenital anomaly",
    "outcome_low_apgar5_lt7": "5-min Apgar <7",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.4,
            "axes.titlesize": 8.7,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.72,
            "axes.edgecolor": "#222222",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def label(ax: plt.Axes, text: str, x: float = -0.10, y: float = 1.10) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top")


def clean(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.65, alpha=0.95)
    ax.set_axisbelow(True)


def pct_axis(x: float, _pos: int) -> str:
    return f"{x * 100:.0f}"


def draw_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, edge: str, face: str) -> None:
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", color=DARK, linespacing=1.25)


def panel_endpoint_burden(ax: plt.Axes, prevalence: pd.DataFrame) -> pd.DataFrame:
    burden = prevalence[
        prevalence["year"].eq(2024) & prevalence["outcome"].isin(BURDEN_LABEL)
    ].copy()
    burden["label"] = burden["outcome"].map(BURDEN_LABEL)
    burden["prevalence_pct"] = burden["positive_pct_known"].astype(float)
    burden["positive_n"] = burden["positive_n"].astype(int)
    burden["known_n"] = burden["known_n"].astype(int)
    burden["order"] = burden["outcome"].map({name: idx for idx, name in enumerate(BURDEN_LABEL)})
    burden = burden.sort_values("order", ascending=True)

    y = np.arange(len(burden))
    colors = [
        BLUE if outcome == "outcome_sinasc_broad_birth_status" else
        ORANGE if outcome == "outcome_sinasc_severe_birth_status" else
        GRAY
        for outcome in burden["outcome"]
    ]
    ax.barh(y, burden["prevalence_pct"], color=colors, edgecolor="white", linewidth=0.45)
    for yi, pct, n in zip(y, burden["prevalence_pct"], burden["positive_n"]):
        ax.text(pct + 0.35, yi, f"{pct:.1f}%  n={n:,}", va="center", ha="left", fontsize=6.5, color=DARK)
    label(ax, "A")
    ax.set_xlabel("Prevalence among known records, %")
    ax.set_yticks(y)
    ax.set_yticklabels(burden["label"])
    ax.set_xlim(0, max(20, burden["prevalence_pct"].max() * 1.34))
    ax.invert_yaxis()
    clean(ax, "x")
    return burden


def panel_metric(ax: plt.Axes, metrics: pd.DataFrame, metric: str, title: str, ylabel: str, panel: str) -> None:
    endpoints = list(ENDPOINT_LABEL)
    models = [
        "sinasc_overlap_inputs_lightgbm",
        "ssl_embedding_logit",
        "ssl_plus_phenotype_logit",
        "phenotype_only_logit",
    ]
    width = 0.18
    x = np.arange(len(endpoints))
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(models))
    for idx, model in enumerate(models):
        values = []
        for endpoint in endpoints:
            row = metrics[(metrics["endpoint"].eq(endpoint)) & (metrics["model"].eq(model))]
            values.append(float(row[metric].iloc[0]))
        ax.bar(x + offsets[idx], values, width=width, color=MODEL_COLOR[model], label=MODEL_LABEL[model].replace("\n", " "), edgecolor="white", linewidth=0.45)
    label(ax, panel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([ENDPOINT_LABEL[e] for e in endpoints])
    ymax = metrics[metrics["endpoint"].isin(endpoints)][metric].max()
    ax.set_ylim(0, ymax * 1.22)
    if panel == "B":
        ax.legend(loc="upper right", ncols=2, bbox_to_anchor=(1.0, 1.02), handlelength=1.3, columnspacing=0.9)
    clean(ax, "y")


def panel_phenotype_rates(ax: plt.Axes, pheno: pd.DataFrame) -> None:
    test = pheno[pheno["split"].eq("test")].copy()
    test["phenotype"] = test["phenotype"].astype(str)
    x = np.arange(len(test))
    severe = test["outcome_sinasc_severe_birth_status_rate"].astype(float).to_numpy()
    broad = test["outcome_sinasc_broad_birth_status_rate"].astype(float).to_numpy()
    ax.plot(x, severe, marker="o", color=ORANGE, lw=1.45, label="Severe birth-status")
    ax.plot(x, broad, marker="o", color=BLUE, lw=1.45, label="Broad birth-status")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{item}" for item in test["phenotype"]])
    label(ax, "D")
    ax.set_ylabel("Event rate, %")
    ax.yaxis.set_major_formatter(FuncFormatter(pct_axis))
    ax.set_ylim(0, max(broad.max(), severe.max()) * 1.22)
    ax.legend(loc="upper right", ncols=1, handlelength=1.7)
    clean(ax, "y")


def build() -> None:
    setup_style()
    metrics = pd.read_csv(TABLE_DIR / f"sinasc_ssl_stress_test_metrics_{TAG}.csv")
    pheno = pd.read_csv(TABLE_DIR / f"sinasc_ssl_phenotype_rates_{TAG}.csv")
    history = pd.read_csv(TABLE_DIR / f"sinasc_masked_tabular_ssl_history_{TAG}.csv")
    prevalence = pd.read_csv(TABLE_DIR / "sinasc_harmonized_outcome_prevalence.csv")

    figure_source = metrics[
        metrics["endpoint"].isin(ENDPOINT_LABEL)
        & metrics["model"].isin(MODEL_LABEL)
    ].copy()
    figure_source.to_csv(TABLE_DIR / f"figure7_sinasc_registry_stress_test_source_{TAG}.csv", index=False)
    pheno.to_csv(TABLE_DIR / f"figure7_sinasc_phenotype_rates_source_{TAG}.csv", index=False)
    history.to_csv(TABLE_DIR / f"figure7_sinasc_ssl_history_source_{TAG}.csv", index=False)
    endpoint_burden = prevalence[
        prevalence["year"].eq(2024) & prevalence["outcome"].isin(BURDEN_LABEL)
    ].copy()
    endpoint_burden["label"] = endpoint_burden["outcome"].map(BURDEN_LABEL)
    endpoint_burden.to_csv(TABLE_DIR / f"figure7_sinasc_2024_endpoint_burden_source_{TAG}.csv", index=False)

    SUBMISSION_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    for source in [
        TABLE_DIR / "sinasc_natality_variable_crosswalk.csv",
        TABLE_DIR / "sinasc_harmonized_outcome_prevalence.csv",
        TABLE_DIR / "sinasc_harmonized_missingness.csv",
        TABLE_DIR / f"figure7_sinasc_registry_stress_test_source_{TAG}.csv",
        TABLE_DIR / f"figure7_sinasc_2024_endpoint_burden_source_{TAG}.csv",
        TABLE_DIR / f"figure7_sinasc_phenotype_rates_source_{TAG}.csv",
        TABLE_DIR / f"figure7_sinasc_ssl_history_source_{TAG}.csv",
        TABLE_DIR / f"sinasc_ssl_cluster_selection_{TAG}.csv",
    ]:
        target = SUBMISSION_SOURCE_DIR / source.name
        target.write_bytes(source.read_bytes())

    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.15), constrained_layout=True)
    panel_endpoint_burden(axes[0, 0], prevalence)
    panel_metric(
        axes[0, 1],
        figure_source,
        "auprc_over_prevalence",
        "AUPRC enrichment over prevalence",
        "AUPRC / prevalence",
        "B",
    )
    panel_metric(
        axes[1, 0],
        figure_source,
        "top_enrichment_over_prevalence",
        "Top 1% risk enrichment",
        "Event rate / prevalence",
        "C",
    )
    panel_phenotype_rates(axes[1, 1], pheno)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_FIG_DIR.mkdir(parents=True, exist_ok=True)
    name = "Figure_7_sinasc_registry_stress_test"
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(FIGURE_DIR / f"{name}{suffix}", bbox_inches="tight")
        fig.savefig(SUBMISSION_FIG_DIR / f"{name}{suffix}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build()
