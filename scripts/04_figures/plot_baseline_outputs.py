#!/usr/bin/env python
"""Create baseline summary figures from saved result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"


def short_endpoint(value: str) -> str:
    return (
        value.replace("outcome_", "")
        .replace("maternal_morbidity_core", "Maternal core morbidity")
        .replace("severe_neonatal_no_nicu", "Severe neonatal")
    )


def plot_top_risk_enrichment() -> None:
    path = TABLE_DIR / "nat2024_baseline_top_risk_enrichment.csv"
    data = pd.read_csv(path)
    data = data[
        (data["probability_type"] == "platt")
        & (data["feature_set"] == "all_inputs")
        & (data["model"] == "lightgbm")
    ].copy()
    data["endpoint_short"] = data["endpoint"].map(short_endpoint)
    data["top_label"] = (100 * data["top_fraction"]).astype(int).astype(str) + "%"

    endpoints = list(data["endpoint_short"].unique())
    fractions = ["1%", "5%", "10%"]
    colors = {"1%": "#2F6B9A", "5%": "#57A773", "10%": "#D08C32"}

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    width = 0.22
    x = range(len(endpoints))
    for offset, fraction in enumerate(fractions):
        subset = data[data["top_label"] == fraction].set_index("endpoint_short")
        values = [subset.loc[item, "enrichment_over_prevalence"] for item in endpoints]
        positions = [item + (offset - 1) * width for item in x]
        ax.bar(positions, values, width=width, label=f"Top {fraction}", color=colors[fraction])
    ax.axhline(1, color="#555555", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(endpoints)
    ax.set_ylabel("Event-rate enrichment over prevalence")
    ax.set_title("Top-risk enrichment, Platt-calibrated LightGBM")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "nat2024_baseline_top_risk_enrichment.png", dpi=300)
    plt.close(fig)


def plot_importance() -> None:
    path = TABLE_DIR / "nat2024_baseline_lightgbm_importance.csv"
    data = pd.read_csv(path)
    data = data[
        (data["feature_set"] == "all_inputs")
        & (data["model"] == "lightgbm")
        & (data["endpoint"].isin(
            ["outcome_maternal_morbidity_core", "outcome_severe_neonatal_no_nicu"]
        ))
    ].copy()
    data["endpoint_short"] = data["endpoint"].map(short_endpoint)
    data["feature_clean"] = data["feature"].str.replace("input_", "", regex=False)
    data["feature_clean"] = data["feature_clean"].str.replace(
        "missing_input_", "missing_", regex=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))
    for ax, endpoint in zip(axes, ["Maternal core morbidity", "Severe neonatal"]):
        subset = (
            data[data["endpoint_short"] == endpoint]
            .sort_values("importance_gain", ascending=False)
            .head(12)
            .sort_values("importance_gain")
        )
        ax.barh(subset["feature_clean"], subset["importance_gain"], color="#3A6EA5")
        ax.set_title(endpoint)
        ax.set_xlabel("LightGBM gain")
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.suptitle("Top LightGBM Feature Importance", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURE_DIR / "nat2024_baseline_lightgbm_importance.png", dpi=300)
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_top_risk_enrichment()
    plot_importance()
    print(f"wrote {FIGURE_DIR / 'nat2024_baseline_top_risk_enrichment.png'}")
    print(f"wrote {FIGURE_DIR / 'nat2024_baseline_lightgbm_importance.png'}")


if __name__ == "__main__":
    main()
