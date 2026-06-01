#!/usr/bin/env python
"""Create temporal-baseline summary figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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


def plot_temporal_auprc() -> None:
    data = pd.read_csv(TABLE_DIR / "temporal_baseline_2016_2024_metrics.csv")
    data = data[data["probability_type"] == "platt"].copy()
    data["endpoint_short"] = data["endpoint"].map(short_endpoint)
    data["label"] = data["endpoint_short"] + "\n" + data["feature_set"]
    data = data.sort_values(["endpoint", "auprc"], ascending=[True, False])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = range(len(data))
    ax.bar(x, data["auprc"], color="#3568A8", label="AUPRC")
    ax.scatter(x, data["prevalence_test"], color="#D95F02", zorder=3, label="Prevalence")
    ax.set_xticks(list(x))
    ax.set_xticklabels(data["label"], rotation=25, ha="right")
    ax.set_ylabel("AUPRC")
    ax.set_title("Temporal Baseline on 2024 Test Set")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "temporal_baseline_2016_2024_auprc.png", dpi=300)
    plt.close(fig)


def plot_temporal_enrichment() -> None:
    data = pd.read_csv(TABLE_DIR / "temporal_baseline_2016_2024_top_risk_enrichment.csv")
    data = data[
        (data["probability_type"] == "platt")
        & (data["feature_set"] == "all_inputs")
        & (data["top_fraction"].isin([0.01, 0.05, 0.10]))
    ].copy()
    data["endpoint_short"] = data["endpoint"].map(short_endpoint)
    data["top_label"] = (100 * data["top_fraction"]).astype(int).astype(str) + "%"
    endpoints = list(data["endpoint_short"].unique())
    fractions = ["1%", "5%", "10%"]
    colors = {"1%": "#2F6B9A", "5%": "#57A773", "10%": "#D08C32"}

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    width = 0.22
    for offset, fraction in enumerate(fractions):
        subset = data[data["top_label"] == fraction].set_index("endpoint_short")
        values = [subset.loc[item, "enrichment_over_prevalence"] for item in endpoints]
        positions = [idx + (offset - 1) * width for idx in range(len(endpoints))]
        ax.bar(positions, values, width=width, label=f"Top {fraction}", color=colors[fraction])
    ax.axhline(1, color="#555555", linewidth=0.8)
    ax.set_xticks(list(range(len(endpoints))))
    ax.set_xticklabels(endpoints)
    ax.set_ylabel("Event-rate enrichment over prevalence")
    ax.set_title("Temporal Top-Risk Enrichment on 2024 Test Set")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "temporal_baseline_2016_2024_top_risk_enrichment.png", dpi=300)
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_temporal_auprc()
    plot_temporal_enrichment()
    print(f"wrote {FIGURE_DIR / 'temporal_baseline_2016_2024_auprc.png'}")
    print(f"wrote {FIGURE_DIR / 'temporal_baseline_2016_2024_top_risk_enrichment.png'}")


if __name__ == "__main__":
    main()
