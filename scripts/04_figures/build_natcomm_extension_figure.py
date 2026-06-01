#!/usr/bin/env python
"""Build Figure 6 for full-scale SSL and linked severe-endpoint transfer analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "results" / "figures"
SUBMISSION_FIG_DIR = PROJECT_ROOT / "submission_package" / "springer_nature_latex" / "figures"
BMC_FIG_DIR = PROJECT_ROOT / "submission_package" / "bmc_midd_latex_upload" / "figures"
PRETRAIN_TAG = "full2016_2022_mask035_d48_l2_cuda"
LINKED_TAG = "full2016_2022_mask035_d48_l2_cuda_full2023dev"
FULL_2024_TAG = "full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "figure.dpi": 160,
        }
    )


def panel_label(ax, label: str) -> None:
    ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def save_figure(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_FIG_DIR.mkdir(parents=True, exist_ok=True)
    BMC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf", ".svg"]:
        path = FIG_DIR / f"{name}{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        fig.savefig(SUBMISSION_FIG_DIR / f"{name}{suffix}", dpi=300, bbox_inches="tight")
        if suffix == ".png":
            fig.savefig(BMC_FIG_DIR / f"{name}{suffix}", dpi=300, bbox_inches="tight")


def build() -> Path:
    setup_style()
    history_full = pd.read_csv(PROJECT_ROOT / "results" / "tables" / f"masked_tabular_ssl_history_{PRETRAIN_TAG}.csv")
    history_proto = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "masked_tabular_ssl_history.csv")
    incremental = pd.read_csv(PROJECT_ROOT / "results" / "tables" / f"incremental_ssl_lightgbm_metrics_{FULL_2024_TAG}.csv")
    linked_rates = pd.read_csv(PROJECT_ROOT / "results" / "tables" / f"linked_infant_death_phenotype_rate_bootstrap_ci_{LINKED_TAG}.csv")
    linked_high = pd.read_csv(PROJECT_ROOT / "results" / "tables" / f"linked_infant_death_predefined_highrisk_enrichment_{LINKED_TAG}.csv")
    supervised_transfer = pd.read_csv(PROJECT_ROOT / "results" / "tables" / f"linked_infant_death_supervised_transfer_baseline_{LINKED_TAG}.csv")

    fig = plt.figure(figsize=(11.0, 7.6))
    gs = fig.add_gridspec(2, 2, wspace=0.35, hspace=0.45)
    blue = "#4C78A8"
    orange = "#F58518"
    green = "#54A24B"
    grey = "#7F7F7F"

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")
    pretrain = pd.DataFrame(
        [
            {
                "run": "Prototype\n140k train",
                "train_rows": 140_000,
                "dev_loss": float(history_proto.iloc[-1]["dev_loss"]),
            },
            {
                "run": "Full-scale\n26.35M train",
                "train_rows": int(history_full.iloc[-1]["train_rows"]),
                "dev_loss": float(history_full.iloc[-1]["dev_loss"]),
            },
        ]
    )
    ax.bar(pretrain["run"], pretrain["dev_loss"], color=[grey, blue], width=0.55)
    for i, row in pretrain.iterrows():
        ax.text(i, row["dev_loss"] + 0.008, f"{row['dev_loss']:.3f}", ha="center", va="bottom")
    ax.set_ylabel("2023 reconstruction loss")
    ax.set_ylim(0, max(pretrain["dev_loss"]) * 1.25)
    ax.grid(axis="x", visible=False)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "B")
    endpoint_order = [
        "outcome_maternal_morbidity_core",
        "outcome_severe_neonatal_no_nicu",
    ]
    endpoint_labels = {
        "outcome_maternal_morbidity_core": "Maternal\nmorbidity",
        "outcome_severe_neonatal_no_nicu": "Severe\nneonatal",
    }
    feature_order = ["all_inputs", "ssl_embeddings", "all_inputs_plus_ssl"]
    feature_labels = ["All inputs", "SSL", "All + SSL"]
    colors = [blue, orange, green]
    x = np.arange(len(endpoint_order))
    width = 0.22
    for j, feature in enumerate(feature_order):
        vals = []
        for endpoint in endpoint_order:
            vals.append(
                float(incremental[(incremental["endpoint"].eq(endpoint)) & (incremental["feature_set"].eq(feature))]["auprc"].iloc[0])
            )
        ax.bar(x + (j - 1) * width, vals, width=width, label=feature_labels[j], color=colors[j])
    ax.set_xticks(x)
    ax.set_xticklabels([endpoint_labels[item] for item in endpoint_order])
    ax.set_ylabel("2024 AUPRC")
    ax.set_ylim(0, 0.34)
    ax.legend(
        frameon=False,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.62),
        borderaxespad=0,
        handlelength=1.4,
    )

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "C")
    infant = linked_rates[linked_rates["outcome"].eq("outcome_infant_death")].sort_values("phenotype")
    x = np.linspace(1 / (2 * len(infant)), 1 - 1 / (2 * len(infant)), len(infant))
    y = infant["event_rate"].to_numpy() * 100
    yerr = np.vstack([
        y - infant["ci_low"].to_numpy() * 100,
        infant["ci_high"].to_numpy() * 100 - y,
    ])
    ax.errorbar(x, y, yerr=yerr, fmt="o", color=blue, capsize=4)
    ax.set_xlim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{int(v)}" for v in infant["phenotype"]])
    ax.set_ylabel("Infant death rate, %")
    ax.set_xlabel("Fixed SSL phenotype")

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "D")
    infant_transfer = supervised_transfer[supervised_transfer["outcome"].eq("outcome_infant_death")].copy()
    phenotype_row = infant_transfer[infant_transfer["rule"].eq("phenotype_0")].iloc[0]
    lgbm_row = infant_transfer[infant_transfer["rule"].str.startswith("all_input_lgbm_top_4.579")].iloc[0]
    bench = pd.DataFrame(
        [
            {
                "label": "SSL phenotype 0",
                "capture": phenotype_row["event_capture_pct"],
                "rate": phenotype_row["event_rate_selected"] * 100,
                "enrichment": phenotype_row["enrichment_over_prevalence"],
                "color": blue,
            },
            {
                "label": "All-input LightGBM\nmatched fraction",
                "capture": lgbm_row["event_capture_pct"],
                "rate": lgbm_row["event_rate_selected"] * 100,
                "enrichment": lgbm_row["enrichment_over_prevalence"],
                "color": orange,
            },
        ]
    )
    x = np.arange(len(bench))
    ax.bar(x, bench["capture"], color=bench["color"], width=0.48)
    for i, row in bench.iterrows():
        ax.text(
            i,
            row["capture"] + 1.5,
            f"{row['capture']:.1f}%\n{row['rate']:.2f}%, {row['enrichment']:.2f}x",
            ha="center",
            va="bottom",
            fontsize=6.8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(bench["label"])
    ax.set_ylabel("Infant deaths captured, %")
    ax.set_ylim(0, max(bench["capture"]) * 1.28)
    ax.grid(axis="x", visible=False)

    name = "Figure_6_fullscale_linked_validation"
    save_figure(fig, name)
    plt.close(fig)
    return FIG_DIR / f"{name}.png"


def main() -> None:
    print(build())


if __name__ == "__main__":
    main()
