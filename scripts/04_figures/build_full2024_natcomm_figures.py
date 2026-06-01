#!/usr/bin/env python
"""Build full-2024 manuscript figures 2-5 from full-year source tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
OBJECT_DIR = PROJECT_ROOT / "results" / "objects"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
SUBMISSION_FIG_DIR = PROJECT_ROOT / "submission_package" / "springer_nature_latex" / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

TAG = "full2016_2022_mask035_d48_l2_cuda_full2023dev_full2024"
BASE_TAG = "full2016_2022_mask035_d48_l2_cuda_full2023dev"

BLUE = "#4E79A7"
LIGHT_BLUE = "#A0CBE8"
ORANGE = "#F28E2B"
LIGHT_ORANGE = "#FFBE7D"
GREEN = "#59A14F"
PURPLE = "#B07AA1"
GRAY = "#7F7F7F"
LIGHT_GRAY = "#E6E6E6"
DARK = "#202020"
GRID = "#E5E5E5"

ENDPOINTS = ["outcome_maternal_morbidity_core", "outcome_severe_neonatal_no_nicu"]
ENDPOINT_LABEL = {
    "outcome_maternal_morbidity_core": "Maternal core morbidity",
    "outcome_severe_neonatal_no_nicu": "Severe neonatal outcome",
}


def table(stem: str, tag: str = TAG) -> Path:
    return TABLE_DIR / f"{stem}_{tag}.csv"


def obj(stem: str, tag: str = TAG) -> Path:
    return OBJECT_DIR / f"{stem}_{tag}.parquet"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.titlesize": 8.3,
            "axes.labelsize": 7.4,
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


def save(fig: plt.Figure, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_FIG_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / f"{name}.png"
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(FIGURE_DIR / f"{name}{suffix}", bbox_inches="tight")
        fig.savefig(SUBMISSION_FIG_DIR / f"{name}{suffix}", bbox_inches="tight")
    plt.close(fig)
    return output


def label(ax: plt.Axes, text: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontweight="bold", fontsize=10, va="top", ha="left")


def clean(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.65, alpha=0.9)
    ax.set_axisbelow(True)


def ci_merge(data: pd.DataFrame, ci: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = data.copy()
    keys = ["endpoint", "feature_set", "model", "probability_type"]
    for metric in metrics:
        part = ci[ci["metric"].eq(metric)][keys + ["point", "ci_low", "ci_high"]].copy()
        part = part.rename(
            columns={
                "point": f"{metric}_point",
                "ci_low": f"{metric}_ci_low",
                "ci_high": f"{metric}_ci_high",
            }
        )
        out = out.merge(part, on=keys, how="left")
        if metric in out:
            out[metric] = out[metric].fillna(out[f"{metric}_point"])
        else:
            out[metric] = out[f"{metric}_point"]
    return out


def figure2_source() -> pd.DataFrame:
    inc = pd.read_csv(table("incremental_ssl_lightgbm_metrics"))
    inc_ci = pd.read_csv(table("incremental_ssl_lightgbm_bootstrap_ci"))
    inc = ci_merge(inc, inc_ci, ["auprc", "top1_enrichment_over_prevalence"])
    inc["model_label"] = inc["feature_set"].map(
        {
            "all_inputs": "LightGBM all inputs",
            "all_inputs_plus_ssl": "All inputs + SSL",
            "ssl_embeddings": "SSL LightGBM",
        }
    )

    ssl = pd.read_csv(table("ssl_phenotype_risk_metrics"))
    ssl = ssl[ssl["probability_type"].eq("platt")].copy()
    ssl_ci = pd.read_csv(table("ssl_phenotype_risk_metric_bootstrap_ci"))
    ssl = ci_merge(ssl, ssl_ci, ["auprc", "top1_enrichment_over_prevalence"])
    ssl["model_label"] = ssl["feature_set"].map(
        {
            "ssl_embedding": "SSL logistic",
            "ssl_plus_phenotype": "SSL + phenotype",
            "phenotype": "Phenotype only",
        }
    )
    data = pd.concat([inc, ssl], ignore_index=True, sort=False).dropna(subset=["model_label"])
    order = {
        "LightGBM all inputs": 0,
        "All inputs + SSL": 1,
        "SSL LightGBM": 2,
        "SSL logistic": 3,
        "SSL + phenotype": 4,
        "Phenotype only": 5,
    }
    data["model_order"] = data["model_label"].map(order)
    data = data.sort_values(["endpoint", "model_order"])
    data.to_csv(table("figure2_full2024_source"), index=False)
    return data


def plot_model_metric(ax: plt.Axes, data: pd.DataFrame, endpoint: str, metric: str, panel: str, title: str, ylabel: str) -> None:
    order = ["LightGBM all inputs", "All inputs + SSL", "SSL LightGBM", "SSL logistic", "SSL + phenotype", "Phenotype only"]
    colors = {
        "LightGBM all inputs": BLUE,
        "All inputs + SSL": "#2F6FA3",
        "SSL LightGBM": PURPLE,
        "SSL logistic": LIGHT_ORANGE,
        "SSL + phenotype": ORANGE,
        "Phenotype only": GRAY,
    }
    short = {
        "LightGBM all inputs": "LGBM\nall",
        "All inputs + SSL": "All +\nSSL",
        "SSL LightGBM": "SSL\nLGBM",
        "SSL logistic": "SSL\nlogit",
        "SSL + phenotype": "SSL +\nphen",
        "Phenotype only": "Phen\nonly",
    }
    panel_data = data[data["endpoint"].eq(endpoint)].set_index("model_label").reindex(order).dropna(subset=[metric])
    x = np.arange(len(panel_data))
    ax.scatter(x, panel_data[metric], s=35, color=[colors[item] for item in panel_data.index], edgecolor="white", linewidth=0.35, zorder=3)
    low = f"{metric}_ci_low"
    high = f"{metric}_ci_high"
    if low in panel_data and high in panel_data:
        has_ci = panel_data[low].notna() & panel_data[high].notna()
        y = panel_data.loc[has_ci, metric].to_numpy(float)
        yerr = np.vstack(
            [
                np.maximum(0, y - panel_data.loc[has_ci, low].to_numpy(float)),
                np.maximum(0, panel_data.loc[has_ci, high].to_numpy(float) - y),
            ]
        )
        ax.errorbar(x[has_ci.to_numpy()], y, yerr=yerr, fmt="none", ecolor="#333333", elinewidth=0.85, capsize=2.2, zorder=2)
    ax.axhline(float(panel_data["prevalence_test"].dropna().iloc[0]) if metric == "auprc" else 1.0, color=GRAY, linestyle="--", linewidth=0.85, alpha=0.65)
    label(ax, panel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([short[item] for item in panel_data.index])
    ax.set_ylim(bottom=0, top=float(panel_data[metric].max()) * 1.18)
    clean(ax, "y")


def build_figure2() -> Path:
    data = figure2_source()
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.5))
    specs = [
        ("outcome_maternal_morbidity_core", "auprc", "A", "Maternal core morbidity", "AUPRC"),
        ("outcome_severe_neonatal_no_nicu", "auprc", "B", "Severe neonatal outcome", "AUPRC"),
        ("outcome_maternal_morbidity_core", "top1_enrichment_over_prevalence", "C", "Maternal core morbidity", "Top 1% enrichment"),
        ("outcome_severe_neonatal_no_nicu", "top1_enrichment_over_prevalence", "D", "Severe neonatal outcome", "Top 1% enrichment"),
    ]
    for ax, spec in zip(axes.ravel(), specs):
        plot_model_metric(ax, data, *spec)
    fig.tight_layout(rect=[0.03, 0.02, 1, 1])
    return save(fig, "Figure_2_full_year_model_summary")


def build_figure3() -> Path:
    assign = pd.read_parquet(obj("ssl_phenotype_dev_assignments"))
    if len(assign) > 20_000:
        assign = assign.sample(n=20_000, random_state=20260527)
    selection = pd.read_csv(table("ssl_phenotype_cluster_selection"))
    stability = pd.read_csv(table("ssl_phenotype_stability"))
    rates = pd.read_csv(table("ssl_phenotype_outcome_rates"))
    rate_ci = pd.read_csv(table("ssl_phenotype_outcome_rate_bootstrap_ci"))
    profile = pd.read_csv(table("cns_phenotype_standardized_profiles"))

    fig = plt.figure(figsize=(13.2, 9.4))
    gs = fig.add_gridspec(2, 3, wspace=0.36, hspace=0.5)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    colors = {0: ORANGE, 1: LIGHT_BLUE, 2: GRAY}

    ax = axes[0]
    label(ax, "A")
    for phenotype, group in assign.groupby("phenotype"):
        ax.scatter(group["pc1"], group["pc2"], s=3, alpha=0.35, color=colors[int(phenotype)], label=f"P{int(phenotype)}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=3, loc="best", handletextpad=0.2)

    ax = axes[1]
    label(ax, "B")
    size_data = rates[rates["split"].eq("test")].sort_values("phenotype")
    ax.bar(size_data["phenotype"].astype(str), size_data["n"], color=[colors[int(p)] for p in size_data["phenotype"]], width=0.68)
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Records")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:.1f}M" if value >= 1_000_000 else f"{int(value / 1000)}k"))
    clean(ax, "y")

    ax = axes[2]
    label(ax, "C")
    diagnostics = [
        ("Silhouette", selection["silhouette"].to_numpy(float), "higher"),
        ("Min cluster\nproportion", selection["min_cluster_prop"].to_numpy(float), "higher"),
        ("Davies-Bouldin\nindex", selection["davies_bouldin"].to_numpy(float), "lower"),
    ]
    scaled_rows = []
    for _, values, direction in diagnostics:
        scaled = (values - values.min()) / max(values.max() - values.min(), 1e-9)
        scaled_rows.append(1 - scaled if direction == "lower" else scaled)
    matrix = np.vstack(scaled_rows)
    ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    k3_idx = int(np.where(selection["k"].to_numpy() == 3)[0][0])
    ax.add_patch(patches.Rectangle((k3_idx - 0.5, -0.5), 1.0, len(diagnostics), fill=False, edgecolor=ORANGE, linewidth=1.8))
    ax.set_xlabel("Number of clusters")
    ax.set_xticks(np.arange(len(selection)))
    ax.set_xticklabels(selection["k"].astype(int))
    ax.set_yticks(np.arange(len(diagnostics)))
    ax.set_yticklabels([item[0] for item in diagnostics])

    ax = axes[3]
    label(ax, "D")
    metrics = [("ari_vs_primary", "ARI"), ("nmi_vs_primary", "NMI"), ("min_cluster_prop", "Min prop.")]
    ypos = np.arange(len(metrics))[::-1]
    for y, (col, short) in zip(ypos, metrics):
        values = stability[col].to_numpy(float)
        ax.boxplot(values, vert=False, positions=[y], widths=0.34, patch_artist=True, boxprops=dict(facecolor=LIGHT_BLUE, edgecolor=BLUE, linewidth=0.8), medianprops=dict(color=BLUE, linewidth=1.15), whiskerprops=dict(color=BLUE, linewidth=0.8), capprops=dict(color=BLUE, linewidth=0.8), flierprops=dict(marker="o", markersize=2, markerfacecolor=BLUE, markeredgecolor="none", alpha=0.25))
        ax.text(min(float(np.median(values)) + 0.035, 1.03), y, f"{np.median(values):.3f}", va="center", ha="left", fontsize=6.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([item[1] for item in metrics])
    ax.set_xlim(0, 1.14)
    ax.set_xlabel("Metric value")
    clean(ax, "x")

    ax = axes[4]
    label(ax, "E")
    for endpoint, offset, color in zip(ENDPOINTS, [-0.08, 0.08], [BLUE, ORANGE]):
        panel = rate_ci[rate_ci["endpoint"].eq(endpoint)].sort_values("phenotype")
        x = panel["phenotype"].to_numpy(float) + offset
        y = panel["event_rate"].to_numpy(float) * 100
        yerr = np.vstack([y - panel["ci_low"].to_numpy(float) * 100, panel["ci_high"].to_numpy(float) * 100 - y])
        ax.errorbar(x, y, yerr=yerr, fmt="o", color=color, markeredgecolor="white", markeredgewidth=0.35, capsize=2.2, label=ENDPOINT_LABEL[endpoint])
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("2024 event rate (%)")
    ax.legend(loc="upper right", handletextpad=0.25)
    clean(ax, "y")

    ax = axes[5]
    label(ax, "F")
    wanted = ["Age", "BMI", "Prenatal visits", "Weight gain", "GDM", "Hypertension", "Prior preterm", "Prior cesarean", "Smoking", "Infertility/ART", "Multiple gestation", "Chlamydia"]
    pivot = profile.pivot(index="phenotype", columns="feature", values="standardized_difference").reindex(index=[0, 1, 2], columns=wanted)
    mat = pivot.to_numpy(float)
    lim = max(0.75, float(np.nanmax(np.abs(mat))))
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-lim, vmax=lim)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(["P0", "P1", "P2"])
    ax.set_xticks(np.arange(len(wanted)))
    ax.set_xticklabels(wanted, rotation=34, ha="right", rotation_mode="anchor")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label("Standardized difference")
    return save(fig, "Figure_3_ssl_phenotype_discovery")


def build_figure4() -> Path:
    ci = pd.read_csv(table("ssl_phenotype_risk_metric_bootstrap_ci"))
    calibration = pd.read_csv(table("ssl_phenotype_calibration_bins"))
    pca = pd.read_csv(table("ssl_pca_sensitivity"))
    metrics = pd.read_csv(table("ssl_phenotype_risk_metrics"))
    fig = plt.figure(figsize=(13.2, 8.8))
    gs = fig.add_gridspec(2, 3, wspace=0.36, hspace=0.52)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    feature_order = ["phenotype", "ssl_embedding", "ssl_plus_phenotype"]
    labels = {"phenotype": "Phenotype", "ssl_embedding": "SSL embedding", "ssl_plus_phenotype": "SSL + phenotype"}
    for ax, metric, panel, title, ylabel in [
        (axes[0], "auprc_over_prevalence", "A", "AUPRC enrichment", "AUPRC / prevalence"),
        (axes[1], "top1_enrichment_over_prevalence", "B", "Top 1% enrichment", "Enrichment over prevalence"),
    ]:
        for endpoint, offset, color in zip(ENDPOINTS, [-0.11, 0.11], [BLUE, ORANGE]):
            panel_data = ci[(ci["endpoint"].eq(endpoint)) & (ci["metric"].eq(metric))].set_index("feature_set").reindex(feature_order)
            x = np.arange(len(panel_data)) + offset
            y = panel_data["point"].to_numpy(float)
            yerr = np.vstack(
                [
                    np.maximum(0, y - panel_data["ci_low"].to_numpy(float)),
                    np.maximum(0, panel_data["ci_high"].to_numpy(float) - y),
                ]
            )
            ax.errorbar(x, y, yerr=yerr, fmt="o", color=color, markeredgecolor="white", markeredgewidth=0.35, capsize=2.2, label=ENDPOINT_LABEL[endpoint])
        label(ax, panel)
        ax.set_xticks(np.arange(len(feature_order)))
        ax.set_xticklabels([labels[item] for item in feature_order], rotation=24, ha="right")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", handletextpad=0.25)
        clean(ax, "y")

    for ax, endpoint, panel_label in [(axes[2], ENDPOINTS[0], "C"), (axes[3], ENDPOINTS[1], "D")]:
        label(ax, panel_label)
        cal = calibration[(calibration["endpoint"].eq(endpoint)) & (calibration["feature_set"].eq("ssl_plus_phenotype")) & (calibration["probability_type"].eq("platt"))].sort_values("bin")
        ax.plot(cal["mean_pred"], cal["event_rate"], marker="o", ms=3.8, color=ORANGE, lw=1.35)
        max_v = max(float(cal["mean_pred"].max()), float(cal["event_rate"].max())) * 1.1
        ax.plot([0, max_v], [0, max_v], color=GRAY, lw=0.95, linestyle="--")
        row = metrics[(metrics["endpoint"].eq(endpoint)) & (metrics["feature_set"].eq("ssl_plus_phenotype")) & (metrics["probability_type"].eq("platt"))].iloc[0]
        ax.text(0.05, 0.92, f"ECE = {row['ece_10']:.4f}\nBrier = {row['brier']:.4f}", transform=ax.transAxes, va="top", ha="left", fontsize=7.6, bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=LIGHT_GRAY))
        ax.set_xlim(0, max_v)
        ax.set_ylim(0, max_v)
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Observed event rate")
        clean(ax, "both")

    for ax, metric, panel_label, title, ylabel in [(axes[4], "auprc", "E", "PCA sensitivity: AUPRC", "AUPRC"), (axes[5], "top1_enrichment_over_prevalence", "F", "PCA sensitivity: top 1%", "Enrichment over prevalence")]:
        label(ax, panel_label)
        for endpoint, color in zip(ENDPOINTS, [BLUE, ORANGE]):
            panel = pca[pca["endpoint"].eq(endpoint)].sort_values("feature_order")
            ax.plot(panel["feature_set"], panel[metric], marker="o", ms=3.8, lw=1.35, color=color, label=ENDPOINT_LABEL[endpoint])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(loc="best", handletextpad=0.25)
        clean(ax, "y")
    return save(fig, "Figure_4_calibration_sensitivity")


def build_figure5() -> Path:
    topk = pd.read_csv(table("cns_topk_utility"))
    dca = pd.read_csv(table("cns_decision_curve"))
    birth = pd.read_csv(table("phenotype_birth_status_profile"))
    cause = pd.read_csv(table("linked_infant_death_cause_highrisk_enrichment", BASE_TAG))
    subgroup = pd.read_csv(table("cns_subgroup_metrics"))

    fig = plt.figure(figsize=(13.2, 8.9))
    gs = fig.add_gridspec(2, 3, wspace=0.36, hspace=0.48)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    for ax, endpoint, panel, title in [(axes[0], ENDPOINTS[1], "A", "Severe neonatal top-risk triage"), (axes[1], ENDPOINTS[0], "B", "Maternal morbidity top-risk triage")]:
        label(ax, panel)
        panel_data = topk[topk["endpoint"].eq(endpoint)].sort_values("top_fraction")
        ax.plot(panel_data["top_fraction"] * 100, panel_data["event_rate"] * 100, marker="o", color=ORANGE, lw=1.4)
        baseline = float(panel_data["baseline_prevalence"].iloc[0]) * 100
        ax.axhline(baseline, color=GRAY, lw=0.9, linestyle="--")
        ax.set_xlabel("Highest-risk records selected (%)")
        ax.set_ylabel("Observed event rate (%)")
        clean(ax, "y")

    for ax, endpoint, panel, title in [(axes[2], ENDPOINTS[1], "C", "Decision curve: severe neonatal"), (axes[3], ENDPOINTS[0], "D", "Decision curve: maternal morbidity")]:
        label(ax, panel)
        panel_data = dca[dca["endpoint"].eq(endpoint)].sort_values("threshold")
        ax.plot(panel_data["threshold"], panel_data["net_benefit"], marker="o", color=ORANGE, lw=1.3, label="SSL + phenotype")
        ax.plot(panel_data["threshold"], panel_data["treat_all_net_benefit"], color=GRAY, linestyle="--", lw=1.0, label="Treat all")
        ax.axhline(0, color="#333333", lw=0.85, linestyle=":", label="Treat none")
        ax.set_xlabel("Risk threshold")
        ax.set_ylabel("Net benefit")
        ax.legend(loc="best", handletextpad=0.25)
        clean(ax, "y")

    ax = axes[4]
    label(ax, "E")
    metrics = [
        ("preterm_lt37_rate", "Preterm"),
        ("low_birthweight_lt2500g_rate", "LBW"),
        ("very_low_birthweight_lt1500g_rate", "VLBW"),
        ("low_apgar5_lt7_rate", "Low Apgar"),
    ]
    x = np.arange(len(metrics))
    width = 0.24
    colors = [ORANGE, LIGHT_BLUE, GRAY]
    for j, phenotype in enumerate([0, 1, 2]):
        row = birth[birth["phenotype"].eq(phenotype)].iloc[0]
        ax.bar(x + (j - 1) * width, [row[col] * 100 for col, _ in metrics], width=width, color=colors[j], label=f"P{phenotype}")
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in metrics], rotation=18, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.legend(loc="upper right", ncol=3, handlelength=1.2)
    clean(ax, "y")

    ax = axes[5]
    label(ax, "F")
    cause = cause.sort_values("enrichment_over_prevalence", ascending=True)
    ax.barh(cause["cause_group"], cause["enrichment_over_prevalence"], color=[ORANGE if item == "Perinatal conditions" else BLUE for item in cause["cause_group"]])
    ax.axvline(1.0, color=GRAY, linestyle="--", lw=0.9)
    ax.set_xlabel("Phenotype 0 enrichment over prevalence")
    clean(ax, "x")

    fig.tight_layout(rect=[0.03, 0.02, 1, 1])
    return save(fig, "Figure_5_clinical_utility_interpretability")


def main() -> None:
    setup_style()
    outputs = [build_figure2(), build_figure3(), build_figure4(), build_figure5()]
    lines = ["# Full-2024 Figure Build Report", "", "Rebuilt Figures 2-5 using full-year 2024 source-data tables where available.", ""]
    lines.extend([f"- `{path}`" for path in outputs])
    (DOCS_DIR / f"42_full2024_figure_build_{TAG}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
