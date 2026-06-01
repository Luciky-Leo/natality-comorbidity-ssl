#!/usr/bin/env python
"""Build full-scale Nat Commun-oriented manuscript figures from tagged source tables."""

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
DOCS_DIR = PROJECT_ROOT / "docs"
TAG = "full2016_2022_mask035_d48_l2_cuda"

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

ENDPOINTS = [
    "outcome_maternal_morbidity_core",
    "outcome_severe_neonatal_no_nicu",
]
ENDPOINT_LABEL = {
    "outcome_maternal_morbidity_core": "Maternal core morbidity",
    "outcome_severe_neonatal_no_nicu": "Severe neonatal outcome",
}


def path(stem: str, suffix: str = "csv") -> Path:
    return TABLE_DIR / f"{stem}_{TAG}.{suffix}"


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
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def label(ax: plt.Axes, text: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontweight="bold", fontsize=10, va="top", ha="left")


def clean(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.65, alpha=0.9)
    ax.set_axisbelow(True)


def add_ci_columns(data: pd.DataFrame, ci: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = data.copy()
    for metric in metrics:
        cols = ["endpoint", "feature_set", "model", "probability_type"]
        part = ci[ci["metric"] == metric][cols + ["point", "ci_low", "ci_high"]].copy()
        part = part.rename(
            columns={
                "point": f"{metric}_point",
                "ci_low": f"{metric}_ci_low",
                "ci_high": f"{metric}_ci_high",
            }
        )
        out = out.merge(part, on=cols, how="left")
        if metric not in out.columns:
            out[metric] = out[f"{metric}_point"]
        else:
            out[metric] = out[metric].fillna(out[f"{metric}_point"])
    return out


def figure2_source() -> pd.DataFrame:
    matched = pd.read_csv(path("matched_lightgbm_metrics"))
    matched_ci = pd.read_csv(path("matched_lightgbm_bootstrap_ci"))
    matched = add_ci_columns(matched, matched_ci, ["auprc", "top1_enrichment_over_prevalence"])
    matched["model_label"] = matched["feature_set"].map(
        {
            "comorbidity_only": "LightGBM comorbidity",
            "all_inputs": "LightGBM all inputs",
        }
    )
    matched["source"] = "matched"

    ssl = pd.read_csv(path("ssl_phenotype_risk_metrics"))
    ssl = ssl[ssl["probability_type"] == "platt"].copy()
    ssl_ci = pd.read_csv(path("ssl_phenotype_risk_metric_bootstrap_ci"))
    ssl = add_ci_columns(ssl, ssl_ci, ["auprc", "top1_enrichment_over_prevalence"])
    ssl["model_label"] = ssl["feature_set"].map(
        {
            "ssl_embedding": "SSL logistic",
            "ssl_plus_phenotype": "SSL + phenotype",
            "phenotype": "Phenotype only",
        }
    )
    ssl["source"] = "ssl-logistic"

    inc = pd.read_csv(path("incremental_ssl_lightgbm_metrics"))
    inc = inc[(inc["probability_type"] == "platt") & (inc["feature_set"] != "all_inputs")].copy()
    inc_ci = pd.read_csv(path("incremental_ssl_lightgbm_bootstrap_ci"))
    inc = add_ci_columns(inc, inc_ci, ["auprc", "top1_enrichment_over_prevalence"])
    inc["model_label"] = inc["feature_set"].map(
        {
            "ssl_embeddings": "SSL LightGBM",
            "all_inputs_plus_ssl": "All inputs + SSL",
        }
    )
    inc["source"] = "incremental"

    data = pd.concat([matched, ssl, inc], ignore_index=True, sort=False)
    order = {
        "LightGBM comorbidity": 0,
        "LightGBM all inputs": 1,
        "All inputs + SSL": 2,
        "SSL LightGBM": 3,
        "SSL logistic": 4,
        "SSL + phenotype": 5,
        "Phenotype only": 6,
    }
    data["model_order"] = data["model_label"].map(order)
    return data.dropna(subset=["model_label"]).sort_values(["endpoint", "model_order"])


def plot_model_metric(ax: plt.Axes, data: pd.DataFrame, endpoint: str, metric: str, panel: str, title: str, xlabel: str) -> None:
    model_order = [
        "LightGBM comorbidity",
        "LightGBM all inputs",
        "All inputs + SSL",
        "SSL LightGBM",
        "SSL logistic",
        "SSL + phenotype",
        "Phenotype only",
    ]
    colors = {
        "LightGBM comorbidity": LIGHT_BLUE,
        "LightGBM all inputs": BLUE,
        "All inputs + SSL": "#2F6FA3",
        "SSL LightGBM": PURPLE,
        "SSL logistic": LIGHT_ORANGE,
        "SSL + phenotype": ORANGE,
        "Phenotype only": GRAY,
    }
    panel_data = data[data["endpoint"] == endpoint].set_index("model_label").reindex(model_order).dropna(subset=[metric])
    short_labels = {
        "LightGBM comorbidity": "LGBM\ncomorb",
        "LightGBM all inputs": "LGBM\nall",
        "All inputs + SSL": "All +\nSSL",
        "SSL LightGBM": "SSL\nLGBM",
        "SSL logistic": "SSL\nlogit",
        "SSL + phenotype": "SSL +\nphen",
        "Phenotype only": "Phen\nonly",
    }
    x = np.arange(len(panel_data))
    ax.scatter(
        x,
        panel_data[metric],
        s=34,
        color=[colors[item] for item in panel_data.index],
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    low = f"{metric}_ci_low"
    high = f"{metric}_ci_high"
    if low in panel_data and high in panel_data:
        has_ci = panel_data[low].notna() & panel_data[high].notna()
        y = panel_data.loc[has_ci, metric].to_numpy()
        yerr = np.vstack([y - panel_data.loc[has_ci, low].to_numpy(), panel_data.loc[has_ci, high].to_numpy() - y])
        ax.errorbar(x[has_ci.to_numpy()], y, yerr=yerr, fmt="none", ecolor="#333333", elinewidth=0.85, capsize=2.2, zorder=2)
    if metric == "auprc":
        baseline = float(panel_data["prevalence_test"].dropna().iloc[0])
        ax.axhline(baseline, color=GRAY, linestyle="--", linewidth=0.85, alpha=0.65)
    else:
        ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=0.85, alpha=0.65)
    label(ax, panel)
    ax.set_title(title, loc="left")
    ax.set_ylabel(xlabel)
    ax.set_xticks(x)
    ax.set_xticklabels([short_labels[item] for item in panel_data.index], rotation=0, ha="center")
    ax.set_xlim(-0.6, len(panel_data) - 0.4)
    ax.set_ylim(bottom=0, top=float(panel_data[metric].max()) * 1.18)
    clean(ax, "y")


def build_figure2() -> Path:
    data = figure2_source()
    data.to_csv(path("figure2_fullscale_source"), index=False)
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
    output = FIGURE_DIR / "Figure_2_matched_model_summary.png"
    save(fig, output)
    return output


def build_figure3() -> Path:
    assign = pd.read_parquet(OBJECT_DIR / f"ssl_phenotype_dev_assignments_{TAG}.parquet")
    if len(assign) > 20000:
        assign = assign.sample(n=20000, random_state=20260527)
    selection = pd.read_csv(path("ssl_phenotype_cluster_selection"))
    stability = pd.read_csv(path("ssl_phenotype_stability"))
    rates = pd.read_csv(path("ssl_phenotype_outcome_rates"))
    rate_ci = pd.read_csv(path("ssl_phenotype_outcome_rate_bootstrap_ci"))
    profile = pd.read_csv(path("cns_phenotype_standardized_profiles"))

    fig = plt.figure(figsize=(13.2, 9.4))
    gs = fig.add_gridspec(2, 3, wspace=0.36, hspace=0.5)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    colors = {0: ORANGE, 1: LIGHT_BLUE, 2: GRAY}

    ax = axes[0]
    label(ax, "A", -0.16, 1.14)
    for phenotype, group in assign.groupby("phenotype"):
        ax.scatter(group["pc1"], group["pc2"], s=3, alpha=0.35, color=colors[int(phenotype)], label=f"P{int(phenotype)}")
    ax.set_title("Development SSL embedding phenotypes")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=3, loc="best", handletextpad=0.2)

    ax = axes[1]
    label(ax, "B")
    size_data = rates[rates["split"] == "test"].sort_values("phenotype")
    ax.bar(size_data["phenotype"].astype(str), size_data["n"], color=[colors[int(p)] for p in size_data["phenotype"]], width=0.68)
    ax.set_title("2024 phenotype size")
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Records")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value / 1000)}k" if value > 0 else "0"))
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
    ax.set_title("Development-only cluster selection")
    ax.set_xlabel("Number of clusters")
    ax.set_xticks(np.arange(len(selection)))
    ax.set_xticklabels(selection["k"].astype(int))
    ax.set_yticks(np.arange(len(diagnostics)))
    ax.set_yticklabels([item[0] for item in diagnostics])
    for i, (_, values, _) in enumerate(diagnostics):
        for j, value in enumerate(values):
            text = f"{100 * value:.1f}%" if i == 1 else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=6.6, color="white" if matrix[i, j] > 0.55 else DARK)

    ax = axes[3]
    label(ax, "D")
    stability_metrics = [("ari_vs_primary", "ARI"), ("nmi_vs_primary", "NMI"), ("min_cluster_prop", "Min prop.")]
    ypos = np.arange(len(stability_metrics))[::-1]
    for y, (col, short) in zip(ypos, stability_metrics):
        values = stability[col].to_numpy(float)
        ax.boxplot(
            values,
            vert=False,
            positions=[y],
            widths=0.34,
            patch_artist=True,
            boxprops=dict(facecolor=LIGHT_BLUE, edgecolor=BLUE, linewidth=0.8),
            medianprops=dict(color=BLUE, linewidth=1.15),
            whiskerprops=dict(color=BLUE, linewidth=0.8),
            capprops=dict(color=BLUE, linewidth=0.8),
            flierprops=dict(marker="o", markersize=2.0, markerfacecolor=BLUE, markeredgecolor="none", alpha=0.25),
        )
        median = float(np.median(values))
        ax.text(min(median + 0.035, 1.03), y, f"{median:.3f}", va="center", ha="left", fontsize=6.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([item[1] for item in stability_metrics])
    ax.set_xlim(0, 1.14)
    ax.set_xlabel("Metric value")
    ax.set_title("Phenotype stability")
    clean(ax, "x")

    ax = axes[4]
    label(ax, "E")
    offsets = [-0.08, 0.08]
    endpoint_colors = [BLUE, ORANGE]
    for endpoint, offset, color in zip(ENDPOINTS, offsets, endpoint_colors):
        panel = rate_ci[rate_ci["endpoint"] == endpoint].sort_values("phenotype")
        x = panel["phenotype"].to_numpy(float) + offset
        y = panel["event_rate"].to_numpy(float) * 100
        yerr = np.vstack([y - panel["ci_low"].to_numpy(float) * 100, panel["ci_high"].to_numpy(float) * 100 - y])
        ax.errorbar(x, y, yerr=yerr, fmt="o", color=color, markeredgecolor="white", markeredgewidth=0.35, capsize=2.2, label=ENDPOINT_LABEL[endpoint])
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("2024 event rate (%)")
    ax.set_title("Outcome enrichment by phenotype")
    ax.legend(loc="upper right", handletextpad=0.25)
    clean(ax, "y")

    ax = axes[5]
    label(ax, "F")
    wanted = [
        "Age",
        "BMI",
        "Prenatal visits",
        "Weight gain",
        "GDM",
        "Hypertension",
        "Prior preterm",
        "Prior cesarean",
        "Smoking",
        "Infertility/ART",
        "Multiple gestation",
        "Chlamydia",
    ]
    pivot = profile.pivot(index="phenotype", columns="feature", values="standardized_difference").reindex(index=[0, 1, 2], columns=wanted)
    mat = pivot.to_numpy(float)
    lim = max(0.75, float(np.nanmax(np.abs(mat))))
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-lim, vmax=lim)
    ax.set_title("Standardized phenotype profile")
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(["P0", "P1", "P2"])
    ax.set_xticks(np.arange(len(wanted)))
    ax.set_xticklabels(wanted, rotation=34, ha="right", rotation_mode="anchor")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label("Standardized difference")

    output = FIGURE_DIR / "Figure_3_ssl_phenotype_discovery.png"
    save(fig, output)
    return output


def plot_ci_points(ax: plt.Axes, ci: pd.DataFrame, metric: str, panel: str, title: str, ylabel: str) -> None:
    feature_order = ["phenotype", "ssl_embedding", "ssl_plus_phenotype"]
    labels = {"phenotype": "Phenotype", "ssl_embedding": "SSL embedding", "ssl_plus_phenotype": "SSL + phenotype"}
    offsets = [-0.11, 0.11]
    for endpoint, offset, color in zip(ENDPOINTS, offsets, [BLUE, ORANGE]):
        panel_data = ci[(ci["endpoint"] == endpoint) & (ci["metric"] == metric)].set_index("feature_set").reindex(feature_order)
        x = np.arange(len(panel_data)) + offset
        y = panel_data["point"].to_numpy(float)
        yerr = np.vstack([y - panel_data["ci_low"].to_numpy(float), panel_data["ci_high"].to_numpy(float) - y])
        ax.errorbar(x, y, yerr=yerr, fmt="o", color=color, markeredgecolor="white", markeredgewidth=0.35, capsize=2.2, label=ENDPOINT_LABEL[endpoint])
    label(ax, panel)
    ax.set_title(title, loc="left")
    ax.set_xticks(np.arange(len(feature_order)))
    ax.set_xticklabels([labels[item] for item in feature_order], rotation=24, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", handletextpad=0.25)
    clean(ax, "y")


def build_figure4() -> Path:
    ci = pd.read_csv(path("ssl_phenotype_risk_metric_bootstrap_ci"))
    calibration = pd.read_csv(path("ssl_phenotype_calibration_bins"))
    pca = pd.read_csv(path("ssl_pca_sensitivity"))
    metrics = pd.read_csv(path("ssl_phenotype_risk_metrics"))

    fig = plt.figure(figsize=(13.2, 8.8))
    gs = fig.add_gridspec(2, 3, wspace=0.36, hspace=0.52)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    plot_ci_points(axes[0], ci, "auprc_over_prevalence", "A", "AUPRC enrichment", "AUPRC / prevalence")
    plot_ci_points(axes[1], ci, "top1_enrichment_over_prevalence", "B", "Top 1% enrichment", "Enrichment over prevalence")

    for ax, endpoint, panel_label in [
        (axes[2], "outcome_maternal_morbidity_core", "C"),
        (axes[3], "outcome_severe_neonatal_no_nicu", "D"),
    ]:
        label(ax, panel_label)
        cal = calibration[
            (calibration["endpoint"] == endpoint)
            & (calibration["feature_set"] == "ssl_plus_phenotype")
            & (calibration["probability_type"] == "platt")
        ].sort_values("bin")
        ax.plot(cal["mean_pred"], cal["event_rate"], marker="o", ms=3.8, color=ORANGE, lw=1.35)
        max_v = max(float(cal["mean_pred"].max()), float(cal["event_rate"].max())) * 1.1
        ax.plot([0, max_v], [0, max_v], color=GRAY, lw=0.95, linestyle="--")
        row = metrics[
            (metrics["endpoint"] == endpoint)
            & (metrics["feature_set"] == "ssl_plus_phenotype")
            & (metrics["probability_type"] == "platt")
        ].iloc[0]
        ax.text(
            0.05,
            0.92,
            f"ECE = {row['ece_10']:.4f}\nBrier = {row['brier']:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.6,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=LIGHT_GRAY),
        )
        ax.set_xlim(0, max_v)
        ax.set_ylim(0, max_v)
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Observed event rate")
        ax.set_title(f"Calibration: {ENDPOINT_LABEL[endpoint]}", loc="left")
        clean(ax, "both")

    for ax, metric, panel_label, title, ylabel in [
        (axes[4], "auprc", "E", "PCA sensitivity: AUPRC", "AUPRC"),
        (axes[5], "top1_enrichment_over_prevalence", "F", "PCA sensitivity: top 1%", "Enrichment over prevalence"),
    ]:
        label(ax, panel_label)
        for endpoint, color in zip(ENDPOINTS, [BLUE, ORANGE]):
            panel = pca[pca["endpoint"] == endpoint].sort_values("feature_order")
            ax.plot(panel["feature_set"], panel[metric], marker="o", ms=3.8, lw=1.35, color=color, label=ENDPOINT_LABEL[endpoint])
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(loc="best", handletextpad=0.25)
        clean(ax, "y")

    output = FIGURE_DIR / "Figure_4_calibration_sensitivity.png"
    save(fig, output)
    return output


def plot_topk(ax: plt.Axes, topk: pd.DataFrame, endpoint: str, panel: str) -> None:
    label(ax, panel)
    colors = {"SSL + phenotype": ORANGE, "LightGBM all inputs": BLUE}
    for model, group in topk[topk["endpoint"] == endpoint].groupby("model"):
        group = group.sort_values("top_fraction")
        ax.plot(group["top_fraction"] * 100, group["event_rate"] * 100, marker="o", ms=4.0, lw=1.35, color=colors.get(model, GRAY), label=model)
    prevalence = float((topk[topk["endpoint"] == endpoint]["event_rate"] / topk[topk["endpoint"] == endpoint]["enrichment_over_prevalence"]).dropna().iloc[0])
    ax.axhline(prevalence * 100, color=GRAY, lw=0.95, linestyle="--", label="Overall prevalence")
    ax.set_xticks([0.5, 1, 2, 5])
    ax.set_xlabel("Highest-risk records evaluated (%)")
    ax.set_ylabel("Observed event rate (%)")
    ax.set_title(f"Top-risk enrichment: {ENDPOINT_LABEL[endpoint]}", loc="left")
    ax.legend(loc="best", handletextpad=0.25)
    clean(ax, "y")


def plot_dca(ax: plt.Axes, dca: pd.DataFrame, endpoint: str, panel: str) -> None:
    label(ax, panel)
    colors = {"SSL + phenotype": ORANGE, "LightGBM all inputs": BLUE}
    panel_data = dca[dca["endpoint"] == endpoint].sort_values(["model", "threshold"])
    for model, group in panel_data.groupby("model"):
        ax.plot(group["threshold"] * 100, group["net_benefit"], marker="o", ms=3.4, lw=1.25, color=colors.get(model, GRAY), label=model)
    baseline = panel_data.drop_duplicates("threshold").sort_values("threshold")
    ax.plot(baseline["threshold"] * 100, baseline["treat_all_net_benefit"], color=GRAY, linestyle="--", lw=0.95, label="Treat all")
    ax.axhline(0, color="#AAAAAA", linestyle=":", lw=0.9, label="Treat none")
    ax.set_xlabel("Risk threshold (%)")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"Decision curve: {ENDPOINT_LABEL[endpoint]}", loc="left")
    ax.legend(loc="best", handletextpad=0.25)
    clean(ax, "y")


def build_figure5() -> Path:
    topk = pd.read_csv(path("cns_topk_utility"))
    dca = pd.read_csv(path("cns_decision_curve"))
    family = pd.read_csv(path("cns_feature_family_importance"))
    subgroup = pd.read_csv(path("cns_subgroup_metrics"))

    fig = plt.figure(figsize=(13.3, 11.0))
    gs = fig.add_gridspec(3, 2, wspace=0.36, hspace=0.58)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    plot_topk(axes[0], topk, "outcome_severe_neonatal_no_nicu", "A")
    plot_topk(axes[1], topk, "outcome_maternal_morbidity_core", "B")
    plot_dca(axes[2], dca, "outcome_severe_neonatal_no_nicu", "C")
    plot_dca(axes[3], dca, "outcome_maternal_morbidity_core", "D")

    ax = axes[4]
    label(ax, "E")
    order = family.groupby("feature_family")["gain_fraction"].mean().sort_values().index.tolist()
    y = np.arange(len(order))
    offsets = [-0.15, 0.15]
    for endpoint, offset, color in zip(ENDPOINTS, offsets, [BLUE, ORANGE]):
        panel = family[family["endpoint"] == endpoint].set_index("feature_family").reindex(order)
        values = panel["gain_fraction"].to_numpy(float) * 100
        ax.hlines(y + offset, 0, values, color=color, lw=1.35)
        ax.scatter(values, y + offset, s=24, color=color, edgecolor="white", linewidth=0.3, zorder=3, label=ENDPOINT_LABEL[endpoint])
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("LightGBM gain contribution (%)")
    ax.set_title("Feature-family contribution", loc="left")
    ax.legend(loc="lower right", handletextpad=0.25)
    clean(ax, "x")

    ax = axes[5]
    label(ax, "F")
    wanted = [
        "age_group",
        "bmi_group",
        "diabetes",
        "hypertensive_disorder",
        "infertility_art",
        "plurality",
        "prior_cesarean",
    ]
    rows = ["Age", "BMI", "Diabetes", "Hypertension", "Infertility/ART", "Plurality", "Prior cesarean"]
    filt = subgroup[(subgroup["model"] == "SSL + phenotype") & (subgroup["n"] >= 1000) & (subgroup["events"] >= 10)]
    heat = (
        filt[filt["subgroup_variable"].isin(wanted)]
        .groupby(["subgroup_variable", "endpoint"])["auprc_over_prevalence"]
        .min()
        .unstack("endpoint")
        .reindex(wanted)
        .reindex(columns=ENDPOINTS)
    )
    mat = heat.to_numpy(float)
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=1.0)
    ax.set_xticks(np.arange(len(ENDPOINTS)))
    ax.set_xticklabels(["Maternal", "Severe neonatal"], rotation=15, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title("Worst-stratum AUPRC enrichment", loc="left")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.1f}x", ha="center", va="center", fontsize=8, color="white" if mat[i, j] >= 3.5 else DARK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Minimum AUPRC / prevalence")

    output = FIGURE_DIR / "Figure_5_clinical_utility_interpretability.png"
    save(fig, output)
    return output


def write_report(outputs: list[Path]) -> None:
    lines = [
        "# Full-scale Nat Commun Figure Build Report",
        "",
        "Figures 2-5 were rebuilt from the full-scale tagged source-data tables, not from the earlier 200k prototype tables.",
        "",
        "## Outputs",
        "",
    ]
    for output in outputs:
        lines.append(f"- `{output}`")
    lines.extend(
        [
            "",
            "## Source tag",
            "",
            f"- `{TAG}`",
        ]
    )
    (DOCS_DIR / f"35_fullscale_natcomm_figure_build_{TAG}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    outputs = [build_figure2(), build_figure3(), build_figure4(), build_figure5()]
    write_report(outputs)
    for output in outputs:
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
