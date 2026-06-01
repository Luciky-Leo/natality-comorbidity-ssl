#!/usr/bin/env python
"""Build final manuscript-style multi-panel figures from source-data tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
OBJECT_DIR = PROJECT_ROOT / "results" / "objects"
DOCS_DIR = PROJECT_ROOT / "docs"

BLUE = "#4E79A7"
DARK_BLUE = "#1F4E79"
LIGHT_BLUE = "#A0CBE8"
ORANGE = "#F28E2B"
DARK_ORANGE = "#D95F02"
LIGHT_ORANGE = "#FFBE7D"
GREEN = "#59A14F"
PURPLE = "#B07AA1"
GRAY = "#7F7F7F"
DARK_GRAY = "#3A3A3A"
LIGHT_GRAY = "#E6E6E6"
GRID = "#E3E3E3"
TEXT = "#202020"

ENDPOINT_LABELS = {
    "outcome_maternal_morbidity_core": "Maternal core morbidity",
    "outcome_severe_neonatal_no_nicu": "Severe neonatal outcome",
}

FEATURE_LABELS = {
    "ssl_embedding": "SSL embedding",
    "ssl_plus_phenotype": "SSL + phenotype",
    "phenotype": "Phenotype only",
}

MODEL_COLORS = {
    "SSL + phenotype": ORANGE,
    "LightGBM all inputs": BLUE,
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
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


def panel_label(ax, label: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def add_box(ax, xy, width, height, text, facecolor, edgecolor=None, fontsize=8, weight="normal") -> None:
    edgecolor = edgecolor or facecolor
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=0.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT,
        fontweight=weight,
        wrap=True,
    )


def add_arrow(ax, start, end, color=GRAY, lw=1.2) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=2, shrinkB=2),
    )


def clean_axes(ax, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.65, alpha=0.9)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")


def save_pub_figure(fig, output: Path) -> None:
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")


def build_figure1() -> Path:
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "natality_2016_2024_split_manifest.csv")
    train_n = int(manifest.loc[manifest["split"] == "train", "rows"].sum())
    dev_n = int(manifest.loc[manifest["split"] == "development", "rows"].sum())
    test_n = int(manifest.loc[manifest["split"] == "test", "rows"].sum())
    export_meta = json.loads((TABLE_DIR / "masked_tabular_ssl_embedding_export_200k_metadata.json").read_text())

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.6))
    axes = axes.ravel()
    for ax in axes:
        ax.set_axis_off()

    ax = axes[0]
    panel_label(ax, "A")
    ax.set_title("Cohort assembly and temporal split", loc="left", pad=8)
    timeline = [
        (0.04, 0.60, 0.43, f"2016-2022\nTrain / SSL pretrain\n{train_n / 1_000_000:.2f}M records", LIGHT_BLUE),
        (0.52, 0.60, 0.20, f"2023\nDevelop / calibrate\n{dev_n / 1_000_000:.2f}M records", LIGHT_ORANGE),
        (0.78, 0.60, 0.18, f"2024\nTemporal test\n{test_n / 1_000_000:.2f}M records", "#D8D8D8"),
    ]
    for x, y, w, label, color in timeline:
        add_box(ax, (x, y), w, 0.23, label, color, fontsize=7, weight="bold")
    add_arrow(ax, (0.47, 0.715), (0.52, 0.715))
    add_arrow(ax, (0.72, 0.715), (0.78, 0.715))
    ax.text(0.04, 0.44, "CDC/NCHS public-use fixed-width files\n-> harmonized analytic cohorts", fontsize=8, ha="left")
    ax.text(
        0.52,
        0.32,
        f"Matched SSL evaluation sample:\n{export_meta['dev_rows']:,} development records\n{export_meta['test_rows']:,} test records",
        fontsize=8,
        ha="left",
        va="top",
        color=TEXT,
    )
    ax.text(0.04, 0.12, "Final labels used only after model development", fontsize=8, ha="left", color=ORANGE, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1]
    panel_label(ax, "B")
    ax.set_title("Leakage-controlled input domains", loc="left", pad=8)
    groups = [
        ("Demographics", "age, race/ethnicity,\nmarital/education"),
        ("Anthropometry", "BMI, height,\nweight gain"),
        ("Prenatal care", "timing, visits,\nWIC/payment"),
        ("Diabetes / hypertension", "pre-pregnancy and\ngestational indicators"),
        ("Smoking / infections", "cigarettes,\nselected infections"),
        ("Infertility / ART", "infertility treatment,\nART, fertility drugs"),
        ("Obstetric history", "prior preterm,\nprior cesarean, plurality"),
        ("Missingness", "feature-level\nmissing indicators"),
    ]
    y_values = np.linspace(0.78, 0.24, len(groups))
    for i, ((group, detail), y) in enumerate(zip(groups, y_values)):
        add_box(
            ax,
            (0.08 + (i % 2) * 0.42, y),
            0.34,
            0.09,
            f"{group}\n{detail}",
            "#F2F2F2",
            edgecolor="#C8C8C8",
            fontsize=7,
        )
    ax.text(0.08, 0.08, "44 categorical + 68 numeric/missingness variables", fontsize=8, ha="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[2]
    panel_label(ax, "C")
    ax.set_title("Masked tabular SSL encoder", loc="left", pad=8)
    boxes = [
        (0.03, 0.64, 0.16, "Categorical\n+ numeric tokens", "#F2F2F2"),
        (0.24, 0.64, 0.16, "Feature-position\nembeddings", "#EAEAEA"),
        (0.45, 0.64, 0.16, "Mixed\nmasking", "#FFE6CC"),
        (0.66, 0.64, 0.16, "2-layer\ntransformer", "#FAD7A0"),
        (0.84, 0.64, 0.13, "Mean pool\n48-d", "#F5B041"),
    ]
    for x, y, w, text, color in boxes:
        add_box(ax, (x, y), w, 0.16, text, color, fontsize=7, weight="bold")
    for start_x, end_x in [(0.19, 0.24), (0.40, 0.45), (0.61, 0.66), (0.82, 0.84)]:
        add_arrow(ax, (start_x, 0.67), (end_x, 0.67), color=ORANGE)
    ax.text(
        0.05,
        0.39,
        "Pretraining objective:\nmasked categorical cross-entropy\n+ masked numeric reconstruction",
        fontsize=8,
        ha="left",
        va="center",
    )
    ax.text(
        0.58,
        0.39,
        "Prototype scale:\n140,000 pretraining records\n3 CPU epochs",
        fontsize=8,
        ha="left",
        va="center",
    )
    ax.text(0.05, 0.16, "Encoder trained on 2016-2022 only; 2024 labels reserved for final evaluation", fontsize=8, ha="left", color=ORANGE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[3]
    panel_label(ax, "D")
    ax.set_title("Phenotype discovery, prediction, and reviewer checks", loc="left", pad=8)
    boxes = [
        (0.03, 0.64, 0.17, "2023\nembeddings", "#FFE6CC"),
        (0.25, 0.64, 0.17, "PCA +\nk-means", "#FAD7A0"),
        (0.47, 0.64, 0.17, "Fixed\ncentroids", "#F5B041"),
        (0.03, 0.30, 0.17, "2024\nembeddings", "#F2F2F2"),
        (0.25, 0.30, 0.17, "Phenotype\nassignment", "#EAEAEA"),
        (0.47, 0.30, 0.17, "SSL logistic\n+ Platt", "#D8E7F5"),
        (0.68, 0.30, 0.13, "LightGBM\nbaseline", "#D8E7F5"),
        (0.84, 0.30, 0.13, "AUPRC\nTop-k\nDCA", "#D8E7F5"),
    ]
    for x, y, w, text, color in boxes:
        add_box(ax, (x, y), w, 0.15, text, color, fontsize=7, weight="bold")
    for start, end in [((0.20, 0.715), (0.25, 0.715)), ((0.42, 0.715), (0.47, 0.715)), ((0.20, 0.375), (0.25, 0.375)), ((0.42, 0.375), (0.47, 0.375)), ((0.64, 0.375), (0.68, 0.375)), ((0.81, 0.375), (0.84, 0.375))]:
        add_arrow(ax, start, end)
    add_arrow(ax, (0.55, 0.64), (0.34, 0.47), color=ORANGE)
    ax.text(0.03, 0.12, "Reviewer checks added: subgroup robustness, top-k utility, decision curves, feature-family importance", fontsize=8, ha="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    output = FIGURE_DIR / "Figure_1_study_design_ssl_pipeline.png"
    fig.tight_layout()
    save_pub_figure(fig, output)
    plt.close(fig)
    return output


def build_figure2() -> Path:
    data = pd.read_csv(TABLE_DIR / "manuscript_model_summary_matched200k.csv")
    output = FIGURE_DIR / "Figure_2_matched_model_summary.png"

    metric_panels = [
        ("outcome_maternal_morbidity_core", "auprc", "A", "Maternal core morbidity", "AUPRC"),
        ("outcome_severe_neonatal_no_nicu", "auprc", "B", "Severe neonatal outcome", "AUPRC"),
        (
            "outcome_maternal_morbidity_core",
            "top1_enrichment_over_prevalence",
            "C",
            "Maternal core morbidity",
            "Top 1% enrichment",
        ),
        (
            "outcome_severe_neonatal_no_nicu",
            "top1_enrichment_over_prevalence",
            "D",
            "Severe neonatal outcome",
            "Top 1% enrichment",
        ),
    ]
    color_map = {
        "LightGBM comorbidity": LIGHT_BLUE,
        "LightGBM all inputs": BLUE,
        "SSL embedding": LIGHT_ORANGE,
        "SSL + phenotype": ORANGE,
        "Phenotype only": GRAY,
    }
    model_order = [
        "LightGBM comorbidity",
        "LightGBM all inputs",
        "SSL embedding",
        "SSL + phenotype",
        "Phenotype only",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.0), sharey=True)
    axes = axes.flatten()
    for ax, (endpoint, metric, label, title, xlabel) in zip(axes, metric_panels):
        panel = data[data["endpoint"] == endpoint].copy()
        panel["model_label"] = pd.Categorical(panel["model_label"], categories=model_order, ordered=True)
        panel = panel.sort_values("model_label", ascending=False)
        y = np.arange(len(panel))
        colors = [color_map[str(item)] for item in panel["model_label"]]
        ax.scatter(panel[metric], y, s=42, color=colors, edgecolor="white", linewidth=0.35, zorder=3)
        ci_low = f"{metric}_ci_low"
        ci_high = f"{metric}_ci_high"
        if ci_low in panel.columns and ci_high in panel.columns:
            has_ci = panel[ci_low].notna() & panel[ci_high].notna()
            if has_ci.any():
                x = panel.loc[has_ci, metric].to_numpy()
                xerr = np.vstack(
                    [
                        x - panel.loc[has_ci, ci_low].to_numpy(),
                        panel.loc[has_ci, ci_high].to_numpy() - x,
                    ]
                )
                ax.errorbar(
                    x,
                    y[has_ci.to_numpy()],
                    xerr=xerr,
                    fmt="none",
                    ecolor="#222222",
                    elinewidth=0.9,
                    capsize=2.5,
                    zorder=2,
                )
        if metric == "auprc":
            baseline = float(panel["prevalence_test"].iloc[0])
            ax.axvline(baseline, color=GRAY, linestyle="--", linewidth=0.85, alpha=0.55)
            ax.text(
                baseline,
                4.35,
                "prevalence",
                rotation=90,
                fontsize=7,
                color="#555555",
                va="top",
                ha="right",
            )
        else:
            ax.axvline(1.0, color=GRAY, linestyle="--", linewidth=0.85, alpha=0.55)
            ax.text(
                1.0,
                4.35,
                "no enrichment",
                rotation=90,
                fontsize=7,
                color="#555555",
                va="top",
                ha="right",
            )
        panel_label(ax, label)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_yticks(y)
        ax.set_yticklabels(panel["model_label"].astype(str))
        clean_axes(ax, "x")
        ax.set_ylim(-0.6, len(panel) - 0.4)
        xmax = float(panel[metric].max()) * 1.18
        xmin = 0.0 if metric == "auprc" else 0.0
        ax.set_xlim(xmin, xmax)
    fig.tight_layout(rect=[0.03, 0.02, 1, 1])
    save_pub_figure(fig, output)
    plt.close(fig)
    return output

def build_figure3() -> Path:
    assign = pd.read_parquet(OBJECT_DIR / "ssl_phenotype_dev_assignments_200k.parquet")
    if len(assign) > 15000:
        assign = assign.sample(n=15000, random_state=20260525)
    selection = pd.read_csv(TABLE_DIR / "ssl_phenotype_cluster_selection_200k.csv")
    stability = pd.read_csv(TABLE_DIR / "ssl_phenotype_stability_200k.csv")
    rate_ci = pd.read_csv(TABLE_DIR / "ssl_phenotype_outcome_rate_bootstrap_ci_200k.csv")
    rates = pd.read_csv(TABLE_DIR / "ssl_phenotype_outcome_rates_200k.csv")
    profile = pd.read_csv(TABLE_DIR / "cns_phenotype_standardized_profiles_200k.csv")
    model_npz = np.load(OBJECT_DIR / "ssl_phenotype_model_200k.npz")
    pca_var = model_npz["explained_variance_ratio"]

    fig = plt.figure(figsize=(13.4, 9.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], wspace=0.35, hspace=0.48)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    ax = axes[0]
    panel_label(ax, "A", x=-0.18, y=1.16)
    colors = {0: LIGHT_BLUE, 1: ORANGE, 2: GRAY}
    for phenotype, group in assign.groupby("phenotype"):
        ax.scatter(group["pc1"], group["pc2"], s=3, alpha=0.35, color=colors[int(phenotype)], label=f"P{phenotype}")
    ax.set_title("2023 SSL embedding phenotypes")
    ax.set_xlabel(f"PC1 ({100 * pca_var[0]:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({100 * pca_var[1]:.1f}% var.)")
    ax.legend(frameon=False, markerscale=3, loc="best")

    ax = axes[1]
    panel_label(ax, "B")
    size_data = rates[rates["split"] == "test"].sort_values("phenotype")
    ax.bar(size_data["phenotype"].astype(str), size_data["n"], color=[colors[int(p)] for p in size_data["phenotype"]])
    ax.set_title("2024 assigned phenotype size")
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Records")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value / 1000)}k" if value > 0 else "0"))

    ax = axes[2]
    panel_label(ax, "C")
    diagnostics = [
        ("Silhouette", selection["silhouette"].to_numpy(dtype=float), "higher"),
        ("Min cluster\nproportion", selection["min_cluster_prop"].to_numpy(dtype=float), "higher"),
        ("Davies-Bouldin\nindex", selection["davies_bouldin"].to_numpy(dtype=float), "lower"),
    ]
    score_rows = []
    for _, values, direction in diagnostics:
        scaled = (values - values.min()) / np.maximum(values.max() - values.min(), 1e-9)
        if direction == "lower":
            scaled = 1 - scaled
        score_rows.append(scaled)
    score_matrix = np.vstack(score_rows)
    ax.imshow(score_matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.add_patch(patches.Rectangle((-0.5, -0.5), 1.0, len(diagnostics), fill=False, edgecolor=ORANGE, linewidth=1.8))
    ax.set_title("Cluster selection on 2023")
    ax.set_xlabel("Number of clusters")
    ax.set_xticks(np.arange(len(selection)))
    ax.set_xticklabels(selection["k"].astype(int))
    ax.set_yticks(np.arange(len(diagnostics)))
    ax.set_yticklabels([label for label, _, _ in diagnostics])
    for i, (label, values, _) in enumerate(diagnostics):
        for j, value in enumerate(values):
            if label.startswith("Min"):
                text = f"{100 * value:.1f}%"
            else:
                text = f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="white" if score_matrix[i, j] > 0.55 else TEXT)

    ax = axes[3]
    panel_label(ax, "D")
    stability_metrics = [
        ("ari_vs_primary", "ARI"),
        ("nmi_vs_primary", "NMI"),
        ("min_cluster_prop", "Min prop."),
    ]
    ypos = np.arange(len(stability_metrics))[::-1]
    for y, (col, label) in zip(ypos, stability_metrics):
        values = stability[col].to_numpy(dtype=float)
        median = float(np.median(values))
        ax.boxplot(
            values,
            vert=False,
            positions=[y],
            widths=0.34,
            patch_artist=True,
            boxprops=dict(facecolor=LIGHT_BLUE, edgecolor=BLUE, linewidth=0.8),
            medianprops=dict(color=DARK_BLUE, linewidth=1.2),
            whiskerprops=dict(color=BLUE, linewidth=0.8),
            capprops=dict(color=BLUE, linewidth=0.8),
            flierprops=dict(marker="o", markersize=2.0, markerfacecolor=BLUE, markeredgecolor="none", alpha=0.25),
        )
        label_x = min(median + 0.035, 1.04)
        ax.text(label_x, y, f"{median:.3f}", va="center", ha="left", fontsize=6.8, color=TEXT, clip_on=False)
    ax.set_yticks(ypos)
    ax.set_yticklabels([label for _, label in stability_metrics])
    ax.set_xlim(0, 1.16)
    ax.set_xlabel("Metric value")
    ax.set_title("Phenotype stability (80% resampling)")
    clean_axes(ax, "x")

    ax = axes[4]
    panel_label(ax, "E")
    endpoints = ["outcome_maternal_morbidity_core", "outcome_severe_neonatal_no_nicu"]
    offsets = [-0.08, 0.08]
    endpoint_colors = [BLUE, ORANGE]
    for endpoint, offset, color in zip(endpoints, offsets, endpoint_colors):
        panel = rate_ci[rate_ci["endpoint"] == endpoint].sort_values("phenotype")
        x = panel["phenotype"].to_numpy(dtype=float) + offset
        y = panel["event_rate"].to_numpy() * 100
        yerr = np.vstack([y - panel["ci_low"].to_numpy() * 100, panel["ci_high"].to_numpy() * 100 - y])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.35,
            capsize=2.5,
            elinewidth=0.9,
            label=ENDPOINT_LABELS[endpoint],
        )
        baseline = panel["events"].sum() / panel["n"].sum() * 100
        ax.axhline(baseline, color=color, linestyle=":", linewidth=0.9, alpha=0.65)
        ax.text(2.19, baseline, "baseline", color=color, fontsize=5.8, va="center", ha="left")
        for xi, yi in zip(x, y):
            y_offset = 0.38 if yi > 5 else 0.18
            ax.text(
                xi,
                yi + y_offset,
                f"{yi:.1f}%",
                fontsize=5.8,
                ha="center",
                color=color,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.78),
            )
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("2024 event rate, %")
    ax.set_title("Outcome enrichment by phenotype")
    ax.legend(frameon=False, fontsize=6.6, loc="upper left", handletextpad=0.35)
    clean_axes(ax, "y")

    ax = axes[5]
    panel_label(ax, "F")
    feature_order = [
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
    pivot = profile.pivot(index="phenotype", columns="feature", values="standardized_difference").loc[[0, 1, 2], feature_order]
    matrix = pivot.to_numpy(dtype=float)
    lim = max(0.75, float(np.nanmax(np.abs(matrix))))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_yticks(np.arange(len(pivot)))
    ax.set_yticklabels([f"P{int(p)}" for p in pivot.index])
    ax.set_xticks(np.arange(len(feature_order)))
    ax.set_xticklabels(feature_order, rotation=32, ha="right", rotation_mode="anchor")
    ax.set_title("Standardized phenotype profile")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Standardized difference vs overall", fontsize=6.8)

    output = FIGURE_DIR / "Figure_3_ssl_phenotype_discovery.png"
    save_pub_figure(fig, output)
    plt.close(fig)
    return output


def plot_ci_points(ax, data: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    feature_order = ["phenotype", "ssl_embedding", "ssl_plus_phenotype"]
    offsets = [-0.12, 0.12]
    endpoint_colors = [BLUE, ORANGE]
    for endpoint, offset, color in zip(ENDPOINT_LABELS, offsets, endpoint_colors):
        panel = data[(data["endpoint"] == endpoint) & (data["metric"] == metric)].set_index("feature_set")
        panel = panel.loc[feature_order].reset_index()
        x = np.arange(len(panel)) + offset
        y = panel["point"].to_numpy()
        yerr = np.vstack([y - panel["ci_low"].to_numpy(), panel["ci_high"].to_numpy() - y])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.35,
            capsize=2.5,
            elinewidth=0.9,
            label=ENDPOINT_LABELS[endpoint],
        )
    ax.set_xticks(np.arange(len(feature_order)))
    ax.set_xticklabels([FEATURE_LABELS[item] for item in feature_order], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=6.7, loc="best", handletextpad=0.35)
    clean_axes(ax, "y")


def build_figure4() -> Path:
    ci = pd.read_csv(TABLE_DIR / "ssl_phenotype_risk_metric_bootstrap_ci_200k.csv")
    calibration = pd.read_csv(TABLE_DIR / "ssl_phenotype_calibration_bins_200k.csv")
    pca = pd.read_csv(TABLE_DIR / "ssl_pca_sensitivity_200k.csv")
    metrics = pd.read_csv(TABLE_DIR / "ssl_phenotype_risk_metrics_200k.csv")

    fig = plt.figure(figsize=(13.2, 8.8))
    gs = fig.add_gridspec(2, 3, wspace=0.35, hspace=0.5)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    ax = axes[0]
    panel_label(ax, "A")
    plot_ci_points(ax, ci, "auprc_over_prevalence", "Platt-calibrated AUPRC enrichment", "AUPRC / prevalence")

    ax = axes[1]
    panel_label(ax, "B")
    plot_ci_points(ax, ci, "top1_enrichment_over_prevalence", "Top 1% enrichment", "Enrichment over prevalence")

    for ax, endpoint, label in [
        (axes[2], "outcome_maternal_morbidity_core", "C"),
        (axes[3], "outcome_severe_neonatal_no_nicu", "D"),
    ]:
        panel_label(ax, label)
        panel = calibration[
            (calibration["endpoint"] == endpoint)
            & (calibration["feature_set"] == "ssl_plus_phenotype")
            & (calibration["probability_type"] == "platt")
        ].sort_values("bin")
        ax.plot(panel["mean_pred"], panel["event_rate"], marker="o", color=ORANGE, lw=1.5)
        max_v = max(panel["mean_pred"].max(), panel["event_rate"].max()) * 1.08
        ax.plot([0, max_v], [0, max_v], color=GRAY, lw=1.0, linestyle="--")
        metric_row = metrics[
            (metrics["endpoint"] == endpoint)
            & (metrics["feature_set"] == "ssl_plus_phenotype")
            & (metrics["probability_type"] == "platt")
        ].iloc[0]
        ax.text(
            0.04,
            0.93,
            f"ECE = {metric_row['ece_10']:.4f}\nBrier = {metric_row['brier']:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=LIGHT_GRAY),
        )
        ax.set_xlim(0, max_v)
        ax.set_ylim(0, max_v)
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Observed event rate")
        ax.set_title(f"Calibration: {ENDPOINT_LABELS[endpoint]}")
        clean_axes(ax, "both")

    for ax, metric, title, ylabel, label in [
        (axes[4], "auprc", "PCA sensitivity: AUPRC", "AUPRC", "E"),
        (axes[5], "top1_enrichment_over_prevalence", "PCA sensitivity: top 1% enrichment", "Enrichment over prevalence", "F"),
    ]:
        panel_label(ax, label)
        for endpoint, color in zip(ENDPOINT_LABELS, [BLUE, ORANGE]):
            panel = pca[pca["endpoint"] == endpoint].sort_values("feature_order")
            ax.plot(panel["feature_set"], panel[metric], marker="o", color=color, label=ENDPOINT_LABELS[endpoint])
            for _, row in panel.iterrows():
                if str(row["feature_set"]).startswith("PCA 20"):
                    ax.annotate(
                        f"{row['pca_explained_variance_total']:.1%} var.",
                        (row["feature_set"], row[metric]),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        fontsize=7,
                        color=color,
                    )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(frameon=False, fontsize=7, loc="best")
        clean_axes(ax, "y")

    output = FIGURE_DIR / "Figure_4_calibration_sensitivity.png"
    save_pub_figure(fig, output)
    plt.close(fig)
    return output


def _plot_topk_event_rate(ax, topk: pd.DataFrame, endpoint: str, label: str) -> None:
    panel_label(ax, label)
    panel = topk[topk["endpoint"] == endpoint].sort_values(["model", "top_fraction"])
    for model, model_panel in panel.groupby("model"):
        color = MODEL_COLORS.get(model, GRAY)
        ax.plot(
            model_panel["top_fraction"] * 100,
            model_panel["event_rate"] * 100,
            marker="o",
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.3,
            lw=1.35,
            color=color,
            label=model,
        )
    prevalence = float((panel["event_rate"] / panel["enrichment_over_prevalence"]).dropna().iloc[0])
    ax.axhline(prevalence * 100, color=GRAY, lw=1.0, linestyle="--", label="Overall prevalence")
    ax.set_xlabel("Highest-risk records evaluated (%)")
    ax.set_ylabel("Observed event rate (%)")
    ax.set_title(f"Top-risk enrichment: {ENDPOINT_LABELS[endpoint]}")
    ax.set_xticks([0.5, 1, 2, 5])
    clean_axes(ax, "y")
    ax.legend(frameon=False, fontsize=6.7, loc="best", handlelength=1.6, handletextpad=0.35)


def _plot_decision_curve(ax, dca: pd.DataFrame, endpoint: str, label: str) -> None:
    panel_label(ax, label)
    panel = dca[dca["endpoint"] == endpoint].sort_values(["model", "threshold"])
    for model, model_panel in panel.groupby("model"):
        color = MODEL_COLORS.get(model, GRAY)
        ax.plot(
            model_panel["threshold"] * 100,
            model_panel["net_benefit"],
            marker="o",
            ms=3.4,
            markeredgecolor="white",
            markeredgewidth=0.25,
            lw=1.25,
            color=color,
            label=model,
        )
    baseline_panel = panel.drop_duplicates("threshold").sort_values("threshold")
    ax.plot(
        baseline_panel["threshold"] * 100,
        baseline_panel["treat_all_net_benefit"],
        color=GRAY,
        lw=1.0,
        linestyle="--",
        label="Treat all",
    )
    ax.axhline(0, color="#AAAAAA", lw=0.9, linestyle=":", label="Treat none")
    ax.set_xlabel("Risk threshold (%)")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"Decision-curve analysis: {ENDPOINT_LABELS[endpoint]}")
    clean_axes(ax, "y")
    ax.legend(frameon=False, fontsize=6.7, loc="best", handlelength=1.6, handletextpad=0.35)


def _clean_feature_name(name: str) -> str:
    name = str(name).replace("input_", "")
    replacements = {
        "MAGER": "Maternal age",
        "MAGER9": "Maternal age group",
        "RF_CESAR": "Prior cesarean",
        "PREVIS": "Prenatal visits",
        "PREVIS_REC": "Prenatal visits category",
        "WTGAIN": "Gestational weight gain",
        "ILLB_R11": "Interpregnancy interval",
        "TBO_REC": "Total birth order",
        "LBO_REC": "Live birth order",
        "CIG_0": "Prepregnancy smoking",
        "DLMP_MM": "LMP month",
        "DOB_MM": "Birth month",
    }
    return replacements.get(name, name.replace("_", " "))


def build_figure5() -> Path:
    topk = pd.read_csv(TABLE_DIR / "cns_topk_utility_200k.csv")
    dca = pd.read_csv(TABLE_DIR / "cns_decision_curve_200k.csv")
    family = pd.read_csv(TABLE_DIR / "cns_feature_family_importance_200k.csv")
    feature = pd.read_csv(TABLE_DIR / "cns_lightgbm_gain_importance_200k.csv")
    subgroup = pd.read_csv(TABLE_DIR / "cns_subgroup_metrics_200k.csv")

    fig = plt.figure(figsize=(13.4, 11.0))
    gs = fig.add_gridspec(3, 2, wspace=0.36, hspace=0.58)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]

    _plot_topk_event_rate(axes[0], topk, "outcome_severe_neonatal_no_nicu", "A")
    _plot_topk_event_rate(axes[1], topk, "outcome_maternal_morbidity_core", "B")
    _plot_decision_curve(axes[2], dca, "outcome_severe_neonatal_no_nicu", "C")
    _plot_decision_curve(axes[3], dca, "outcome_maternal_morbidity_core", "D")

    ax = axes[4]
    panel_label(ax, "E")
    order = (
        family.groupby("feature_family")["gain_fraction"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    y = np.arange(len(order))
    offset = 0.16
    for endpoint, offset, color in [
        ("outcome_maternal_morbidity_core", -offset, BLUE),
        ("outcome_severe_neonatal_no_nicu", offset, ORANGE),
    ]:
        panel = family[family["endpoint"] == endpoint].set_index("feature_family").reindex(order)
        values = panel["gain_fraction"].to_numpy(dtype=float) * 100
        ax.hlines(y + offset, 0, values, color=color, linewidth=1.4, alpha=0.88)
        ax.scatter(values, y + offset, s=24, color=color, edgecolor="white", linewidth=0.3, zorder=3, label=ENDPOINT_LABELS[endpoint])
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("LightGBM gain contribution (%)")
    ax.set_title("Feature-family contribution")
    clean_axes(ax, "x")
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    ax = axes[5]
    panel_label(ax, "F")
    filtered = subgroup[
        (subgroup["model"] == "SSL + phenotype")
        & (subgroup["n"] >= 1000)
        & (subgroup["events"] >= 10)
    ].copy()
    wanted = [
        "age_group",
        "bmi_group",
        "diabetes",
        "hypertensive_disorder",
        "infertility_art",
        "plurality",
        "prior_cesarean",
    ]
    endpoint_order = ["outcome_maternal_morbidity_core", "outcome_severe_neonatal_no_nicu"]
    heat = (
        filtered[filtered["subgroup_variable"].isin(wanted)]
        .groupby(["subgroup_variable", "endpoint"])["auprc_over_prevalence"]
        .min()
        .unstack("endpoint")
        .reindex(wanted)
        .reindex(columns=endpoint_order)
    )
    display_rows = [
        "Age",
        "BMI",
        "Diabetes",
        "Hypertensive disorder",
        "Infertility/ART",
        "Plurality",
        "Prior cesarean",
    ]
    matrix = heat.to_numpy(dtype=float)
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=1.0)
    ax.set_xticks(np.arange(len(endpoint_order)))
    ax.set_xticklabels([ENDPOINT_LABELS[item].replace(" outcome", "") for item in endpoint_order], rotation=15, ha="right")
    ax.set_yticks(np.arange(len(display_rows)))
    ax.set_yticklabels(display_rows)
    ax.set_title("Worst-stratum AUPRC enrichment")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                text_color = "white" if value >= 3.5 else TEXT
                ax.text(j, i, f"{value:.1f}x", ha="center", va="center", fontsize=8, color=text_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Minimum AUPRC / prevalence", fontsize=8)

    output = FIGURE_DIR / "Figure_5_clinical_utility_interpretability.png"
    save_pub_figure(fig, output)
    plt.close(fig)
    return output


def write_report(paths: list[Path]) -> Path:
    report = DOCS_DIR / "21_final_figure_build_report.md"
    lines = [
        "# Final Manuscript Figure Build Report",
        "",
        "Generated final manuscript-style figures from source-data tables and existing matched analyses.",
        "",
        "## Figures",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Figure 2 uses the matched 200,000-record supervised and SSL comparison.",
            "- Figure 3 emphasizes SSL phenotype discovery, stability, outcome enrichment, and standardized prototype profiles.",
            "- Figure 4 emphasizes calibration, top-risk enrichment, PCA dimensionality sensitivity, and explicit ECE/Brier annotations.",
            "- Figure 5 adds reviewer-facing clinical-utility, decision-curve, feature-family, and subgroup-robustness evidence.",
            "- Figure 1 explicitly states the 2016-2022 / 2023 / 2024 temporal design and prevents leakage overclaiming.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    setup_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        build_figure1(),
        build_figure2(),
        build_figure3(),
        build_figure4(),
        build_figure5(),
    ]
    report = write_report(paths)
    for path in paths:
        print(f"wrote {path}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
