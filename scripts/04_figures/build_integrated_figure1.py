#!/usr/bin/env python
"""Build an integrated publication-style Figure 1.

This replaces the earlier image-panel collage with a single vector-like
matplotlib drawing. The goal is to keep the same A-D scientific content while
using one typographic system, one coordinate space, and a restrained palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
SUBMISSION_FIG_DIR = PROJECT_ROOT / "submission_package" / "figures"
SN_LATEX_FIG_DIR = PROJECT_ROOT / "submission_package" / "springer_nature_latex" / "figures"
BMC_FIG_DIR = PROJECT_ROOT / "submission_package" / "bmc_midd_latex_upload" / "figures"

TEXT = "#1F2933"
MUTED = "#5D6773"
GRID = "#D9DEE7"
PANEL_EDGE = "#C7CED9"
BLUE = "#255C99"
BLUE_FILL = "#EEF5FC"
ORANGE = "#E87428"
ORANGE_FILL = "#FFF3E8"
GREEN = "#3E8E57"
GREEN_FILL = "#EFF8F1"
PURPLE = "#8C5AA5"
PURPLE_FILL = "#F5EFF8"
GRAY_FILL = "#F6F7F9"
LOSS = "#C44E52"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.0,
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    edge: str = PANEL_EDGE,
    face: str = "white",
    color: str = TEXT,
    fontsize: float = 7.4,
    weight: str = "normal",
    radius: float = 0.010,
    lw: float = 0.9,
    linespacing: float = 1.16,
) -> patches.FancyBboxPatch:
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        linespacing=linespacing,
    )
    return patch


def label(ax: plt.Axes, letter: str, title: str, x: float, y: float) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y - 0.018),
            0.026,
            0.026,
            boxstyle="round,pad=0.002,rounding_size=0.004",
            facecolor=TEXT,
            edgecolor=TEXT,
            linewidth=0,
        )
    )
    ax.text(x + 0.013, y - 0.005, letter, ha="center", va="center", fontsize=9.0, color="white", fontweight="bold")
    ax.text(x + 0.034, y - 0.004, title, ha="left", va="center", fontsize=10.2, color=TEXT, fontweight="bold")


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = MUTED, lw: float = 1.0) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=0, shrinkB=0, mutation_scale=9),
    )


def connector(ax: plt.Axes, xs: list[float], ys: list[float], *, color: str = MUTED, lw: float = 0.9) -> None:
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round")


def section_band(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.014",
            facecolor="white",
            edgecolor=PANEL_EDGE,
            linewidth=0.75,
        )
    )


def draw_matrix(ax: plt.Axes, x: float, y: float, rows: int, cols: int, cell: float, gap: float, colors: list[str]) -> None:
    for r in range(rows):
        for c in range(cols):
            face = colors[(r + c) % len(colors)]
            ax.add_patch(
                patches.Rectangle(
                    (x + c * (cell + gap), y - r * (cell + gap)),
                    cell,
                    cell,
                    facecolor=face,
                    edgecolor="white",
                    linewidth=0.35,
                )
            )


def draw_tokens(ax: plt.Axes, x: float, y: float, *, masked: bool = False) -> None:
    cell_w = 0.010
    cell_h = 0.022
    gap_x = 0.006
    gap_y = 0.010
    masked_cells = {(0, 2), (1, 4), (2, 1), (2, 5)} if masked else set()
    for r in range(3):
        for c in range(6):
            face = BLUE_FILL if c < 4 else GRAY_FILL
            edge = BLUE if c < 4 else "#AEB7C2"
            hatch = None
            if (r, c) in masked_cells:
                face = ORANGE_FILL
                edge = ORANGE
                hatch = "////"
            ax.add_patch(
                patches.FancyBboxPatch(
                    (x + c * (cell_w + gap_x), y - r * (cell_h + gap_y)),
                    cell_w,
                    cell_h,
                    boxstyle="round,pad=0.001,rounding_size=0.002",
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.45,
                    hatch=hatch,
                )
            )


def draw_transformer(ax: plt.Axes, x: float, y: float) -> None:
    box(ax, x, y, 0.078, 0.120, "", edge=BLUE, face=BLUE_FILL, radius=0.009, lw=0.9)
    for yy in [y + 0.074, y + 0.028]:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x + 0.014, yy),
                0.050,
                0.030,
                boxstyle="round,pad=0.003,rounding_size=0.006",
                facecolor="white",
                edgecolor=BLUE,
                linewidth=0.7,
            )
        )
        pts = [(x + 0.026, yy + 0.010), (x + 0.039, yy + 0.021), (x + 0.053, yy + 0.010)]
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color=BLUE, lw=0.45)
        for px, py in pts:
            ax.add_patch(patches.Circle((px, py), 0.0042, facecolor=BLUE_FILL, edgecolor=BLUE, linewidth=0.5))


def draw_embedding(ax: plt.Axes, x: float, y: float) -> None:
    box(ax, x, y, 0.034, 0.120, "", edge=BLUE, face="white", radius=0.007, lw=0.8)
    for i in range(5):
        ax.add_patch(patches.Circle((x + 0.017, y + 0.098 - i * 0.021), 0.006, facecolor=BLUE, edgecolor=BLUE, linewidth=0))


def build_figure() -> Path:
    setup_style()
    fig = plt.figure(figsize=(15.8, 7.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Main regions use very light borders only; no shadows, no pasted panels.
    section_band(ax, 0.030, 0.695, 0.940, 0.250)
    section_band(ax, 0.030, 0.390, 0.255, 0.255)
    section_band(ax, 0.315, 0.390, 0.360, 0.255)
    section_band(ax, 0.705, 0.390, 0.265, 0.255)
    section_band(ax, 0.030, 0.090, 0.940, 0.165)

    label(ax, "A", "Calendar-time cohort spine", 0.050, 0.925)
    label(ax, "B", "Leakage-controlled registry matrix", 0.050, 0.620)
    label(ax, "C", "Masked tabular SSL encoder", 0.335, 0.620)
    label(ax, "D", "Phenotypes, risk enrichment, and transfer tests", 0.725, 0.620)

    # Panel A: data source and time split.
    box(ax, 0.060, 0.760, 0.115, 0.085, "CDC/NCHS\nNatality records", edge=BLUE, face="white", fontsize=7.1, weight="bold")
    for x in [0.195, 0.330, 0.465]:
        ax.plot([x, x], [0.760, 0.845], color=GRID, lw=0.8)
    timeline = [
        (0.200, "2016-2022", "Train + SSL pretrain\n26.35M records", BLUE, BLUE_FILL, 0.170),
        (0.400, "2023", "Development +\ncalibration\n3.61M records", ORANGE, ORANGE_FILL, 0.155),
        (0.585, "2024", "Temporal test\n3.64M records", "#6B7280", GRAY_FILL, 0.140),
    ]
    for x, year, desc, edge, face, w in timeline:
        box(ax, x, 0.755, w, 0.095, f"{year}\n{desc}", edge=edge, face=face, fontsize=7.3, weight="bold", linespacing=1.13)
    arrow(ax, (0.175, 0.802), (0.200, 0.802), color=MUTED, lw=0.9)
    arrow(ax, (0.370, 0.802), (0.400, 0.802), color=MUTED, lw=0.9)
    arrow(ax, (0.555, 0.802), (0.585, 0.802), color=MUTED, lw=0.9)
    ax.text(0.765, 0.866, "Transfer and registry stress-tests", ha="left", va="center", fontsize=7.4, color=MUTED, fontweight="bold")
    box(ax, 0.765, 0.800, 0.165, 0.050, "Linked birth/infant death\nsevere-endpoint transfer", edge=PURPLE, face=PURPLE_FILL, fontsize=6.8)
    box(ax, 0.765, 0.730, 0.165, 0.050, "Brazil SINASC 2023-2024\nregistry stress-test", edge=GREEN, face=GREEN_FILL, fontsize=6.8)
    arrow(ax, (0.725, 0.802), (0.765, 0.825), color=PURPLE, lw=0.75)
    arrow(ax, (0.725, 0.782), (0.765, 0.755), color=GREEN, lw=0.75)
    ax.text(
        0.060,
        0.710,
        "Temporal hold-out protects final-year evaluation; 2024 and transfer labels are used only after model development.",
        ha="left",
        va="center",
        fontsize=7.1,
        color=MUTED,
    )

    # Panel B: matrix and domains.
    ax.text(0.055, 0.545, "Input domains", ha="left", va="center", fontsize=8.1, fontweight="bold", color=TEXT)
    domains = [
        ("Demographics", "Anthropometry"),
        ("Prenatal care", "Diabetes / hypertension"),
        ("Smoking / infections", "Infertility / ART"),
        ("Obstetric history", "Missingness indicators"),
    ]
    for i, (left, right) in enumerate(domains):
        y = 0.512 - i * 0.031
        ax.add_patch(patches.Circle((0.060, y + 0.002), 0.0035, facecolor=BLUE if i < 2 else ORANGE, edgecolor="none"))
        ax.text(0.068, y, left, ha="left", va="center", fontsize=6.15, color=TEXT)
        ax.add_patch(patches.Circle((0.150, y + 0.002), 0.0035, facecolor=BLUE if i < 2 else ORANGE, edgecolor="none"))
        ax.text(0.158, y, right, ha="left", va="center", fontsize=6.15, color=TEXT)
    ax.text(0.210, 0.550, "112 registry variables", ha="left", va="bottom", fontsize=6.45, color=MUTED, fontweight="bold")
    draw_matrix(ax, 0.212, 0.532, rows=9, cols=9, cell=0.0062, gap=0.0020, colors=[BLUE_FILL, "#DCEAF8", ORANGE_FILL, "#F4F5F7"])
    ax.text(0.055, 0.403, "No outcome components in inputs", ha="left", va="center", fontsize=6.45, color=ORANGE, fontweight="bold")

    # Panel C: encoder.
    c_y = 0.500
    ax.text(0.342, 0.545, "Feature tokens", ha="center", va="center", fontsize=7.0, color=TEXT, fontweight="bold")
    draw_tokens(ax, 0.326, 0.510, masked=False)
    arrow(ax, (0.430, c_y), (0.448, c_y), color=MUTED)
    ax.text(0.476, 0.545, "Mixed masking", ha="center", va="center", fontsize=7.0, color=TEXT, fontweight="bold")
    draw_tokens(ax, 0.452, 0.510, masked=True)
    arrow(ax, (0.556, c_y), (0.578, c_y), color=MUTED)
    ax.text(0.617, 0.545, "2-layer transformer", ha="center", va="center", fontsize=7.0, color=TEXT, fontweight="bold")
    draw_transformer(ax, 0.578, 0.440)
    arrow(ax, (0.656, c_y), (0.666, c_y), color=MUTED)
    draw_embedding(ax, 0.668, 0.440)
    ax.text(0.686, 0.545, "48-d\nembedding", ha="center", va="center", fontsize=6.7, color=TEXT, fontweight="bold")
    box(ax, 0.430, 0.405, 0.085, 0.040, "categorical\nreconstruction", edge=BLUE, face="white", fontsize=5.75)
    box(ax, 0.526, 0.405, 0.078, 0.040, "numeric\nreconstruction", edge=LOSS, face="white", fontsize=5.75)
    ax.text(0.342, 0.412, "Mask rate 0.35", ha="left", va="center", fontsize=6.5, color=MUTED)
    ax.text(0.612, 0.412, "Pretrained on 2016-2022 only", ha="left", va="center", fontsize=6.5, color=MUTED)

    # Panel D: downstream and transfer.
    box(ax, 0.724, 0.520, 0.060, 0.046, "2023\nembeddings", edge=BLUE, face=BLUE_FILL, fontsize=5.8, weight="bold")
    box(ax, 0.805, 0.520, 0.060, 0.046, "PCA +\nk-means", edge=BLUE, face=BLUE_FILL, fontsize=5.8, weight="bold")
    box(ax, 0.886, 0.520, 0.060, 0.046, "fixed\ncentroids", edge=BLUE, face=BLUE_FILL, fontsize=5.8, weight="bold")
    arrow(ax, (0.784, 0.543), (0.805, 0.543), color=MUTED, lw=0.82)
    arrow(ax, (0.865, 0.543), (0.886, 0.543), color=MUTED, lw=0.82)
    box(ax, 0.724, 0.456, 0.060, 0.046, "2024\nembeddings", edge="#6B7280", face=GRAY_FILL, fontsize=5.8, weight="bold")
    box(ax, 0.805, 0.456, 0.060, 0.046, "phenotype\nassignment", edge=ORANGE, face=ORANGE_FILL, fontsize=5.55, weight="bold")
    box(ax, 0.886, 0.456, 0.060, 0.046, "Platt risk\n+ metrics", edge=ORANGE, face=ORANGE_FILL, fontsize=5.8, weight="bold")
    arrow(ax, (0.784, 0.479), (0.805, 0.479), color=MUTED, lw=0.82)
    arrow(ax, (0.865, 0.479), (0.886, 0.479), color=MUTED, lw=0.82)
    connector(ax, [0.916, 0.916, 0.835], [0.520, 0.506, 0.502], color=BLUE, lw=0.75)
    box(ax, 0.724, 0.405, 0.095, 0.027, "linked mortality", edge=PURPLE, face=PURPLE_FILL, fontsize=5.4)
    box(ax, 0.830, 0.405, 0.095, 0.027, "SINASC stress-test", edge=GREEN, face=GREEN_FILL, fontsize=5.4)

    # Bottom legend: image2-style visual key, with exact manuscript-safe text.
    legend_items = [
        (BLUE, BLUE_FILL, "Training / pretraining\n2016-2022"),
        (ORANGE, ORANGE_FILL, "Development /\ncalibration 2023"),
        ("#6B7280", GRAY_FILL, "Temporal test\n2024"),
        (PURPLE, PURPLE_FILL, "Transfer endpoint\nlinked infant death"),
        (GREEN, GREEN_FILL, "Registry stress-test\nSINASC"),
        ("#6B7280", "white", "Registry-defined\ninputs only"),
        (TEXT, "white", "Risk enrichment,\nnot bedside diagnosis"),
    ]
    x0 = 0.060
    for edge, face, txt in legend_items:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x0, 0.168),
                0.024,
                0.024,
                boxstyle="round,pad=0.001,rounding_size=0.003",
                facecolor=face,
                edgecolor=edge,
                linewidth=0.9,
            )
        )
        ax.text(x0 + 0.032, 0.181, txt, ha="left", va="center", fontsize=6.3, color=TEXT, linespacing=1.15)
        x0 += 0.129
    ax.text(
        0.060,
        0.121,
        "All model-development steps precede final-year and transfer-endpoint evaluation; outputs are interpreted as calibrated risk-enrichment signals.",
        ha="left",
        va="center",
        fontsize=6.8,
        color=MUTED,
    )
    ax.set_ylim(0.060, 0.965)

    out = FIGURE_DIR / "Figure_1_study_design_ssl_pipeline.png"
    for target in [FIGURE_DIR, SUBMISSION_FIG_DIR, SN_LATEX_FIG_DIR, BMC_FIG_DIR]:
        target.mkdir(parents=True, exist_ok=True)
        fig.savefig(target / out.name, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(target / out.with_suffix(".pdf").name, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(target / out.with_suffix(".svg").name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def main() -> None:
    output = build_figure()
    print(output)


if __name__ == "__main__":
    main()
