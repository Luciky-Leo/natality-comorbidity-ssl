#!/usr/bin/env python
"""Build synchronized manuscript Figure 1 from exact vector-style panels.

The script first creates four content panels without A/B/C/D labels, then
composes them into the final labelled Figure 1 with uniform typography.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
SOURCE_DIR = FIGURE_DIR / "source_assets"
LATEX_FIG_DIR = PROJECT_ROOT / "submission_package" / "springer_nature_latex" / "figures"
SUBMISSION_FIG_DIR = PROJECT_ROOT / "submission_package" / "figures"

BLUE = "#4E79A7"
DARK_BLUE = "#1F5A96"
LIGHT_BLUE = "#EEF4FB"
ORANGE = "#F28E2B"
LIGHT_ORANGE = "#FFF4EA"
GREEN = "#59A14F"
LIGHT_GREEN = "#EEF7ED"
PURPLE = "#B07AA1"
LIGHT_PURPLE = "#F6F0F6"
GRAY = "#7A7A7A"
LIGHT_GRAY = "#F5F5F5"
DARK = "#202020"
GRID = "#E6E6E6"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.0,
            "axes.titlesize": 10.2,
            "axes.labelsize": 8.0,
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_panel(fig: plt.Figure, name: str) -> Path:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    path = SOURCE_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(SOURCE_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(SOURCE_DIR / f"{name}.svg", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    edge: str,
    face: str,
    fontsize: float = 8.4,
    weight: str = "normal",
) -> patches.FancyBboxPatch:
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.05,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=DARK,
        linespacing=1.18,
    )
    return box


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = GRAY, lw: float = 1.25) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=0, shrinkB=0))


def panel_title(ax: plt.Axes, title: str) -> None:
    ax.text(0.03, 0.96, title, transform=ax.transAxes, ha="left", va="top", fontsize=10.8, fontweight="bold", color=DARK)


def draw_database_icon(ax: plt.Axes, cx: float, cy: float, width: float = 0.09, height: float = 0.12, color: str = DARK_BLUE) -> None:
    ax.add_patch(patches.Ellipse((cx, cy + height / 2), width, height * 0.28, facecolor=LIGHT_BLUE, edgecolor=color, lw=1.0))
    ax.add_patch(patches.Rectangle((cx - width / 2, cy - height / 2), width, height, facecolor=LIGHT_BLUE, edgecolor=color, lw=1.0))
    ax.add_patch(patches.Ellipse((cx, cy - height / 2), width, height * 0.28, facecolor="white", edgecolor=color, lw=1.0))
    for offset in [-0.02, 0.02]:
        ax.plot([cx - width / 2, cx + width / 2], [cy + offset, cy + offset], color=color, lw=0.7, alpha=0.75)


def draw_panel_a() -> Path:
    fig, ax = plt.subplots(figsize=(6.3, 3.25))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_title(ax, "Cohort and temporal split")

    ax.add_patch(patches.FancyBboxPatch((0.04, 0.43), 0.10, 0.22, boxstyle="round,pad=0.012", edgecolor=DARK_BLUE, facecolor="white", lw=1.15))
    ax.text(0.09, 0.55, "CDC", ha="center", va="center", color=DARK_BLUE, fontsize=9.2, fontweight="bold")
    ax.text(0.09, 0.37, "CDC/NCHS\npublic-use\nrecords", ha="center", va="top", fontsize=6.6, color=DARK)

    rounded_box(ax, (0.18, 0.46), 0.23, 0.18, "2016--2022\nTrain / SSL pretrain\n26.35M records", DARK_BLUE, LIGHT_BLUE, 7.15, "bold")
    rounded_box(ax, (0.49, 0.46), 0.20, 0.18, "2023\nDevelopment /\ncalibration\n3.61M records", ORANGE, LIGHT_ORANGE, 6.9, "bold")
    rounded_box(ax, (0.78, 0.46), 0.18, 0.18, "2024\nTemporal test\n3.64M records", GRAY, LIGHT_GRAY, 7.15, "bold")
    arrow(ax, (0.41, 0.55), (0.49, 0.55))
    arrow(ax, (0.69, 0.55), (0.78, 0.55))
    rounded_box(ax, (0.22, 0.08), 0.24, 0.12, "Linked birth/\ninfant death file", PURPLE, LIGHT_PURPLE, 6.9)
    rounded_box(ax, (0.56, 0.08), 0.25, 0.12, "SINASC 2023--2024\nregistry stress-test", GREEN, LIGHT_GREEN, 6.9)
    arrow(ax, (0.30, 0.46), (0.34, 0.21), PURPLE, 1.0)
    arrow(ax, (0.59, 0.46), (0.68, 0.21), GREEN, 1.0)
    return save_panel(fig, "Figure_1A_content_no_label")


def draw_panel_b() -> Path:
    fig, ax = plt.subplots(figsize=(6.3, 3.25))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_title(ax, "Leakage-controlled registry inputs")

    draw_database_icon(ax, 0.50, 0.53, width=0.12, height=0.18)
    ax.text(0.50, 0.37, "Analytic\nvariables", ha="center", va="center", fontsize=7.7, color=DARK, fontweight="bold")
    items = [
        ("Demographics", 0.09, 0.72),
        ("Prenatal care", 0.09, 0.54),
        ("Smoking /\ninfections", 0.09, 0.36),
        ("Obstetric history", 0.09, 0.18),
        ("Anthropometry", 0.68, 0.72),
        ("Diabetes /\nhypertension", 0.68, 0.54),
        ("Infertility / ART", 0.68, 0.36),
        ("Missingness", 0.68, 0.18),
    ]
    for text, x, y in items:
        rounded_box(ax, (x, y), 0.23, 0.11, text, "#BDBDBD", "white", 7.6)
        side_x = x + 0.23 if x < 0.5 else x
        arrow_end_x = 0.44 if x < 0.5 else 0.56
        arrow(ax, (side_x, y + 0.055), (arrow_end_x, 0.53), "#9B9B9B", 0.75)
    ax.text(0.50, 0.045, "U.S. Natality: 44 categorical + 68 numeric/missingness variables", ha="center", va="center", fontsize=7.6, color=DARK)
    return save_panel(fig, "Figure_1B_content_no_label")


def draw_tokens(ax: plt.Axes, x: float, y: float, rows: int = 3, cols: int = 6, masked: set[tuple[int, int]] | None = None) -> None:
    masked = masked or set()
    w, h, gap = 0.018, 0.045, 0.010
    for r in range(rows):
        for c in range(cols):
            xx = x + c * (w + gap)
            yy = y - r * (h + gap)
            is_masked = (r, c) in masked
            face = LIGHT_ORANGE if is_masked else LIGHT_BLUE if c < 4 else LIGHT_GRAY
            edge = ORANGE if is_masked else BLUE if c < 4 else "#AAAAAA"
            hatch = "////" if is_masked else None
            ax.add_patch(patches.FancyBboxPatch((xx, yy), w, h, boxstyle="round,pad=0.002", facecolor=face, edgecolor=edge, lw=0.55, hatch=hatch))


def draw_transformer(ax: plt.Axes, x: float, y: float) -> None:
    rounded_box(ax, (x, y), 0.12, 0.24, "", DARK_BLUE, LIGHT_BLUE, 7)
    for layer_y in [y + 0.16, y + 0.07]:
        ax.add_patch(patches.FancyBboxPatch((x + 0.024, layer_y), 0.072, 0.055, boxstyle="round,pad=0.006", facecolor="white", edgecolor=BLUE, lw=0.8))
        nodes = [(x + 0.04, layer_y + 0.018), (x + 0.06, layer_y + 0.038), (x + 0.08, layer_y + 0.018)]
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color=BLUE, lw=0.55)
        for nx, ny in nodes:
            ax.add_patch(patches.Circle((nx, ny), 0.007, facecolor=LIGHT_BLUE, edgecolor=BLUE, lw=0.55))


def draw_panel_c() -> Path:
    fig, ax = plt.subplots(figsize=(6.3, 3.25))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_title(ax, "Masked tabular SSL encoder")

    xs = [0.08, 0.31, 0.52, 0.71, 0.88]
    labels = ["Categorical +\nnumeric tokens", "Feature\nembeddings", "Mixed\nmasking", "2-layer\ntransformer", "48-d record\nembedding"]
    for x, lab in zip(xs, labels):
        ax.text(x, 0.78, lab, ha="center", va="center", fontsize=7.4, fontweight="bold")
    draw_tokens(ax, 0.035, 0.61)
    draw_tokens(ax, 0.265, 0.61)
    draw_tokens(ax, 0.475, 0.61, masked={(0, 2), (1, 1), (1, 4), (2, 3)})
    draw_transformer(ax, 0.65, 0.43)
    for i in range(5):
        ax.add_patch(patches.Circle((0.88, 0.63 - i * 0.055), 0.014, facecolor=DARK_BLUE, edgecolor=DARK_BLUE))
    arrow(ax, (0.19, 0.53), (0.25, 0.53))
    arrow(ax, (0.41, 0.53), (0.47, 0.53))
    arrow(ax, (0.61, 0.53), (0.65, 0.53))
    arrow(ax, (0.77, 0.53), (0.84, 0.53))
    rounded_box(ax, (0.54, 0.13), 0.16, 0.10, "Categorical\nreconstruction loss", DARK_BLUE, "white", 6.8)
    rounded_box(ax, (0.73, 0.13), 0.16, 0.10, "Numeric\nreconstruction loss", ORANGE, "white", 6.8)
    arrow(ax, (0.70, 0.43), (0.62, 0.24), DARK_BLUE, 0.9)
    arrow(ax, (0.73, 0.43), (0.81, 0.24), ORANGE, 0.9)
    ax.text(0.50, 0.045, "Pretrained on 2016--2022 only; labels used only after encoder training", ha="center", va="center", fontsize=7.2, color=GRAY)
    return save_panel(fig, "Figure_1C_content_no_label")


def draw_panel_d() -> Path:
    fig, ax = plt.subplots(figsize=(6.3, 3.25))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_title(ax, "Phenotyping, risk enrichment, and transfer tests")

    rounded_box(ax, (0.04, 0.61), 0.16, 0.11, "2023 SSL\nembeddings", DARK_BLUE, LIGHT_BLUE, 6.95)
    rounded_box(ax, (0.25, 0.61), 0.15, 0.11, "PCA +\nk-means", DARK_BLUE, LIGHT_BLUE, 6.95)
    rounded_box(ax, (0.47, 0.59), 0.16, 0.15, "Fixed\nphenotype\ncentroids", DARK_BLUE, LIGHT_BLUE, 6.95)
    arrow(ax, (0.20, 0.665), (0.25, 0.665))
    arrow(ax, (0.40, 0.665), (0.47, 0.665))

    rounded_box(ax, (0.04, 0.34), 0.16, 0.11, "2024\nembeddings", GRAY, LIGHT_GRAY, 6.95)
    rounded_box(ax, (0.25, 0.31), 0.17, 0.145, "Phenotype\nassignment", ORANGE, LIGHT_ORANGE, 6.95)
    rounded_box(ax, (0.50, 0.31), 0.18, 0.145, "Platt-calibrated\nrisk model", ORANGE, LIGHT_ORANGE, 6.85)
    arrow(ax, (0.55, 0.59), (0.34, 0.46), GRAY, 0.95)
    arrow(ax, (0.20, 0.395), (0.25, 0.395))
    arrow(ax, (0.42, 0.385), (0.50, 0.385))

    outputs = ["AUPRC", "Top-k enrichment", "Calibration", "Decision curve", "Subgroup\nrobustness"]
    for idx, item in enumerate(outputs):
        y = 0.64 - idx * 0.09
        rounded_box(ax, (0.79, y), 0.17, 0.058, item, "#BDBDBD", "white", 5.9)
        arrow(ax, (0.68, 0.385), (0.79, y + 0.029), GRAY, 0.72)

    rounded_box(ax, (0.18, 0.07), 0.25, 0.105, "Linked infant death\ntransfer endpoint", PURPLE, LIGHT_PURPLE, 6.75)
    rounded_box(ax, (0.55, 0.07), 0.25, 0.105, "SINASC registry\nstress-test", GREEN, LIGHT_GREEN, 6.75)
    arrow(ax, (0.34, 0.31), (0.30, 0.18), PURPLE, 0.85)
    arrow(ax, (0.36, 0.31), (0.67, 0.18), GREEN, 0.85)
    ax.text(0.50, 0.015, "2024 and transfer labels used only for evaluation", ha="center", va="bottom", fontsize=6.55, color=GRAY, style="italic")
    return save_panel(fig, "Figure_1D_content_no_label")


def compose() -> None:
    from PIL import Image, ImageDraw, ImageFont

    panel_paths = [draw_panel_a(), draw_panel_b(), draw_panel_c(), draw_panel_d()]
    panels = [Image.open(path).convert("RGB") for path in panel_paths]
    cell_w, cell_h = 1800, 960
    gutter = 56
    outer = 46
    width = 2 * cell_w + gutter + 2 * outer
    height = 2 * cell_h + gutter + 2 * outer
    canvas = Image.new("RGB", (width, height), "white")
    positions = [
        (outer, outer),
        (outer + cell_w + gutter, outer),
        (outer, outer + cell_h + gutter),
        (outer + cell_w + gutter, outer + cell_h + gutter),
    ]
    for panel, pos in zip(panels, positions):
        fitted = Image.new("RGB", (cell_w, cell_h), "white")
        scale = min(cell_w / panel.width, cell_h / panel.height)
        resized = panel.resize((int(panel.width * scale), int(panel.height * scale)), Image.Resampling.LANCZOS)
        fitted.paste(resized, ((cell_w - resized.width) // 2, (cell_h - resized.height) // 2))
        canvas.paste(fitted, pos)

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 56)
    except OSError:
        font = ImageFont.load_default()
    for label, (x, y) in zip(["A", "B", "C", "D"], positions):
        draw.text((x + 10, y + 4), label, fill=(20, 20, 20), font=font)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_FIG_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / "Figure_1_study_design_ssl_pipeline.png"
    for target_dir in [FIGURE_DIR, LATEX_FIG_DIR, SUBMISSION_FIG_DIR]:
        out_png = target_dir / "Figure_1_study_design_ssl_pipeline.png"
        canvas.save(out_png, dpi=(450, 450))
        canvas.save(target_dir / "Figure_1_study_design_ssl_pipeline.pdf", resolution=450)
    plt.close("all")


def main() -> None:
    setup_style()
    compose()


if __name__ == "__main__":
    main()
