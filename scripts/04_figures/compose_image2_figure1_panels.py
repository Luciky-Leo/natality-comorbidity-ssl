#!/usr/bin/env python
"""Compose image2-generated Figure 1 panels into the manuscript canvas."""

from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "results" / "figures" / "source_assets"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
LATEX_FIG_DIR = PROJECT_ROOT / "submission_package" / "springer_nature_latex" / "figures"
SUBMISSION_FIG_DIR = PROJECT_ROOT / "submission_package" / "figures"

PANEL_FILES = [
    "Figure_1A_image2_cohort_temporal_split.png",
    "Figure_1B_image2_registry_inputs.png",
    "Figure_1C_image2_ssl_encoder.png",
    "Figure_1D_image2_phenotyping_transfer.png",
]
PANEL_TITLES = [
    "Cohort and temporal split",
    "Leakage-controlled registry inputs",
    "Masked tabular SSL encoder",
    "Phenotyping, risk enrichment, and transfer tests",
]
TITLE_CROP_FRACTIONS = [0.15, 0.11, 0.13, 0.12]


def content_bbox(img: Image.Image, threshold: int = 246, pad: int = 18) -> tuple[int, int, int, int]:
    rgb = img.convert("RGB")
    pixels = rgb.load()
    width, height = rgb.size
    left, top = width, height
    right, bottom = 0, 0
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if min(r, g, b) < threshold:
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)
    if right <= left or bottom <= top:
        return (0, 0, width, height)
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(width, right + pad),
        min(height, bottom + pad),
    )


def fit_to_cell(img: Image.Image, cell_w: int, cell_h: int, title: str, crop_fraction: float, pad: int = 24) -> Image.Image:
    top_crop = int(img.height * crop_fraction)
    body = img.crop((0, top_crop, img.width, img.height))
    cropped = body.crop(content_bbox(body))
    title_band = 106
    target_w = cell_w - 2 * pad
    target_h = cell_h - title_band - 2 * pad
    scale = min(target_w / cropped.width, target_h / cropped.height)
    resized = cropped.resize((int(cropped.width * scale), int(cropped.height * scale)), Image.Resampling.LANCZOS)
    cell = Image.new("RGB", (cell_w, cell_h), "white")
    x = (cell_w - resized.width) // 2
    y = title_band + (cell_h - title_band - resized.height) // 2
    cell.paste(resized, (x, y))

    draw = ImageDraw.Draw(cell)
    try:
        title_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 48)
    except OSError:
        title_font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=title_font)
    tx = (cell_w - (bbox[2] - bbox[0])) // 2
    draw.text((tx, 30), title, fill=(18, 18, 18), font=title_font)
    return cell


def write_embedded_svg(png_path: Path, svg_path: Path, width: int, height: int) -> None:
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'  <image href="data:image/png;base64,{encoded}" width="{width}" height="{height}"/>\n'
        "</svg>\n"
    )
    svg_path.write_text(svg, encoding="utf-8")


def main() -> None:
    panels = [Image.open(SOURCE_DIR / name).convert("RGB") for name in PANEL_FILES]

    cell_w, cell_h = 1800, 930
    gutter = 52
    outer = 48
    label_offset_x = 6
    label_offset_y = 4

    width = 2 * cell_w + gutter + 2 * outer
    height = 2 * cell_h + gutter + 2 * outer
    canvas = Image.new("RGB", (width, height), "white")
    positions = [
        (outer, outer),
        (outer + cell_w + gutter, outer),
        (outer, outer + cell_h + gutter),
        (outer + cell_w + gutter, outer + cell_h + gutter),
    ]

    for panel, title, crop_fraction, pos in zip(panels, PANEL_TITLES, TITLE_CROP_FRACTIONS, positions):
        canvas.paste(fit_to_cell(panel, cell_w, cell_h, title, crop_fraction), pos)

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 58)
    except OSError:
        font = ImageFont.load_default()
    for label, (x, y) in zip(["A", "B", "C", "D"], positions):
        draw.text((x + label_offset_x, y + label_offset_y), label, fill=(18, 18, 18), font=font)

    for target_dir in [FIGURE_DIR, LATEX_FIG_DIR, SUBMISSION_FIG_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        out_png = target_dir / "Figure_1_study_design_ssl_pipeline.png"
        canvas.save(out_png, dpi=(450, 450))
        canvas.save(target_dir / "Figure_1_study_design_ssl_pipeline.pdf", resolution=450)
        write_embedded_svg(out_png, target_dir / "Figure_1_study_design_ssl_pipeline.svg", width, height)


if __name__ == "__main__":
    main()
