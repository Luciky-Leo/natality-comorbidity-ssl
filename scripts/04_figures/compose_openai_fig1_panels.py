#!/usr/bin/env python
"""Compose OpenAI-generated Figure 1 panels into the manuscript Figure 1 canvas.

The source panel files were created before the full-scale analysis. To avoid
regressing the cohort-size text, the compact compositor first snapshots the
current full-scale Figure 1 and then rebuilds a tighter panel grid from that
snapshot.
"""

from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "results" / "figures" / "source_assets"
OUT_DIR = PROJECT_ROOT / "results" / "figures"
SUBMISSION_FIG_DIR = PROJECT_ROOT / "submission_package" / "figures"
LATEX_FIG_DIR = PROJECT_ROOT / "submission_package" / "springer_nature_latex" / "figures"

PANEL_FILES = [
    "Figure_1A_openai_cohort_temporal_split.png",
    "Figure_1B_openai_inputs.png",
    "Figure_1C_openai_ssl_encoder.png",
    "Figure_1D_openai_phenotype_risk.png",
]

OUTPUT_NAME = "Figure_1_study_design_ssl_pipeline"
COMPACT_SOURCE = OUT_DIR / f"{OUTPUT_NAME}_precompact_source.png"


def content_bbox(img: Image.Image, threshold: int = 242, pad: int = 12) -> tuple[int, int, int, int]:
    pixels = img.load()
    width, height = img.size
    left, top = width, height
    right, bottom = 0, 0
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if min(r, g, b) < threshold:
                if x < left:
                    left = x
                if y < top:
                    top = y
                if x > right:
                    right = x
                if y > bottom:
                    bottom = y
    if right <= left or bottom <= top:
        return (0, 0, width, height)
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(width, right + pad),
        min(height, bottom + pad),
    )


def crop_current_panels(composite: Image.Image) -> list[Image.Image]:
    width, height = composite.size
    old_outer = 26
    old_gutter = 36
    old_cell_w = (width - 2 * old_outer - old_gutter) // 2
    old_cell_h = (height - 2 * old_outer - old_gutter) // 2
    boxes = [
        (old_outer, old_outer, old_outer + old_cell_w, old_outer + old_cell_h),
        (old_outer + old_cell_w + old_gutter, old_outer, width - old_outer, old_outer + old_cell_h),
        (old_outer, old_outer + old_cell_h + old_gutter, old_outer + old_cell_w, height - old_outer),
        (old_outer + old_cell_w + old_gutter, old_outer + old_cell_h + old_gutter, width - old_outer, height - old_outer),
    ]
    panels = []
    for box in boxes:
        panel = composite.crop(box)
        panels.append(panel.crop(content_bbox(panel)))
    panels[0] = update_panel_a_downstream_label(panels[0])
    return panels


def update_panel_a_downstream_label(panel: Image.Image) -> Image.Image:
    """Remove legacy downstream-sample text from Figure 1A."""

    out = panel.copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    draw.rectangle((int(width * 0.30), int(height * 0.78), int(width * 0.86), height), fill="white")
    return out


def fit_panel(img: Image.Image, cell_w: int, cell_h: int, margin: int) -> Image.Image:
    max_w = cell_w - 2 * margin
    max_h = cell_h - 2 * margin
    scale = min(max_w / img.width, max_h / img.height)
    new_size = (int(img.width * scale), int(img.height * scale))
    resized = img.resize(new_size, Image.Resampling.LANCZOS)
    cell = Image.new("RGB", (cell_w, cell_h), "white")
    x = margin
    y = margin
    cell.paste(resized, (x, y))
    return cell


def write_svg_with_embedded_png(png_path: Path, svg_path: Path, width: int, height: int) -> None:
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'  <image href="data:image/png;base64,{encoded}" width="{width}" height="{height}"/>\n'
        "</svg>\n"
    )
    svg_path.write_text(svg, encoding="utf-8")


def main() -> None:
    current = LATEX_FIG_DIR / f"{OUTPUT_NAME}.png"
    if not COMPACT_SOURCE.exists():
        Image.open(current).convert("RGB").save(COMPACT_SOURCE, dpi=(450, 450))

    panels = crop_current_panels(Image.open(COMPACT_SOURCE).convert("RGB"))

    cell_w, cell_h, margin = 1760, 1000, 18
    gutter = 30
    outer = 24
    width = 2 * cell_w + gutter + 2 * outer
    height = 2 * cell_h + gutter + 2 * outer
    canvas = Image.new("RGB", (width, height), "white")

    positions = [
        (outer, outer),
        (outer + cell_w + gutter, outer),
        (outer, outer + cell_h + gutter),
        (outer + cell_w + gutter, outer + cell_h + gutter),
    ]

    for panel_img, pos in zip(panels, positions):
        panel = fit_panel(panel_img, cell_w, cell_h, margin)
        canvas.paste(panel, pos)

    output = OUT_DIR / f"{OUTPUT_NAME}.png"
    canvas.save(output, dpi=(450, 450))
    canvas.save(OUT_DIR / f"{OUTPUT_NAME}.pdf", resolution=450)
    write_svg_with_embedded_png(output, OUT_DIR / f"{OUTPUT_NAME}.svg", width, height)

    for target_dir in [SUBMISSION_FIG_DIR, LATEX_FIG_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        canvas.save(target_dir / output.name, dpi=(450, 450))
        canvas.save(target_dir / f"{OUTPUT_NAME}.pdf", resolution=450)
        write_svg_with_embedded_png(target_dir / output.name, target_dir / f"{OUTPUT_NAME}.svg", width, height)

    print(f"wrote {output}")


if __name__ == "__main__":
    main()
