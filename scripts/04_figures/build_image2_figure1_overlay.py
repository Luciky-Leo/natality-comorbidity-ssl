#!/usr/bin/env python
"""Build Figure 1 from an image2-generated schematic plus exact text overlay."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
ASSET_DIR = FIGURE_DIR / "source_assets"
SUBMISSION_FIG_DIR = PROJECT_ROOT / "submission_package" / "figures"
SN_LATEX_FIG_DIR = PROJECT_ROOT / "submission_package" / "springer_nature_latex" / "figures"
BMC_FIG_DIR = PROJECT_ROOT / "submission_package" / "bmc_midd_latex_upload" / "figures"

IMAGE2_SOURCE = ASSET_DIR / "Figure_1_image2_schematic_background.png"
OUTPUT_NAME = "Figure_1_study_design_ssl_pipeline"

TEXT = "#1E2832"
MUTED = "#56616F"
BLUE = "#114F9B"
ORANGE = "#E66B1A"
GRAY = "#6E7682"
PURPLE = "#7A3D99"
GREEN = "#25824A"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    *,
    fill: str = TEXT,
    spacing: int = 4,
) -> None:
    draw.multiline_text(xy, text, font=fnt, fill=fill, anchor="mm", align="center", spacing=spacing)


def left_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    *,
    fill: str = TEXT,
    spacing: int = 4,
) -> None:
    draw.multiline_text(xy, text, font=fnt, fill=fill, anchor="lm", align="left", spacing=spacing)


def panel_badge(draw: ImageDraw.ImageDraw, x: int, y: int, letter: str, title: str = "") -> None:
    draw.rounded_rectangle((x, y, x + 58, y + 36), radius=8, fill=TEXT)
    centered_text(draw, (x + 29, y + 18), letter, font(20, True), fill=WHITE)
    if title:
        left_text(draw, (x + 74, y + 18), title, font(18, True))


def box_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: str = TEXT, size: int = 16) -> None:
    centered_text(draw, xy, text, font(size, True), fill=color, spacing=2)


def build() -> Path:
    if not IMAGE2_SOURCE.exists():
        raise FileNotFoundError(f"Missing image2 source asset: {IMAGE2_SOURCE}")

    img = Image.open(IMAGE2_SOURCE).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Panel labels only. Detailed panel meaning is handled by the manuscript caption.
    panel_badge(draw, 72, 48, "A")
    panel_badge(draw, 72, 300, "B")
    panel_badge(draw, 565, 300, "C")
    panel_badge(draw, 1132, 300, "D")

    # A: cohort spine.
    box_label(draw, (512, 132), "2016-2022", BLUE, 13)
    box_label(draw, (920, 132), "2023", ORANGE, 13)
    box_label(draw, (1275, 132), "2024", GRAY, 13)

    # B: input domains and matrix.
    input_labels = [
        ("Demographics", 205, 360),
        ("Anthropometry", 205, 418),
        ("Prenatal care", 205, 476),
        ("Diabetes / HTN", 205, 533),
        ("Smoking / infection", 205, 590),
        ("Infertility / ART", 205, 648),
        ("Obstetric history", 205, 706),
        ("Missingness", 205, 764),
    ]
    for txt, x, y in input_labels:
        centered_text(draw, (x, y), txt, font(11, False), fill=TEXT)
    centered_text(draw, (405, 320), "registry variables", font(13, True), fill=MUTED)
    left_text(draw, (72, 790), "outcome components excluded", font(12, True), fill=ORANGE)

    # C: SSL encoder.
    centered_text(draw, (650, 360), "tokens", font(11, True), fill=TEXT)
    centered_text(draw, (820, 360), "masking", font(11, True), fill=TEXT)
    centered_text(draw, (1018, 360), "encoder", font(11, True), fill=TEXT)
    centered_text(draw, (1118, 360), "48-d", font(11, True), fill=TEXT)
    box_label(draw, (667, 748), "categorical loss", BLUE, 10)
    box_label(draw, (875, 748), "numeric loss", ORANGE, 10)

    # D: phenotype and risk models.
    box_label(draw, (1250, 365), "2023", BLUE, 10)
    box_label(draw, (1398, 365), "PCA", BLUE, 10)
    box_label(draw, (1545, 365), "centroids", BLUE, 10)
    box_label(draw, (1250, 560), "2024", GRAY, 10)
    box_label(draw, (1390, 560), "phenotype", ORANGE, 10)
    box_label(draw, (1538, 560), "risk", ORANGE, 10)
    box_label(draw, (1222, 792), "linked death", PURPLE, 10)
    box_label(draw, (1455, 792), "SINASC", GREEN, 10)

    # Legend.
    legend = [
        ((195, 950), "Training / pretraining\n2016-2022", BLUE),
        ((470, 950), "Development / calibration\n2023", ORANGE),
        ((740, 950), "Temporal test\n2024", GRAY),
        ((1015, 950), "Linked infant death\ntransfer", PURPLE),
        ((1260, 950), "SINASC registry\nstress-test", GREEN),
        ((1515, 950), "Registry-defined\ninputs only", GRAY),
    ]
    for (x, y), txt, color in legend:
        left_text(draw, (x, y), txt, font(10, False), fill=color, spacing=2)

    for target in [FIGURE_DIR, SUBMISSION_FIG_DIR, SN_LATEX_FIG_DIR, BMC_FIG_DIR]:
        target.mkdir(parents=True, exist_ok=True)
        png_path = target / f"{OUTPUT_NAME}.png"
        pdf_path = target / f"{OUTPUT_NAME}.pdf"
        img.save(png_path, dpi=(450, 450))
        img.save(pdf_path, resolution=450.0)

    # Keep an SVG wrapper for upload packages that expect a same-name SVG.
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{img.width}" height="{img.height}" '
        f'viewBox="0 0 {img.width} {img.height}">'
        f'<image href="{OUTPUT_NAME}.png" width="{img.width}" height="{img.height}"/>'
        "</svg>\n"
    )
    for target in [FIGURE_DIR, SUBMISSION_FIG_DIR, SN_LATEX_FIG_DIR, BMC_FIG_DIR]:
        (target / f"{OUTPUT_NAME}.svg").write_text(svg, encoding="utf-8")

    return FIGURE_DIR / f"{OUTPUT_NAME}.png"


def main() -> None:
    print(build())


if __name__ == "__main__":
    main()
