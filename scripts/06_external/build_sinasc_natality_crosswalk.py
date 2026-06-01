#!/usr/bin/env python
"""Build a SINASC-to-U.S. Natality variable mapping feasibility table.

The script reads the OpenDataSUS/SINASC CSV ZIP files without fully extracting
them, extracts available CSV fields and lightweight missingness examples, scans
the official SINASC structure PDF for variable-name context, and writes a
crosswalk against the current U.S. Natality analytic variables used in this
project.
"""

from __future__ import annotations

import csv
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINASC_DIR = PROJECT_ROOT / "data" / "external" / "opendatasus_sinasc"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DOC_DIR = PROJECT_ROOT / "docs"
DICT_PDF = SINASC_DIR / "SINASC_Estrutura_data_dictionary.pdf"
NATALITY_DICT = TABLE_DIR / "nat2024_analytic_column_dictionary.csv"

ZIP_BY_YEAR = {
    2023: SINASC_DIR / "SINASC_2023_csv.zip",
    2024: SINASC_DIR / "SINASC_2024_csv.zip",
}

MISSING_STRINGS = {"", " ", "NA", "NaN", "nan", "NULL", "None"}


@dataclass(frozen=True)
class ConceptMap:
    concept_group: str
    concept: str
    natality_variables: str
    sinasc_candidates: tuple[str, ...]
    role: str
    planned_mapping: str
    feasibility_note: str


CONCEPTS = [
    ConceptMap("Maternal demographics", "Maternal age", "MAGER; MAGER9", ("IDADEMAE",), "input/subgroup", "direct", "Continuous maternal age is available; age-group variables can be derived."),
    ConceptMap("Maternal demographics", "Maternal race/ethnicity", "MRACE31; MRACE6; MRACE15; MHISPX; MHISP_R; MRACEHISP", ("RACACORMAE", "RACACOR"), "input/subgroup", "partial", "Brazil race/color categories are not equivalent to U.S. race and Hispanic-origin variables; use only for local subgroup profiling."),
    ConceptMap("Maternal demographics", "Marital status", "DMAR", ("ESTCIVMAE",), "input", "partial", "Available but category definitions differ from U.S. birth-certificate coding."),
    ConceptMap("Maternal demographics", "Maternal education", "MEDUC", ("ESCMAE", "ESCMAE2010", "ESCMAEAGR1", "SERIESCMAE"), "input/subgroup", "partial", "Available in several SINASC codings; harmonization should use broad education groups."),
    ConceptMap("Geography/context", "Maternal residence geography", "MBSTATE_REC; RESTATUS", ("CODMUNRES", "CODPAISRES"), "input/context", "partial", "Residence municipality/country are available; U.S. residence-status construct is not directly available."),
    ConceptMap("Obstetric history", "Live-birth order / parity", "LBO_REC; TBO_REC", ("QTDFILVIVO", "QTDGESTANT", "PARIDADE", "QTDPARTNOR", "QTDPARTCES"), "input", "partial", "Prior live births and total pregnancies can be approximated; coding and missingness need empirical checks."),
    ConceptMap("Obstetric history", "Previous cesarean", "RF_CESAR; RF_CESARN", ("QTDPARTCES",), "input", "direct-derived", "Number of prior cesarean deliveries can support a derived prior-cesarean flag."),
    ConceptMap("Obstetric history", "Previous preterm birth", "RF_PPTERM", tuple(), "input", "not available", "No direct public SINASC field was found for previous preterm birth."),
    ConceptMap("Prenatal care", "Month prenatal care began", "PRECARE; PRECARE5", ("MESPRENAT",), "input", "direct/partial", "Month of prenatal-care initiation appears available; broad trimester/month grouping can be derived."),
    ConceptMap("Prenatal care", "Number of prenatal visits", "PREVIS; PREVIS_REC", ("CONSULTAS", "CONSPRENAT"), "input", "direct/partial", "Prenatal visit count/grouping is available; exact coding differs."),
    ConceptMap("Social/context", "WIC receipt / payment source", "WIC; PAY_REC", tuple(), "input/context", "not available", "No equivalent public SINASC field was found for U.S. WIC or payment source."),
    ConceptMap("Behavior", "Smoking before/during pregnancy", "CIG_0; CIG_1; CIG_2; CIG_3; CIG_REC", tuple(), "input", "not available", "No direct public SINASC smoking variable was found."),
    ConceptMap("Anthropometry", "Maternal height/BMI/prepregnancy weight/weight gain", "M_Ht_In; BMI; BMI_R; PWgt_R; WTGAIN; WTGAIN_REC", tuple(), "input", "not available", "Maternal anthropometry is a major U.S. Natality feature family but is not directly available in public SINASC."),
    ConceptMap("Endocrine/metabolic", "Pregestational or gestational diabetes", "RF_PDIAB; RF_GDIAB", tuple(), "input", "not available", "No direct public SINASC diabetes field was found."),
    ConceptMap("Hypertensive disorders", "Pregestational hypertension, gestational hypertension, eclampsia", "RF_PHYPE; RF_GHYPE; RF_EHYPE", tuple(), "input", "not available", "No direct public SINASC hypertensive-disorder field was found."),
    ConceptMap("Reproductive medicine", "Infertility treatment / fertility drugs / ART", "RF_INFTR; RF_FEDRG; RF_ARTEC", tuple(), "input", "not available", "No direct public SINASC infertility-treatment or ART field was found."),
    ConceptMap("Infections", "Maternal infections", "IP_GON; IP_SYPH; IP_CHLAM; IP_HEPB; IP_HEPC", tuple(), "input", "not available", "No direct public SINASC infection fields corresponding to the U.S. variables were found."),
    ConceptMap("Pregnancy/birth context", "Plurality", "DPLURAL", ("GRAVIDEZ",), "input/context", "direct/partial", "Singleton/multiple pregnancy can be derived from pregnancy type."),
    ConceptMap("Infant demographics", "Infant sex", "SEX", ("SEXO",), "input/context", "direct", "Infant sex is directly available."),
    ConceptMap("Birth status outcome", "Gestational age", "COMBGEST; outcome_gest_weeks_combined", ("SEMAGESTAC", "GESTACAO"), "outcome/context", "direct/partial", "Gestational age is available as weeks and/or grouped categories; use as outcome/status variable, not an antepartum input."),
    ConceptMap("Birth status outcome", "Birthweight", "DBWT; outcome_birthweight_g", ("PESO",), "outcome/context", "direct", "Birthweight in grams is directly available."),
    ConceptMap("Birth status outcome", "5-minute Apgar", "APGAR5; outcome_apgar5", ("APGAR5",), "outcome", "direct", "Five-minute Apgar is directly available."),
    ConceptMap("Birth status outcome", "Preterm birth", "outcome_preterm_lt37", ("SEMAGESTAC", "GESTACAO"), "outcome", "derived", "Can be derived from gestational age."),
    ConceptMap("Birth status outcome", "Low/very-low birthweight", "outcome_low_birthweight_lt2500g; outcome_very_low_birthweight_lt1500g", ("PESO",), "outcome", "derived", "Can be derived from birthweight."),
    ConceptMap("Birth status outcome", "Low 5-minute Apgar", "outcome_low_apgar5_lt7", ("APGAR5",), "outcome", "derived", "Can be derived from 5-minute Apgar."),
    ConceptMap("Newborn morbidity", "NICU admission, ventilation, seizures", "AB_NICU; AB_VENT; AB_VENT6; AB_SEIZ", tuple(), "outcome", "not available", "No direct public SINASC equivalents were found for the U.S. newborn morbidity fields."),
    ConceptMap("Congenital anomaly", "Congenital anomaly marker/code", "not in primary current model", ("IDANOMAL", "CODANOMAL"), "outcome/exclusion/context", "direct/partial", "Useful for sensitivity or phenotype interpretation, but not equivalent to current primary endpoints."),
    ConceptMap("Maternal morbidity", "Maternal transfusion/ICU/rupture/hysterectomy/laceration", "MM_MTR; MM_UHYST; MM_AICU; MM_RUPT; MR_LAC", tuple(), "outcome", "not available", "Current maternal morbidity endpoint cannot be reproduced directly in public SINASC."),
    ConceptMap("Delivery context", "Delivery mode", "not used as primary input", ("PARTO",), "context/secondary", "direct/partial", "Delivery mode is available but should be treated as delivery-context or secondary analysis to avoid target leakage."),
    ConceptMap("Delivery context", "Robson classification / labor context", "not used in current U.S. SSL model", ("TPROBSON", "STTRABPART", "STCESPARTO", "TPAPRESENT"), "context/secondary", "direct/partial", "Can support obstetric profiling but is not a direct current-model input."),
    ConceptMap("Prenatal care quality", "Kotelchuck/prenatal-care adequacy", "not used in current U.S. SSL model", ("KOTELCHUCK",), "context/secondary", "direct/partial", "Useful for Brazil-specific prenatal-care adequacy profiling."),
]


def open_csv_header(zip_path: Path) -> tuple[str, list[str], str, list[dict[str, str]]]:
    with zipfile.ZipFile(zip_path) as archive:
        entry = archive.namelist()[0]
        with archive.open(entry) as handle:
            sample = handle.read(65536)
    for encoding in ("utf-8-sig", "latin1"):
        try:
            text = sample.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = sample.decode("utf-8", errors="replace")
        encoding = "utf-8"
    dialect = csv.Sniffer().sniff(text[:5000], delimiters=",;|\t")
    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    rows = []
    for idx, row in enumerate(reader):
        if idx >= 5:
            break
        rows.append(dict(row))
    return entry, list(reader.fieldnames or []), dialect.delimiter, rows


def read_selected_stats(year: int, zip_path: Path, columns: list[str], delimiter: str) -> dict[str, dict[str, object]]:
    present_columns = [column for column in columns if column]
    stats = {
        column: {
            "year": year,
            "sinasc_variable": column,
            "n_rows": 0,
            "n_missing": 0,
            "example_values": [],
        }
        for column in present_columns
    }
    if not present_columns:
        return stats
    chunks = pd.read_csv(
        zip_path,
        sep=delimiter,
        usecols=present_columns,
        dtype="string",
        chunksize=200_000,
        encoding="utf-8",
        low_memory=False,
    )
    for chunk in chunks:
        n = len(chunk)
        for column in present_columns:
            series = chunk[column]
            missing = series.isna() | series.astype("string").str.strip().isin(MISSING_STRINGS)
            stats[column]["n_rows"] += int(n)
            stats[column]["n_missing"] += int(missing.sum())
            examples = stats[column]["example_values"]
            if len(examples) < 8:
                values = (
                    series[~missing]
                    .astype(str)
                    .str.strip()
                    .drop_duplicates()
                    .head(8 - len(examples))
                    .tolist()
                )
                for value in values:
                    if value not in examples:
                        examples.append(value)
    return stats


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    try:
        reader = PdfReader(str(pdf_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def context_for_variable(text: str, variable: str, window: int = 180) -> str:
    if not text:
        return ""
    match = re.search(rf"\b{re.escape(variable)}\b", text)
    if not match:
        return ""
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    context = re.sub(r"\s+", " ", text[start:end]).strip()
    return context


def status_from_availability(planned: str, candidates: tuple[str, ...], available_any: bool) -> str:
    if not candidates:
        return "not_available"
    if not available_any:
        return "candidate_not_in_header"
    if planned in {"direct", "derived", "direct-derived"}:
        return planned
    if "direct" in planned and "partial" in planned:
        return "direct_or_partial"
    return planned.replace(" ", "_")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    zip_meta: dict[int, dict[str, object]] = {}
    all_headers: dict[int, list[str]] = {}
    samples: dict[int, list[dict[str, str]]] = {}
    delimiters: dict[int, str] = {}
    for year, zip_path in ZIP_BY_YEAR.items():
        entry, header, delimiter, sample_rows = open_csv_header(zip_path)
        zip_meta[year] = {
            "zip_path": str(zip_path),
            "entry": entry,
            "delimiter": delimiter,
            "n_columns": len(header),
            "columns": header,
        }
        all_headers[year] = header
        samples[year] = sample_rows
        delimiters[year] = delimiter

    header_union = sorted({field for fields in all_headers.values() for field in fields})
    candidate_columns = sorted({candidate for concept in CONCEPTS for candidate in concept.sinasc_candidates})
    candidate_columns = [column for column in candidate_columns if any(column in fields for fields in all_headers.values())]

    stats_frames = []
    for year, zip_path in ZIP_BY_YEAR.items():
        cols = [column for column in candidate_columns if column in all_headers[year]]
        stats = read_selected_stats(year, zip_path, cols, delimiters[year])
        stats_frames.extend(stats.values())
    stats_df = pd.DataFrame(stats_frames)
    if not stats_df.empty:
        stats_df["missing_pct"] = stats_df["n_missing"] / stats_df["n_rows"] * 100
        stats_df["example_values"] = stats_df["example_values"].apply(lambda values: "; ".join(map(str, values)))

    pdf_text = extract_pdf_text(DICT_PDF)
    dict_rows = []
    for variable in header_union:
        dict_rows.append(
            {
                "sinasc_variable": variable,
                "in_2023_header": variable in all_headers.get(2023, []),
                "in_2024_header": variable in all_headers.get(2024, []),
                "in_pdf_text": bool(context_for_variable(pdf_text, variable, 10)),
                "dictionary_context": context_for_variable(pdf_text, variable),
            }
        )
    dict_df = pd.DataFrame(dict_rows)

    natality_df = pd.read_csv(NATALITY_DICT)
    current_inputs = sorted(
        natality_df.loc[natality_df["group"].isin(["input", "context"]), "source_variable"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    current_outcomes = sorted(
        natality_df.loc[natality_df["group"].eq("outcome"), "source_variable"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    crosswalk_rows = []
    for concept in CONCEPTS:
        available_2023 = [candidate for candidate in concept.sinasc_candidates if candidate in all_headers.get(2023, [])]
        available_2024 = [candidate for candidate in concept.sinasc_candidates if candidate in all_headers.get(2024, [])]
        available_any = bool(available_2023 or available_2024)
        contexts = []
        for candidate in concept.sinasc_candidates:
            ctx = context_for_variable(pdf_text, candidate)
            if ctx:
                contexts.append(f"{candidate}: {ctx}")
        crosswalk_rows.append(
            {
                "concept_group": concept.concept_group,
                "concept": concept.concept,
                "current_us_natality_variables": concept.natality_variables,
                "sinasc_candidate_variables": "; ".join(concept.sinasc_candidates) if concept.sinasc_candidates else "",
                "sinasc_available_2023": "; ".join(available_2023),
                "sinasc_available_2024": "; ".join(available_2024),
                "mapping_status": status_from_availability(concept.planned_mapping, concept.sinasc_candidates, available_any),
                "current_project_role": concept.role,
                "can_reproduce_current_us_feature": "yes" if concept.planned_mapping in {"direct", "direct-derived", "derived"} and available_any else ("partial" if available_any else "no"),
                "feasibility_note": concept.feasibility_note,
                "dictionary_context": " || ".join(contexts),
            }
        )
    crosswalk_df = pd.DataFrame(crosswalk_rows)

    header_inventory = pd.DataFrame(
        {
            "sinasc_variable": header_union,
            "in_2023_header": [field in all_headers.get(2023, []) for field in header_union],
            "in_2024_header": [field in all_headers.get(2024, []) for field in header_union],
        }
    ).merge(dict_df[["sinasc_variable", "in_pdf_text", "dictionary_context"]], on="sinasc_variable", how="left")

    crosswalk_path = TABLE_DIR / "sinasc_natality_variable_crosswalk.csv"
    inventory_path = TABLE_DIR / "sinasc_header_dictionary_inventory.csv"
    stats_path = TABLE_DIR / "sinasc_candidate_variable_missingness_examples.csv"
    meta_path = TABLE_DIR / "sinasc_crosswalk_metadata.json"
    report_path = DOC_DIR / "44_sinasc_natality_crosswalk_report.md"

    crosswalk_df.to_csv(crosswalk_path, index=False, encoding="utf-8-sig")
    header_inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")

    status_counts = crosswalk_df["mapping_status"].value_counts().sort_index().to_dict()
    metadata = {
        "zip_meta": zip_meta,
        "n_sinasc_header_union": len(header_union),
        "n_candidate_columns_profiled": len(candidate_columns),
        "mapping_status_counts": status_counts,
        "current_natality_input_or_context_source_variables": current_inputs,
        "current_natality_outcome_source_variables": current_outcomes,
        "outputs": {
            "crosswalk": str(crosswalk_path),
            "header_inventory": str(inventory_path),
            "candidate_stats": str(stats_path),
            "report": str(report_path),
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    status_counts_series = crosswalk_df["mapping_status"].value_counts().sort_index()
    status_count_lines = [f"- {status}: {count}." for status, count in status_counts_series.items()]
    dict_context_count = int(header_inventory["in_pdf_text"].sum())
    year_row_counts = (
        stats_df.groupby("year")["n_rows"].max().dropna().astype(int).to_dict()
        if not stats_df.empty
        else {}
    )
    major_gaps = crosswalk_df[crosswalk_df["mapping_status"].eq("not_available")]
    available_inputs = crosswalk_df[
        crosswalk_df["current_project_role"].str.contains("input", na=False)
        & ~crosswalk_df["mapping_status"].eq("not_available")
    ]

    lines = [
        "# SINASC to U.S. Natality variable mapping feasibility",
        "",
        "## Data parsed",
        "",
    ]
    for year in sorted(zip_meta):
        meta = zip_meta[year]
        lines.append(f"- SINASC {year}: entry `{meta['entry']}`, delimiter `{meta['delimiter']}`, {meta['n_columns']} columns.")
    for year, n_rows in sorted(year_row_counts.items()):
        lines.append(f"- SINASC {year}: {n_rows:,} rows scanned for mapped candidate-variable missingness/examples.")
    lines.extend(
        [
            f"- SINASC header union across years: {len(header_union)} variables.",
            f"- Official PDF dictionary context extracted for {dict_context_count} of {len(header_union)} header variables.",
            f"- Candidate mapped SINASC variables profiled for missingness/examples: {len(candidate_columns)}.",
            "",
            "## Feasibility summary",
            "",
            *status_count_lines,
            "",
            "The SINASC public files can support an independent registry stress-test for demographics, prenatal care, parity, plurality, delivery mode, gestational age, birthweight, Apgar, and congenital anomaly markers. They cannot directly reproduce the current U.S. Natality SSL input space because major feature families are absent, especially maternal BMI/weight gain, smoking, diabetes, hypertensive disorders, infections, infertility/ART, WIC/payment, and the current maternal morbidity endpoint.",
            "",
            "## Mappable input families",
            "",
        ]
    )
    for _, row in available_inputs.iterrows():
        lines.append(f"- {row['concept']}: `{row['sinasc_candidate_variables']}` ({row['mapping_status']}).")
    lines.extend(["", "## Major missing current-model families", ""])
    for _, row in major_gaps.iterrows():
        lines.append(f"- {row['concept']}: U.S. variables `{row['current_us_natality_variables']}`.")
    lines.extend(
        [
            "",
            "## Recommended use",
            "",
            "Use SINASC as an independent registry workflow stress-test, not as direct external validation of the trained U.S. Natality encoder. The defensible design is to build a harmonized SINASC matrix from overlapping variables, train/develop/test a separate masked tabular SSL model within SINASC years, and evaluate phenotype enrichment for preterm birth, low birthweight, very low birthweight, low 5-minute Apgar, and congenital anomaly markers.",
            "",
            "## Output files",
            "",
            f"- Crosswalk: `{crosswalk_path}`",
            f"- SINASC header/dictionary inventory: `{inventory_path}`",
            f"- Candidate variable missingness/examples: `{stats_path}`",
            f"- Metadata: `{meta_path}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(crosswalk_path)
    print(inventory_path)
    print(stats_path)
    print(report_path)


if __name__ == "__main__":
    main()
