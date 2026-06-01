#!/usr/bin/env python
"""Build the 2024 analytic cohort from CDC/NCHS Natality public-use data.

The parser streams the CDC zip through 7-Zip, derives leakage-controlled input
columns and outcome endpoints, and writes a Parquet file suitable for baseline
modelling and later self-supervised learning.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP = PROJECT_ROOT / "data" / "raw" / "Nat2024us.zip"
DEFAULT_FIELDS = PROJECT_ROOT / "config" / "nat2024_smoke_fields.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "nat2024_analytic_cohort.parquet"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_7Z = Path(r"C:\Program Files\NVIDIA Corporation\NVIDIA app\7z.exe")


NUMERIC_INT_FIELDS = {
    "MAGER",
    "PRECARE",
    "PREVIS",
    "CIG_0",
    "CIG_1",
    "CIG_2",
    "CIG_3",
    "M_Ht_In",
    "PWgt_R",
    "DWgt_R",
    "WTGAIN",
    "RF_CESARN",
}
NUMERIC_FLOAT_FIELDS = {"BMI"}

INPUT_FIELDS = [
    "MAGER",
    "MAGER9",
    "MBSTATE_REC",
    "RESTATUS",
    "MRACE31",
    "MRACE6",
    "MRACE15",
    "MHISPX",
    "MHISP_R",
    "MRACEHISP",
    "DMAR",
    "MEDUC",
    "LBO_REC",
    "TBO_REC",
    "ILLB_R11",
    "PRECARE",
    "PRECARE5",
    "PREVIS",
    "PREVIS_REC",
    "WIC",
    "CIG_0",
    "CIG_1",
    "CIG_2",
    "CIG_3",
    "CIG0_R",
    "CIG1_R",
    "CIG2_R",
    "CIG3_R",
    "CIG_REC",
    "M_Ht_In",
    "BMI",
    "BMI_R",
    "PWgt_R",
    "WTGAIN",
    "WTGAIN_REC",
    "RF_PDIAB",
    "RF_GDIAB",
    "RF_PHYPE",
    "RF_GHYPE",
    "RF_EHYPE",
    "RF_PPTERM",
    "RF_INFTR",
    "RF_FEDRG",
    "RF_ARTEC",
    "RF_CESAR",
    "RF_CESARN",
    "NO_RISKS",
    "IP_GON",
    "IP_SYPH",
    "IP_CHLAM",
    "IP_HEPB",
    "IP_HEPC",
    "NO_INFEC",
    "PAY_REC",
    "DPLURAL",
    "SEX",
]

CONTEXT_FIELDS = ["DWgt_R"]


def load_fields(path: Path) -> list[dict[str, object]]:
    fields: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            row["start"] = int(row["start"])
            row["end"] = int(row["end"])
            row["missing_set"] = {
                value for value in row["missing_values"].split("|") if value
            }
            fields.append(row)
    return fields


@contextmanager
def open_record_stream(zip_path: Path, sevenzip: Path | None):
    if sevenzip and sevenzip.exists():
        proc = subprocess.Popen(
            [str(sevenzip), "e", "-so", str(zip_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if proc.stdout is None:
            raise RuntimeError("7-Zip stdout pipe could not be opened")
        try:
            yield proc.stdout, "7z"
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            else:
                return_code = proc.wait()
                if return_code != 0:
                    raise RuntimeError(f"7-Zip exited with status {return_code}")
        return

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise RuntimeError(f"Expected one file in {zip_path}, got {names}")
        with zf.open(names[0], "r") as fh:
            yield fh, "python_zipfile"


def raw_value(raw: bytes, field: dict[str, object]) -> str:
    start = int(field["start"]) - 1
    end = int(field["end"])
    return raw[start:end].decode("latin1").strip()


def is_missing(value: str, field: dict[str, object]) -> bool:
    return value == "" or value in field["missing_set"]


def parse_int(value: str, missing: bool) -> int | None:
    if missing:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_float(value: str, missing: bool) -> float | None:
    if missing:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def clean_feature(name: str, value: str, missing: bool):
    if name in NUMERIC_INT_FIELDS:
        return parse_int(value, missing)
    if name in NUMERIC_FLOAT_FIELDS:
        return parse_float(value, missing)
    if missing:
        return None
    return value


def yes(value: str) -> int:
    return int(value == "Y")


def derive_outcomes(values: dict[str, str]) -> dict[str, int | None]:
    gest_weeks = parse_int(values.get("COMBGEST", ""), values.get("COMBGEST") in {"", "99"})
    birth_weight = parse_int(values.get("DBWT", ""), values.get("DBWT") in {"", "9999"})
    apgar5 = parse_int(values.get("APGAR5", ""), values.get("APGAR5") in {"", "99"})

    preterm = None if gest_weeks is None else int(gest_weeks < 37)
    low_birthweight = None if birth_weight is None else int(birth_weight < 2500)
    very_low_birthweight = None if birth_weight is None else int(birth_weight < 1500)
    low_apgar5 = None if apgar5 is None else int(apgar5 < 7)

    maternal_transfusion = yes(values.get("MM_MTR", ""))
    perineal_laceration = yes(values.get("MM_PLAC", ""))
    ruptured_uterus = yes(values.get("MM_RUPT", ""))
    unplanned_hysterectomy = yes(values.get("MM_UHYST", ""))
    maternal_icu = yes(values.get("MM_AICU", ""))

    ventilation_gt6h = yes(values.get("AB_AVEN6", ""))
    nicu_admission = yes(values.get("AB_NICU", ""))
    newborn_seizures = yes(values.get("AB_SEIZ", ""))

    severe_neonatal_components = [
        very_low_birthweight,
        low_apgar5,
        ventilation_gt6h,
        newborn_seizures,
    ]
    broad_neonatal_components = [
        preterm,
        low_birthweight,
        low_apgar5,
        ventilation_gt6h,
        nicu_admission,
        newborn_seizures,
    ]

    return {
        "outcome_gest_weeks_combined": gest_weeks,
        "outcome_birthweight_g": birth_weight,
        "outcome_apgar5": apgar5,
        "outcome_preterm_lt37": preterm,
        "outcome_low_birthweight_lt2500g": low_birthweight,
        "outcome_very_low_birthweight_lt1500g": very_low_birthweight,
        "outcome_low_apgar5_lt7": low_apgar5,
        "outcome_ventilation_gt6h": ventilation_gt6h,
        "outcome_nicu_admission": nicu_admission,
        "outcome_newborn_seizures": newborn_seizures,
        "outcome_maternal_transfusion": maternal_transfusion,
        "outcome_perineal_laceration": perineal_laceration,
        "outcome_ruptured_uterus": ruptured_uterus,
        "outcome_unplanned_hysterectomy": unplanned_hysterectomy,
        "outcome_maternal_icu": maternal_icu,
        "outcome_maternal_morbidity_core": int(
            any(
                [
                    maternal_transfusion,
                    ruptured_uterus,
                    unplanned_hysterectomy,
                    maternal_icu,
                ]
            )
        ),
        "outcome_maternal_morbidity_extended": int(
            any(
                [
                    maternal_transfusion,
                    ruptured_uterus,
                    unplanned_hysterectomy,
                    maternal_icu,
                    perineal_laceration,
                ]
            )
        ),
        "outcome_severe_neonatal_no_nicu": (
            None
            if all(item is None for item in severe_neonatal_components)
            else int(any(item == 1 for item in severe_neonatal_components))
        ),
        "outcome_severe_neonatal_plus_nicu": (
            None
            if all(item is None for item in severe_neonatal_components)
            else int(any(item == 1 for item in severe_neonatal_components + [nicu_admission]))
        ),
        "outcome_broad_neonatal_composite": (
            None
            if all(item is None for item in broad_neonatal_components)
            else int(any(item == 1 for item in broad_neonatal_components))
        ),
    }


def make_schema(column_names: Iterable[str]) -> pa.Schema:
    fields = []
    for name in column_names:
        if name in {"source_year", "record_id"}:
            fields.append(pa.field(name, pa.int32()))
        elif name.startswith("missing_"):
            fields.append(pa.field(name, pa.bool_()))
        elif name.startswith("outcome_"):
            if name in {
                "outcome_gest_weeks_combined",
                "outcome_birthweight_g",
                "outcome_apgar5",
            }:
                fields.append(pa.field(name, pa.int16()))
            else:
                fields.append(pa.field(name, pa.int8()))
        elif name in {
            "input_MAGER",
            "input_PRECARE",
            "input_PREVIS",
            "input_CIG_0",
            "input_CIG_1",
            "input_CIG_2",
            "input_CIG_3",
            "input_M_Ht_In",
            "input_PWgt_R",
            "input_WTGAIN",
            "input_RF_CESARN",
            "context_DWgt_R",
        }:
            fields.append(pa.field(name, pa.int16()))
        elif name == "input_BMI":
            fields.append(pa.field(name, pa.float32()))
        else:
            fields.append(pa.field(name, pa.string()))
    return pa.schema(fields)


def write_chunk(
    rows: list[dict[str, object]],
    output_path: Path,
    writer: pq.ParquetWriter | None,
    schema: pa.Schema,
) -> pq.ParquetWriter:
    frame = pd.DataFrame.from_records(rows, columns=schema.names)
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    if writer is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(output_path, schema=schema, compression="zstd")
    writer.write_table(table)
    return writer


def summarize_outcomes(parquet_path: Path, table_dir: Path) -> None:
    dataset = pq.read_table(
        parquet_path,
        columns=[
            "outcome_maternal_morbidity_core",
            "outcome_maternal_morbidity_extended",
            "outcome_severe_neonatal_no_nicu",
            "outcome_severe_neonatal_plus_nicu",
            "outcome_broad_neonatal_composite",
            "outcome_nicu_admission",
            "outcome_preterm_lt37",
            "outcome_low_birthweight_lt2500g",
            "outcome_very_low_birthweight_lt1500g",
            "outcome_low_apgar5_lt7",
            "outcome_ventilation_gt6h",
            "outcome_newborn_seizures",
        ],
    ).to_pandas()
    rows = []
    for column in dataset.columns:
        known = dataset[column].notna()
        positive = int((dataset.loc[known, column] == 1).sum())
        known_n = int(known.sum())
        rows.append(
            {
                "endpoint": column,
                "n_total": int(len(dataset)),
                "known_n": known_n,
                "positive_n": positive,
                "positive_pct_known": round(100 * positive / known_n, 4)
                if known_n
                else None,
                "missing_n": int((~known).sum()),
            }
        )
    table_dir.mkdir(parents=True, exist_ok=True)
    year = str(parquet_path.name).split("_")[0].replace("nat", "")
    pd.DataFrame(rows).to_csv(
        table_dir / f"nat{year}_analytic_endpoint_prevalence.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--fields", type=Path, default=DEFAULT_FIELDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--sevenzip", type=Path, default=DEFAULT_7Z)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=500_000)
    args = parser.parse_args()

    if args.zip == DEFAULT_ZIP and args.year != 2024:
        args.zip = PROJECT_ROOT / "data" / "raw" / f"Nat{args.year}us.zip"
    if args.output == DEFAULT_OUTPUT and args.year != 2024:
        args.output = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / f"nat{args.year}_analytic_cohort.parquet"
        )

    fields = load_fields(args.fields)
    field_by_name = {str(field["name"]): field for field in fields}
    required_names = set(INPUT_FIELDS + CONTEXT_FIELDS) | {
        "COMBGEST",
        "DBWT",
        "APGAR5",
        "AB_AVEN6",
        "AB_NICU",
        "AB_SEIZ",
        "MM_MTR",
        "MM_PLAC",
        "MM_RUPT",
        "MM_UHYST",
        "MM_AICU",
    }
    required_fields = [field_by_name[name] for name in required_names]

    columns = ["source_year", "record_id"]
    columns.extend(f"input_{name}" for name in INPUT_FIELDS)
    columns.extend(f"context_{name}" for name in CONTEXT_FIELDS)
    columns.extend(f"missing_input_{name}" for name in INPUT_FIELDS)
    columns.extend(f"missing_context_{name}" for name in CONTEXT_FIELDS)
    outcome_names = [
        "outcome_gest_weeks_combined",
        "outcome_birthweight_g",
        "outcome_apgar5",
        "outcome_preterm_lt37",
        "outcome_low_birthweight_lt2500g",
        "outcome_very_low_birthweight_lt1500g",
        "outcome_low_apgar5_lt7",
        "outcome_ventilation_gt6h",
        "outcome_nicu_admission",
        "outcome_newborn_seizures",
        "outcome_maternal_transfusion",
        "outcome_perineal_laceration",
        "outcome_ruptured_uterus",
        "outcome_unplanned_hysterectomy",
        "outcome_maternal_icu",
        "outcome_maternal_morbidity_core",
        "outcome_maternal_morbidity_extended",
        "outcome_severe_neonatal_no_nicu",
        "outcome_severe_neonatal_plus_nicu",
        "outcome_broad_neonatal_composite",
    ]
    columns.extend(outcome_names)
    schema = make_schema(columns)

    started = time.time()
    rows: list[dict[str, object]] = []
    writer: pq.ParquetWriter | None = None
    n = 0
    missing_counts: Counter[str] = Counter()

    if args.output.exists():
        args.output.unlink()

    with open_record_stream(args.zip, args.sevenzip) as (fh, reader_name):
        for raw_line in fh:
            if not raw_line.strip():
                continue
            raw = raw_line.rstrip(b"\r\n")
            n += 1

            values = {str(field["name"]): raw_value(raw, field) for field in required_fields}
            row: dict[str, object] = {"source_year": args.year, "record_id": n}

            for name in INPUT_FIELDS:
                field = field_by_name[name]
                value = values[name]
                missing = is_missing(value, field)
                row[f"input_{name}"] = clean_feature(name, value, missing)
                row[f"missing_input_{name}"] = missing
                if missing:
                    missing_counts[f"input_{name}"] += 1

            for name in CONTEXT_FIELDS:
                field = field_by_name[name]
                value = values[name]
                missing = is_missing(value, field)
                row[f"context_{name}"] = clean_feature(name, value, missing)
                row[f"missing_context_{name}"] = missing
                if missing:
                    missing_counts[f"context_{name}"] += 1

            row.update(derive_outcomes(values))
            rows.append(row)

            if len(rows) >= args.chunk_size:
                writer = write_chunk(rows, args.output, writer, schema)
                rows.clear()

            if args.progress_every and n % args.progress_every == 0:
                elapsed = time.time() - started
                print(
                    f"processed {n:,} records in {elapsed:,.1f}s with {reader_name}",
                    flush=True,
                )

            if args.max_records and n >= args.max_records:
                break

    if rows:
        writer = write_chunk(rows, args.output, writer, schema)
        rows.clear()
    if writer is not None:
        writer.close()

    summarize_outcomes(args.output, args.tables)

    metadata = {
        "source_zip": str(args.zip),
        "output": str(args.output),
        "source_year": args.year,
        "records_processed": n,
        "reader": reader_name,
        "chunk_size": args.chunk_size,
        "max_records": args.max_records,
        "elapsed_seconds": round(time.time() - started, 3),
        "columns": columns,
        "missing_counts": dict(missing_counts),
    }
    args.tables.mkdir(parents=True, exist_ok=True)
    (args.tables / f"nat{args.year}_analytic_cohort_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"done: {n:,} records")
    print(f"wrote {args.output}")
    print(f"wrote {args.tables / f'nat{args.year}_analytic_endpoint_prevalence.csv'}")


if __name__ == "__main__":
    main()
