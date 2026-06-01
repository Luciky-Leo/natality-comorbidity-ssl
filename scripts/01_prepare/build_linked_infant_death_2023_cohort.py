#!/usr/bin/env python
"""Build a 2023 linked birth/infant death cohort from CDC/NCHS public files."""

from __future__ import annotations

import argparse
import csv
import json
import time
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from build_analytic_cohort_2024 import (
    CONTEXT_FIELDS,
    INPUT_FIELDS,
    clean_feature,
    is_missing,
    load_fields,
    raw_value,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP = PROJECT_ROOT / "data" / "raw_linked_birth_infant_death" / "2024PE2023CO.zip"
DEFAULT_FIELDS = PROJECT_ROOT / "config" / "nat2024_smoke_fields.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "linked_infant_death_2023_cohort.parquet"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"

DENOM_NAME_PREFIX = "VS2023LINK.Public.USDENPUB"
NUMERATOR_PREFIXES = [
    "VS2023LINK.Public.USNUMPUB",
    "VS2024LINK.Public.USNUMPUB",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--fields", type=Path, default=DEFAULT_FIELDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--progress-every", type=int, default=500_000)
    return parser.parse_args()


def raw_slice(raw: bytes, start: int, end: int) -> str:
    return raw[start - 1 : end].decode("latin1").strip()


def linked_key(raw: bytes) -> tuple[str, str]:
    return raw_slice(raw, 365, 371), raw_slice(raw, 372, 375)


def age_at_death_days(raw: bytes) -> int | None:
    value = raw_slice(raw, 1356, 1358)
    try:
        return int(value)
    except ValueError:
        return None


def find_member(names: list[str], prefix: str) -> str:
    matches = [name for name in names if name.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one member with prefix {prefix}, got {matches}")
    return matches[0]


def load_death_records(zip_path: Path) -> dict[tuple[str, str], dict[str, object]]:
    deaths: dict[tuple[str, str], dict[str, object]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for prefix in NUMERATOR_PREFIXES:
            member = find_member(names, prefix)
            with zf.open(member, "r") as fh:
                for raw in fh:
                    if not raw.strip():
                        continue
                    dob_year = raw_slice(raw, 9, 12)
                    re_status = raw_slice(raw, 104, 104)
                    if dob_year != "2023":
                        continue
                    if re_status.isdigit() and int(re_status) >= 4:
                        continue
                    key = linked_key(raw)
                    aged = age_at_death_days(raw)
                    deaths[key] = {
                        "death_source_file": member,
                        "death_year": raw_slice(raw, 372, 375),
                        "age_at_death_days": aged,
                        "neonatal_death_lt28d": int(aged is not None and aged < 28),
                        "early_neonatal_death_lt7d": int(aged is not None and aged < 7),
                        "postneonatal_death_28d_1y": int(aged is not None and aged >= 28),
                    }
    return deaths


def build_row(
    raw: bytes,
    fields_by_name: dict[str, dict[str, object]],
    deaths: dict[tuple[str, str], dict[str, object]],
    record_id: int,
) -> dict[str, object]:
    key = linked_key(raw)
    death = deaths.get(key)
    row: dict[str, object] = {
        "source_year": 2023,
        "record_id": record_id,
        "linked_co_seqnum": key[0],
        "linked_co_yod": key[1],
    }
    for name in INPUT_FIELDS:
        field = fields_by_name[name]
        value = raw_value(raw, field)
        missing = is_missing(value, field)
        row[f"input_{name}"] = clean_feature(name, value, missing)
        row[f"missing_input_{name}"] = bool(missing)
    for name in CONTEXT_FIELDS:
        field = fields_by_name[name]
        value = raw_value(raw, field)
        missing = is_missing(value, field)
        row[f"context_{name}"] = clean_feature(name, value, missing)
        row[f"missing_context_{name}"] = bool(missing)

    row["outcome_infant_death"] = int(death is not None)
    row["outcome_neonatal_death_lt28d"] = int(death["neonatal_death_lt28d"]) if death else 0
    row["outcome_early_neonatal_death_lt7d"] = int(death["early_neonatal_death_lt7d"]) if death else 0
    row["outcome_postneonatal_death_28d_1y"] = int(death["postneonatal_death_28d_1y"]) if death else 0
    row["linked_age_at_death_days"] = death["age_at_death_days"] if death else None
    row["linked_death_year"] = death["death_year"] if death else None
    return row


def write_manifest(output: Path, tables: Path, rows: int, deaths: int, elapsed: float) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    prevalence_path = tables / "linked_infant_death_2023_outcome_prevalence.csv"
    metadata_path = tables / "linked_infant_death_2023_metadata.json"
    with prevalence_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["outcome", "positive_n", "total_n", "prevalence"])
        writer.writeheader()
        writer.writerow(
            {
                "outcome": "outcome_infant_death",
                "positive_n": deaths,
                "total_n": rows,
                "prevalence": deaths / rows if rows else 0.0,
            }
        )
    metadata_path.write_text(
        json.dumps(
            {
                "output": str(output),
                "rows": rows,
                "infant_deaths": deaths,
                "infant_death_prevalence": deaths / rows if rows else 0.0,
                "elapsed_seconds": elapsed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    fields = load_fields(args.fields)
    fields_by_name = {str(field["name"]): field for field in fields}

    start = time.time()
    print("load linked numerator death records", flush=True)
    deaths = load_death_records(args.zip)
    print(f"linked infant deaths for 2023 births: {len(deaths):,}", flush=True)

    rows: list[dict[str, object]] = []
    total = 0
    death_total = 0
    writer: pq.ParquetWriter | None = None
    with zipfile.ZipFile(args.zip) as zf:
        member = find_member(zf.namelist(), DENOM_NAME_PREFIX)
        with zf.open(member, "r") as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                re_status = raw_slice(raw, 104, 104)
                if re_status.isdigit() and int(re_status) >= 4:
                    continue
                total += 1
                row = build_row(raw, fields_by_name, deaths, total)
                death_total += int(row["outcome_infant_death"])
                rows.append(row)
                if len(rows) >= args.chunk_size:
                    table = pa.Table.from_pylist(rows)
                    if writer is None:
                        writer = pq.ParquetWriter(args.output, table.schema, compression="zstd")
                    else:
                        table = table.cast(writer.schema)
                    writer.write_table(table)
                    rows.clear()
                if total % args.progress_every == 0:
                    print(f"processed {total:,} denominator births; deaths={death_total:,}", flush=True)
    if rows:
        table = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(args.output, table.schema, compression="zstd")
        else:
            table = table.cast(writer.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()
    elapsed = time.time() - start
    write_manifest(args.output, args.tables, total, death_total, elapsed)
    print(f"wrote {args.output}")
    print(f"rows={total:,}, infant_deaths={death_total:,}, elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
