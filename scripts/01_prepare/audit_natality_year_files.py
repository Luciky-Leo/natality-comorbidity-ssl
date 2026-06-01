#!/usr/bin/env python
"""Audit whether the 2024 field map works across 2016-2024 Natality files."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "natality_download_manifest.csv"
DEFAULT_FIELDS = PROJECT_ROOT / "config" / "nat2024_smoke_fields.csv"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "09_multiyear_harmonization_audit.md"
DEFAULT_7Z = Path(r"C:\Program Files\NVIDIA Corporation\NVIDIA app\7z.exe")


KEY_FIELD_NAMES = [
    "DOB_YY",
    "MAGER",
    "MRACEHISP",
    "MEDUC",
    "BMI",
    "RF_PDIAB",
    "RF_GDIAB",
    "RF_PHYPE",
    "RF_GHYPE",
    "RF_INFTR",
    "RF_FEDRG",
    "RF_ARTEC",
    "MM_MTR",
    "MM_RUPT",
    "MM_UHYST",
    "MM_AICU",
    "APGAR5",
    "COMBGEST",
    "DBWT",
    "AB_AVEN6",
    "AB_NICU",
    "AB_SEIZ",
]


def load_fields(path: Path) -> dict[str, dict[str, object]]:
    fields = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            row["start"] = int(row["start"])
            row["end"] = int(row["end"])
            fields[row["name"]] = row
    return fields


def load_manifest(path: Path, years: set[int]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = [row for row in csv.DictReader(fh) if int(row["year"]) in years]
    return sorted(rows, key=lambda row: int(row["year"]))


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
                code = proc.wait()
                if code != 0:
                    raise RuntimeError(f"7-Zip exited with status {code}")
        return
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise RuntimeError(f"Expected one member in {zip_path}, got {names}")
        with zf.open(names[0], "r") as fh:
            yield fh, "python_zipfile"


def raw_value(raw: bytes, field: dict[str, object]) -> str:
    start = int(field["start"]) - 1
    end = int(field["end"])
    return raw[start:end].decode("latin1").strip()


def parse_int(value: str, missing_values: set[str]) -> int | None:
    if value == "" or value in missing_values:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def yes(value: str) -> int:
    return int(value == "Y")


def derive_flags(values: dict[str, str]) -> dict[str, int | None]:
    gest = parse_int(values.get("COMBGEST", ""), {"99"})
    bw = parse_int(values.get("DBWT", ""), {"9999"})
    apgar = parse_int(values.get("APGAR5", ""), {"99"})
    preterm = None if gest is None else int(gest < 37)
    lbw = None if bw is None else int(bw < 2500)
    vlbw = None if bw is None else int(bw < 1500)
    low_apgar = None if apgar is None else int(apgar < 7)

    maternal_core = int(
        any(
            [
                yes(values.get("MM_MTR", "")),
                yes(values.get("MM_RUPT", "")),
                yes(values.get("MM_UHYST", "")),
                yes(values.get("MM_AICU", "")),
            ]
        )
    )
    severe_neonatal_parts = [
        vlbw,
        low_apgar,
        yes(values.get("AB_AVEN6", "")),
        yes(values.get("AB_SEIZ", "")),
    ]
    severe_neonatal = (
        None
        if all(part is None for part in severe_neonatal_parts)
        else int(any(part == 1 for part in severe_neonatal_parts))
    )
    return {
        "preterm_lt37": preterm,
        "low_birthweight_lt2500g": lbw,
        "very_low_birthweight_lt1500g": vlbw,
        "low_apgar5_lt7": low_apgar,
        "nicu_admission": yes(values.get("AB_NICU", "")),
        "maternal_morbidity_core": maternal_core,
        "severe_neonatal_no_nicu": severe_neonatal,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fields", type=Path, default=DEFAULT_FIELDS)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sevenzip", type=Path, default=DEFAULT_7Z)
    parser.add_argument("--years", nargs="*", type=int, default=list(range(2016, 2025)))
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    args = parser.parse_args()

    fields = load_fields(args.fields)
    key_fields = {name: fields[name] for name in KEY_FIELD_NAMES}
    rows = load_manifest(args.manifest, set(args.years))

    audit_rows = []
    prevalence_rows = []
    value_rows = []
    started_all = time.time()

    for row in rows:
        year = int(row["year"])
        zip_path = args.raw_dir / row["us_data_file"]
        if not zip_path.exists():
            raise FileNotFoundError(zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zip_entries = zf.namelist()
            entry_size = zf.infolist()[0].file_size if zf.infolist() else None

        n = 0
        line_lengths: Counter[int] = Counter()
        key_counts: dict[str, Counter[str]] = {name: Counter() for name in KEY_FIELD_NAMES}
        outcome_counts: dict[str, Counter[int | None]] = defaultdict(Counter)
        first_values: dict[str, str] | None = None
        started = time.time()

        with open_record_stream(zip_path, args.sevenzip) as (fh, reader):
            for raw_line in fh:
                if not raw_line.strip():
                    continue
                raw = raw_line.rstrip(b"\r\n")
                n += 1
                line_lengths[len(raw)] += 1
                values = {name: raw_value(raw, field) for name, field in key_fields.items()}
                if first_values is None:
                    first_values = dict(values)
                for name, value in values.items():
                    key_counts[name][value] += 1
                for name, flag in derive_flags(values).items():
                    outcome_counts[name][flag] += 1

                if args.progress_every and n % args.progress_every == 0:
                    print(
                        f"{year}: processed {n:,} records in {time.time() - started:,.1f}s",
                        flush=True,
                    )
                if args.max_records and n >= args.max_records:
                    break

        dob_top = key_counts["DOB_YY"].most_common(3)
        audit_rows.append(
            {
                "year": year,
                "records": n,
                "zip_file": row["us_data_file"],
                "zip_entry": ";".join(zip_entries),
                "entry_uncompressed_bytes": entry_size,
                "reader": reader,
                "line_lengths": json.dumps(dict(line_lengths), sort_keys=True),
                "dob_top_values": "; ".join(f"{value}:{count}" for value, count in dob_top),
                "first_record_dob_yy": first_values.get("DOB_YY") if first_values else "",
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )

        for endpoint, counts in sorted(outcome_counts.items()):
            known_n = counts[0] + counts[1]
            prevalence_rows.append(
                {
                    "year": year,
                    "endpoint": endpoint,
                    "n_total": n,
                    "known_n": known_n,
                    "positive_n": counts[1],
                    "positive_pct_known": round(100 * counts[1] / known_n, 4)
                    if known_n
                    else None,
                    "missing_n": counts[None],
                }
            )

        for field_name in KEY_FIELD_NAMES:
            value_rows.append(
                {
                    "year": year,
                    "field": field_name,
                    "top_values": "; ".join(
                        f"{repr(value)}:{count}"
                        for value, count in key_counts[field_name].most_common(8)
                    ),
                }
            )

    args.tables.mkdir(parents=True, exist_ok=True)
    audit_path = args.tables / "natality_2016_2024_file_audit.csv"
    prevalence_path = args.tables / "natality_2016_2024_endpoint_prevalence_audit.csv"
    values_path = args.tables / "natality_2016_2024_key_value_audit.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)
    with prevalence_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(prevalence_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prevalence_rows)
    with values_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(value_rows[0].keys()))
        writer.writeheader()
        writer.writerows(value_rows)

    report_lines = [
        "# Multi-Year Natality Harmonization Audit",
        "",
        f"Years: {min(args.years)}-{max(args.years)}",
        f"Total elapsed seconds: {round(time.time() - started_all, 1)}",
        "",
        "## File Audit",
        "",
        "| Year | Records | Line lengths | DOB top values |",
        "|---:|---:|---|---|",
    ]
    for item in audit_rows:
        report_lines.append(
            f"| {item['year']} | {item['records']:,} | `{item['line_lengths']}` | {item['dob_top_values']} |"
        )
    report_lines.extend(
        [
            "",
            "## Endpoint Prevalence Audit",
            "",
            "| Year | Endpoint | Positive n | Known n | Positive % | Missing n |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for item in prevalence_rows:
        if item["endpoint"] in {
            "maternal_morbidity_core",
            "severe_neonatal_no_nicu",
            "nicu_admission",
            "preterm_lt37",
            "low_birthweight_lt2500g",
        }:
            report_lines.append(
                "| {year} | {endpoint} | {positive_n:,} | {known_n:,} | {positive_pct_known:.4f} | {missing_n:,} |".format(
                    **item
                )
            )
    report_lines.extend(
        [
            "",
            "## Output Tables",
            "",
            f"- `{audit_path}`",
            f"- `{prevalence_path}`",
            f"- `{values_path}`",
            "",
            "## Interpretation",
            "",
            "This audit tests whether the selected 2024 field positions recover plausible key fields across 2016-2024. It does not replace year-specific user-guide verification for final manuscript reproducibility.",
        ]
    )
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"wrote {audit_path}")
    print(f"wrote {prevalence_path}")
    print(f"wrote {values_path}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
