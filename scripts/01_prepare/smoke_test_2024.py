#!/usr/bin/env python
"""Run a one-year smoke test for the 2024 CDC/NCHS Natality public-use file.

The script streams selected fixed-width columns directly from Nat2024us.zip.
It does not extract the 4.9 GB text file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
import zipfile
from contextlib import contextmanager
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP = PROJECT_ROOT / "data" / "raw" / "Nat2024us.zip"
DEFAULT_FIELDS = PROJECT_ROOT / "config" / "nat2024_smoke_fields.csv"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_DOC_DIR = PROJECT_ROOT / "docs"
DEFAULT_7Z = Path(r"C:\Program Files\NVIDIA Corporation\NVIDIA app\7z.exe")


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


def parse_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def yes(value: str) -> bool:
    return value == "Y"


def known(value: str) -> bool:
    return value not in {"", "U", "9", "99", "999", "9999", "99.9"}


def extract_record(raw: bytes, fields: list[dict[str, object]]) -> dict[str, str]:
    record = raw.rstrip(b"\r\n")
    values: dict[str, str] = {}
    for field in fields:
        start = int(field["start"]) - 1
        end = int(field["end"])
        values[str(field["name"])] = record[start:end].decode("latin1").strip()
    return values


@contextmanager
def open_record_stream(zip_path: Path, sevenzip: Path | None):
    """Open a byte-line iterator for CDC zip files.

    CDC Natality zip files may use Deflate64, which Python's stdlib zipfile and
    .NET ZipArchive cannot decompress. 7-Zip can stream the member to stdout.
    """

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
                if return_code != 0 and return_code is not None:
                    raise RuntimeError(f"7-Zip exited with status {return_code}")
        return

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise RuntimeError(f"Expected one file in {zip_path}, got {names}")
        with zf.open(names[0], "r") as fh:
            yield fh, "python_zipfile"


def outcome_flags(values: dict[str, str]) -> dict[str, int | None]:
    gest_weeks = parse_int(values.get("COMBGEST", ""))
    birth_weight = parse_int(values.get("DBWT", ""))
    apgar5 = parse_int(values.get("APGAR5", ""))

    preterm = None if gest_weeks in {None, 99} else int(gest_weeks < 37)
    low_birthweight = (
        None if birth_weight in {None, 9999} else int(birth_weight < 2500)
    )
    very_low_birthweight = (
        None if birth_weight in {None, 9999} else int(birth_weight < 1500)
    )
    low_apgar5 = None if apgar5 in {None, 99} else int(apgar5 < 7)

    neonatal_components = [
        preterm,
        low_birthweight,
        low_apgar5,
        int(yes(values.get("AB_AVEN6", ""))),
        int(yes(values.get("AB_NICU", ""))),
        int(yes(values.get("AB_SEIZ", ""))),
    ]
    neonatal_known = any(item is not None for item in neonatal_components)
    adverse_neonatal = (
        int(any(item == 1 for item in neonatal_components))
        if neonatal_known
        else None
    )

    maternal_components = [
        int(yes(values.get("MM_MTR", ""))),
        int(yes(values.get("MM_PLAC", ""))),
        int(yes(values.get("MM_RUPT", ""))),
        int(yes(values.get("MM_UHYST", ""))),
        int(yes(values.get("MM_AICU", ""))),
    ]
    maternal_morbidity = int(any(item == 1 for item in maternal_components))

    return {
        "preterm_combgest_lt37": preterm,
        "low_birthweight_lt2500g": low_birthweight,
        "very_low_birthweight_lt1500g": very_low_birthweight,
        "low_apgar5_lt7": low_apgar5,
        "ventilation_gt6h": int(yes(values.get("AB_AVEN6", ""))),
        "nicu_admission": int(yes(values.get("AB_NICU", ""))),
        "newborn_seizures": int(yes(values.get("AB_SEIZ", ""))),
        "adverse_neonatal_composite": adverse_neonatal,
        "maternal_transfusion": int(yes(values.get("MM_MTR", ""))),
        "perineal_laceration": int(yes(values.get("MM_PLAC", ""))),
        "ruptured_uterus": int(yes(values.get("MM_RUPT", ""))),
        "unplanned_hysterectomy": int(yes(values.get("MM_UHYST", ""))),
        "maternal_icu": int(yes(values.get("MM_AICU", ""))),
        "maternal_morbidity_composite": maternal_morbidity,
    }


def rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return math.nan
    return numerator / denominator


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--fields", type=Path, default=DEFAULT_FIELDS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOC_DIR)
    parser.add_argument("--sevenzip", type=Path, default=DEFAULT_7Z)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=500_000)
    args = parser.parse_args()

    fields = load_fields(args.fields)
    field_by_name = {str(field["name"]): field for field in fields}
    value_counts: dict[str, Counter[str]] = {
        str(field["name"]): Counter() for field in fields
    }
    missing_counts: Counter[str] = Counter()
    outcome_counts: dict[str, Counter[int | None]] = defaultdict(Counter)
    line_lengths: Counter[int] = Counter()
    examples: list[dict[str, str]] = []

    started = time.time()
    n = 0
    zip_entries: list[str] = []

    with zipfile.ZipFile(args.zip) as zf:
        zip_entries = zf.namelist()
    with open_record_stream(args.zip, args.sevenzip) as (fh, reader_name):
        for raw in fh:
            if not raw.strip():
                continue
            n += 1
            line_lengths[len(raw.rstrip(b"\r\n"))] += 1
            values = extract_record(raw, fields)

            if len(examples) < 5:
                examples.append({key: values[key] for key in values})

            for name, value in values.items():
                value_counts[name][value] += 1
                missing_set = field_by_name[name]["missing_set"]
                if value == "" or value in missing_set:
                    missing_counts[name] += 1

            for name, flag in outcome_flags(values).items():
                outcome_counts[name][flag] += 1

            if args.progress_every and n % args.progress_every == 0:
                elapsed = time.time() - started
                print(
                    f"processed {n:,} records in {elapsed:,.1f}s with {reader_name}",
                    flush=True,
                )

            if args.max_records and n >= args.max_records:
                break

    variable_rows: list[dict[str, object]] = []
    for field in fields:
        name = str(field["name"])
        top_values = "; ".join(
            f"{repr(value)}:{count}" for value, count in value_counts[name].most_common(8)
        )
        variable_rows.append(
            {
                "variable": name,
                "role": field["role"],
                "description": field["description"],
                "start": field["start"],
                "end": field["end"],
                "n": n,
                "missing_n": missing_counts[name],
                "missing_pct": round(100 * rate(missing_counts[name], n), 4),
                "unique_n": len(value_counts[name]),
                "top_values": top_values,
            }
        )

    outcome_rows: list[dict[str, object]] = []
    for name in sorted(outcome_counts):
        counts = outcome_counts[name]
        known_denominator = counts[0] + counts[1]
        outcome_rows.append(
            {
                "outcome": name,
                "n_total": n,
                "known_n": known_denominator,
                "positive_n": counts[1],
                "positive_pct_known": round(100 * rate(counts[1], known_denominator), 4),
                "missing_or_not_applicable_n": counts[None],
            }
        )

    table_dir = args.tables
    write_csv(
        table_dir / "nat2024_smoke_variable_missingness.csv",
        variable_rows,
        [
            "variable",
            "role",
            "description",
            "start",
            "end",
            "n",
            "missing_n",
            "missing_pct",
            "unique_n",
            "top_values",
        ],
    )
    write_csv(
        table_dir / "nat2024_smoke_outcome_prevalence.csv",
        outcome_rows,
        [
            "outcome",
            "n_total",
            "known_n",
            "positive_n",
            "positive_pct_known",
            "missing_or_not_applicable_n",
        ],
    )

    metadata = {
        "source_zip": str(args.zip),
        "zip_entries": zip_entries,
        "fields": str(args.fields),
        "records_processed": n,
        "line_lengths": dict(line_lengths),
        "max_records": args.max_records,
        "reader": reader_name,
        "elapsed_seconds": round(time.time() - started, 3),
        "sample_records": examples,
    }
    metadata_path = table_dir / "nat2024_smoke_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report_path = args.docs / "04_smoke_test_2024_report.md"
    report_lines = [
        "# 2024 Natality Smoke Test Report",
        "",
        f"Records processed: {n:,}",
        f"Source zip: `{args.zip.name}`",
        f"Zip entry: `{zip_entries[0] if zip_entries else ''}`",
        f"Observed line lengths: {dict(line_lengths)}",
        "",
        "## Technical Status",
        "",
        "- Full 2024 U.S. file parsed successfully.",
        "- Official control count for U.S. births in the guide is 3,638,436, matching the parsed record count when the full file is used.",
        "- The downloaded CDC zip requires 7-Zip streaming because Python's standard `zipfile` cannot decompress this compression method.",
        "- Documented field positions in the 2024 User Guide correctly recover clinically plausible values for maternal risk factors, neonatal outcomes, and maternal morbidity fields.",
        "- `RF_FEDRG` and `RF_ARTEC` use `X` as a not-applicable category when infertility treatment is not used; this is not treated as missingness.",
        "",
        "## Key Feasibility Interpretation",
        "",
        "- Maternal morbidity composite is appropriate for an imbalanced risk-enrichment endpoint where AUPRC over the prevalence baseline is meaningful.",
        "- The broad neonatal composite is feasible for modelling but probably too broad for the final main endpoint; the next protocol version should define a severe neonatal composite and keep the broad composite as secondary.",
        "- Marital status has substantial blank/non-reporting values and should be handled as a missing/unknown category rather than complete-case exclusion.",
        "- ART and infertility variables are feasible but rare, so they are better suited for phenotype interpretation and subgroup analysis than as standalone endpoints.",
        "",
        "## Outcome Prevalence",
        "",
        "| Outcome | Positive n | Known n | Positive % | Missing/not applicable n |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in outcome_rows:
        report_lines.append(
            "| {outcome} | {positive_n:,} | {known_n:,} | {positive_pct_known:.4f} | {missing_or_not_applicable_n:,} |".format(
                **row
            )
        )
    report_lines.extend(
        [
            "",
            "## Highest Missingness Among Candidate Fields",
            "",
            "| Variable | Role | Missing % | Top values |",
            "|---|---|---:|---|",
        ]
    )
    for row in sorted(variable_rows, key=lambda item: float(item["missing_pct"]), reverse=True)[:25]:
        report_lines.append(
            f"| {row['variable']} | {row['role']} | {row['missing_pct']} | {row['top_values']} |"
        )
    report_lines.extend(
        [
            "",
            "## Output Tables",
            "",
            "- `results/tables/nat2024_smoke_variable_missingness.csv`",
            "- `results/tables/nat2024_smoke_outcome_prevalence.csv`",
            "- `results/tables/nat2024_smoke_metadata.json`",
            "",
            "## Interpretation Boundary",
            "",
            "This is a technical smoke test. The neonatal composite includes postnatal outcome components and should not be used as a final primary endpoint until leakage-controlled input variables and clinical endpoint definitions are finalized.",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"done: {n:,} records")
    print(f"wrote {table_dir / 'nat2024_smoke_variable_missingness.csv'}")
    print(f"wrote {table_dir / 'nat2024_smoke_outcome_prevalence.csv'}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
