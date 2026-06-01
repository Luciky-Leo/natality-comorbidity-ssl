#!/usr/bin/env python
"""Cause-specific linked infant death enrichment by transferred SSL phenotype."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP = PROJECT_ROOT / "data" / "raw_linked_birth_infant_death" / "2024PE2023CO.zip"
DEFAULT_ASSIGNMENTS = PROJECT_ROOT / "results" / "objects" / "linked_infant_death_phenotype_assignments_full2016_2022_mask035_d48_l2_cuda.parquet"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "41_linked_infant_death_cause_profiles_report.md"

NUMERATOR_PREFIXES = [
    "VS2023LINK.Public.USNUMPUB",
    "VS2024LINK.Public.USNUMPUB",
]

CAUSE_ORDER = [
    "Perinatal conditions",
    "Congenital anomalies",
    "SIDS/ill-defined",
    "Infection/respiratory",
    "External injury",
    "Other causes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--linked-zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-tag", default="full2016_2022_mask035_d48_l2_cuda")
    parser.add_argument("--high-risk-phenotype", type=int, default=0)
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


def cause_group(ucod: str) -> str:
    code = (ucod or "").upper().strip()
    if not code:
        return "Other causes"
    first = code[0]
    if first == "P":
        return "Perinatal conditions"
    if first == "Q":
        return "Congenital anomalies"
    if code.startswith("R95") or first == "R":
        return "SIDS/ill-defined"
    if first in {"A", "B", "J"}:
        return "Infection/respiratory"
    if first in {"V", "W", "X", "Y"}:
        return "External injury"
    return "Other causes"


def load_death_causes(zip_path: Path) -> pd.DataFrame:
    rows = []
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
                    ucod = raw_slice(raw, 1368, 1371)
                    rows.append(
                        {
                            "linked_co_seqnum": key[0],
                            "linked_co_yod": key[1],
                            "linked_death_source": member,
                            "linked_age_at_death_days_parsed": age_at_death_days(raw),
                            "ucod_icd10": ucod,
                            "ucodr130": raw_slice(raw, 1373, 1375),
                            "cause_group": cause_group(ucod),
                        }
                    )
    return pd.DataFrame(rows)


def phenotype_cause_rates(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_n = len(frame)
    for phenotype, group in frame.groupby("phenotype", sort=True):
        for cause in CAUSE_ORDER:
            events = int((group["cause_group"] == cause).sum())
            total_events = int((frame["cause_group"] == cause).sum())
            rate = events / len(group) if len(group) else np.nan
            prevalence = total_events / total_n if total_n else np.nan
            rows.append(
                {
                    "phenotype": int(phenotype),
                    "cause_group": cause,
                    "n": int(len(group)),
                    "events": events,
                    "event_rate": float(rate),
                    "total_events": total_events,
                    "prevalence": float(prevalence),
                    "enrichment_over_prevalence": float(rate / prevalence) if prevalence else np.nan,
                    "event_capture_pct": float(100 * events / total_events) if total_events else np.nan,
                }
            )
    return pd.DataFrame(rows)


def high_risk_summary(rates: pd.DataFrame, high_risk: int) -> pd.DataFrame:
    return rates[rates["phenotype"].eq(high_risk)].sort_values("enrichment_over_prevalence", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_tag}" if args.output_tag else ""

    deaths = load_death_causes(args.linked_zip)
    assignments = pd.read_parquet(
        args.assignments,
        columns=["record_id", "linked_co_seqnum", "linked_co_yod", "outcome_infant_death", "phenotype"],
    )
    frame = assignments.merge(deaths, on=["linked_co_seqnum", "linked_co_yod"], how="left")
    frame["cause_group"] = np.where(frame["outcome_infant_death"].eq(1), frame["cause_group"].fillna("Other causes"), "")
    rates = phenotype_cause_rates(frame)
    high = high_risk_summary(rates, args.high_risk_phenotype)

    death_table_path = args.tables / f"linked_infant_death_cause_records{suffix}.csv"
    rates_path = args.tables / f"linked_infant_death_cause_rates{suffix}.csv"
    high_path = args.tables / f"linked_infant_death_cause_highrisk_enrichment{suffix}.csv"
    metadata_path = args.tables / f"linked_infant_death_cause_profiles_metadata{suffix}.json"
    deaths.to_csv(death_table_path, index=False)
    rates.to_csv(rates_path, index=False)
    high.to_csv(high_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "linked_zip": str(args.linked_zip),
                "assignments": str(args.assignments),
                "parsed_deaths": int(len(deaths)),
                "assignment_rows": int(len(assignments)),
                "high_risk_phenotype": int(args.high_risk_phenotype),
                "ucod_position": "1368-1371",
                "ucodr130_position": "1373-1375",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Linked Infant Death Cause-Specific Phenotype Profiles",
        "",
        "Underlying cause of death was parsed from UCOD ICD-10 positions 1368-1371 in the CDC/NCHS linked birth/infant death numerator records. Cause-specific enrichment is descriptive and grouped into broad ICD-10 categories.",
        "",
        f"- parsed infant death records: {len(deaths):,}",
        f"- linked assignment rows: {len(assignments):,}",
        "",
        "## High-Risk Phenotype Cause-Specific Enrichment",
        "",
        "| Cause group | Events | Rate % | Baseline % | Enrichment | Event capture % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in high.to_dict("records"):
        lines.append(
            "| {cause_group} | {events:,} | {rate:.4f} | {prev:.4f} | {enrich:.2f} | {capture:.2f} |".format(
                cause_group=row["cause_group"],
                events=int(row["events"]),
                rate=100 * float(row["event_rate"]),
                prev=100 * float(row["prevalence"]),
                enrich=float(row["enrichment_over_prevalence"]),
                capture=float(row["event_capture_pct"]),
            )
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{death_table_path}`",
            f"- `{rates_path}`",
            f"- `{high_path}`",
            f"- `{metadata_path}`",
        ]
    )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
