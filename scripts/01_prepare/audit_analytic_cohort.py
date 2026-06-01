#!/usr/bin/env python
"""Audit the generated analytic cohort and write manuscript-ready summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COHORT = PROJECT_ROOT / "data" / "processed" / "nat2024_analytic_cohort.parquet"
DEFAULT_ENDPOINTS = PROJECT_ROOT / "results" / "tables" / "nat2024_analytic_endpoint_prevalence.csv"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_DOC = PROJECT_ROOT / "docs" / "07_analytic_cohort_report.md"


def column_group(column: str) -> str:
    if column.startswith("input_"):
        return "input"
    if column.startswith("context_"):
        return "context"
    if column.startswith("missing_input_"):
        return "input_missingness_flag"
    if column.startswith("missing_context_"):
        return "context_missingness_flag"
    if column.startswith("outcome_"):
        return "outcome"
    if column in {"source_year", "record_id"}:
        return "metadata"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--endpoints", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()

    parquet_file = pq.ParquetFile(args.cohort)
    n_rows = parquet_file.metadata.num_rows
    columns = parquet_file.schema.names

    dictionary_rows = [
        {
            "column": column,
            "group": column_group(column),
            "source_variable": column.removeprefix("input_")
            .removeprefix("context_")
            .removeprefix("missing_input_")
            .removeprefix("missing_context_"),
        }
        for column in columns
    ]
    dictionary = pd.DataFrame(dictionary_rows)
    dictionary.to_csv(args.tables / "nat2024_analytic_column_dictionary.csv", index=False)

    missing_columns = [
        column
        for column in columns
        if column.startswith("missing_input_") or column.startswith("missing_context_")
    ]
    missing_table = pq.read_table(args.cohort, columns=missing_columns).to_pandas()
    missing_rows = []
    for column in missing_columns:
        missing_n = int(missing_table[column].sum())
        missing_rows.append(
            {
                "variable": column.replace("missing_input_", "").replace(
                    "missing_context_", ""
                ),
                "role": "input"
                if column.startswith("missing_input_")
                else "context",
                "missing_n": missing_n,
                "missing_pct": round(100 * missing_n / n_rows, 4),
            }
        )
    missing_summary = pd.DataFrame(missing_rows).sort_values(
        ["missing_pct", "variable"], ascending=[False, True]
    )
    missing_summary.to_csv(
        args.tables / "nat2024_analytic_input_missingness.csv", index=False
    )

    endpoint_summary = pd.read_csv(args.endpoints)

    report_lines = [
        "# 2024 Analytic Cohort Report",
        "",
        f"Analytic cohort file: `{args.cohort}`",
        f"Rows: {n_rows:,}",
        f"Columns: {len(columns):,}",
        f"Row groups: {parquet_file.metadata.num_row_groups:,}",
        "",
        "## Column Groups",
        "",
        "| Group | Columns |",
        "|---|---:|",
    ]
    for group, count in dictionary["group"].value_counts().sort_index().items():
        report_lines.append(f"| {group} | {count} |")

    report_lines.extend(
        [
            "",
            "## Endpoint Prevalence",
            "",
            "| Endpoint | Positive n | Known n | Positive % | Missing n |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in endpoint_summary.to_dict("records"):
        report_lines.append(
            "| {endpoint} | {positive_n:,} | {known_n:,} | {positive_pct_known:.4f} | {missing_n:,} |".format(
                **row
            )
        )

    report_lines.extend(
        [
            "",
            "## Highest Input Missingness",
            "",
            "| Variable | Role | Missing n | Missing % |",
            "|---|---|---:|---:|",
        ]
    )
    for row in missing_summary.head(20).to_dict("records"):
        report_lines.append(
            "| {variable} | {role} | {missing_n:,} | {missing_pct:.4f} |".format(
                **row
            )
        )

    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The primary maternal endpoint should use `outcome_maternal_morbidity_core`.",
            "- The extended maternal endpoint should be kept as sensitivity analysis.",
            "- The primary neonatal endpoint should use `outcome_severe_neonatal_no_nicu`.",
            "- NICU and broad neonatal composites should be secondary or sensitivity endpoints.",
            "- Outcome-derived variables must not be used as primary model inputs.",
            "",
            "## Output Tables",
            "",
            "- `results/tables/nat2024_analytic_endpoint_prevalence.csv`",
            "- `results/tables/nat2024_analytic_input_missingness.csv`",
            "- `results/tables/nat2024_analytic_column_dictionary.csv`",
        ]
    )
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"wrote {args.tables / 'nat2024_analytic_input_missingness.csv'}")
    print(f"wrote {args.tables / 'nat2024_analytic_column_dictionary.csv'}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
