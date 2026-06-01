#!/usr/bin/env python
"""Summarize generated multi-year analytic cohorts and write split manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED = PROJECT_ROOT / "data" / "processed"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "10_multiyear_analytic_cohort_report.md"


def split_role(year: int) -> str:
    if 2016 <= year <= 2022:
        return "train"
    if year == 2023:
        return "development"
    if year == 2024:
        return "test"
    return "unused"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--years", nargs="*", type=int, default=list(range(2016, 2025)))
    args = parser.parse_args()

    manifest_rows = []
    prevalence_frames = []
    for year in args.years:
        path = args.processed / f"nat{year}_analytic_cohort.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        parquet_file = pq.ParquetFile(path)
        manifest_rows.append(
            {
                "year": year,
                "split": split_role(year),
                "path": str(path),
                "rows": parquet_file.metadata.num_rows,
                "columns": parquet_file.metadata.num_columns,
                "row_groups": parquet_file.metadata.num_row_groups,
                "file_size_bytes": path.stat().st_size,
            }
        )

        prevalence_path = args.tables / f"nat{year}_analytic_endpoint_prevalence.csv"
        if prevalence_path.exists():
            frame = pd.read_csv(prevalence_path)
            frame.insert(0, "year", year)
            frame.insert(1, "split", split_role(year))
            prevalence_frames.append(frame)

    args.tables.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = args.tables / "natality_2016_2024_analytic_manifest.csv"
    split_manifest_path = args.processed / "natality_2016_2024_split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    manifest[["year", "split", "path", "rows"]].to_csv(split_manifest_path, index=False)

    prevalence = pd.concat(prevalence_frames, ignore_index=True)
    prevalence_path = args.tables / "natality_2016_2024_analytic_endpoint_prevalence.csv"
    prevalence.to_csv(prevalence_path, index=False)

    split_summary = (
        manifest.groupby("split", as_index=False)
        .agg(years=("year", lambda values: ",".join(str(v) for v in values)), rows=("rows", "sum"))
        .sort_values("split")
    )

    report_lines = [
        "# Multi-Year Analytic Cohort Report",
        "",
        "Generated analytic cohorts for 2016-2024 CDC/NCHS Natality public-use files.",
        "",
        "## Cohort Files",
        "",
        "| Year | Split | Rows | Columns | Size MB |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in manifest.to_dict("records"):
        report_lines.append(
            "| {year} | {split} | {rows:,} | {columns:,} | {size:.1f} |".format(
                year=row["year"],
                split=row["split"],
                rows=row["rows"],
                columns=row["columns"],
                size=row["file_size_bytes"] / (1024 * 1024),
            )
        )

    report_lines.extend(
        [
            "",
            "## Split Summary",
            "",
            "| Split | Years | Rows |",
            "|---|---|---:|",
        ]
    )
    for row in split_summary.to_dict("records"):
        report_lines.append(f"| {row['split']} | {row['years']} | {row['rows']:,} |")

    report_lines.extend(
        [
            "",
            "## Primary Endpoint Prevalence",
            "",
            "| Year | Split | Maternal core morbidity % | Severe neonatal no NICU % |",
            "|---:|---|---:|---:|",
        ]
    )
    for year in args.years:
        sub = prevalence[prevalence["year"] == year].set_index("endpoint")
        maternal = sub.loc["outcome_maternal_morbidity_core", "positive_pct_known"]
        neonatal = sub.loc["outcome_severe_neonatal_no_nicu", "positive_pct_known"]
        report_lines.append(
            f"| {year} | {split_role(year)} | {maternal:.4f} | {neonatal:.4f} |"
        )

    report_lines.extend(
        [
            "",
            "## Output Tables",
            "",
            f"- `{manifest_path}`",
            f"- `{split_manifest_path}`",
            f"- `{prevalence_path}`",
            "",
            "## Interpretation",
            "",
            "- The planned temporal split is now explicit: 2016-2022 training, 2023 development/model selection, and 2024 final test.",
            "- The 2024 final test set must not be used for SSL hyperparameter selection or phenotype cluster-number selection.",
            "- The increasing maternal morbidity prevalence over calendar years should be described and handled through temporal validation rather than random splitting.",
        ]
    )
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    print(f"wrote {split_manifest_path}")
    print(f"wrote {prevalence_path}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
