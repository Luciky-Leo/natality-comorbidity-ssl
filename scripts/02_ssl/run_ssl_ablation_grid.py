#!/usr/bin/env python
"""Run a focused SSL hyperparameter ablation grid.

The grid deliberately writes tagged outputs so the manuscript baseline files are
not overwritten.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "02_ssl" / "train_masked_tabular_ssl.py"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "29_ssl_ablation_grid_report.md"


CONFIGS = [
    {"tag": "abl_mask010_d48_l2", "mask_rate": 0.10, "d_model": 48, "n_layers": 2},
    {"tag": "abl_mask020_d48_l2", "mask_rate": 0.20, "d_model": 48, "n_layers": 2},
    {"tag": "abl_mask035_d48_l2", "mask_rate": 0.35, "d_model": 48, "n_layers": 2},
    {"tag": "abl_mask020_d32_l2", "mask_rate": 0.20, "d_model": 32, "n_layers": 2},
    {"tag": "abl_mask020_d48_l1", "mask_rate": 0.20, "d_model": 48, "n_layers": 1},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-train-per-year", type=int, default=50_000)
    parser.add_argument("--max-dev-rows", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def history_path(tables: Path, tag: str) -> Path:
    return tables / f"masked_tabular_ssl_history_{tag}.csv"


def run_config(args: argparse.Namespace, config: dict[str, object]) -> None:
    tag = str(config["tag"])
    if args.skip_existing and history_path(args.tables, tag).exists():
        print(f"skip existing {tag}")
        return
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--max-train-per-year",
        str(args.max_train_per_year),
        "--max-dev-rows",
        str(args.max_dev_rows),
        "--max-test-rows",
        "1",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--mask-rate",
        str(config["mask_rate"]),
        "--d-model",
        str(config["d_model"]),
        "--n-layers",
        str(config["n_layers"]),
        "--seed",
        str(args.seed),
        "--output-tag",
        tag,
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def summarize(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for config in CONFIGS:
        tag = str(config["tag"])
        path = history_path(args.tables, tag)
        if not path.exists():
            rows.append({**config, "status": "missing"})
            continue
        history = pd.read_csv(path)
        final = history.iloc[-1].to_dict()
        rows.append(
            {
                **config,
                "status": "complete",
                "epochs": int(history["epoch"].max()),
                "train_rows": args.max_train_per_year * 7,
                "dev_rows": args.max_dev_rows,
                "final_train_loss": float(final["train_loss"]),
                "final_dev_loss": float(final["dev_loss"]),
                "final_dev_cat_loss": float(final["dev_cat_loss"]),
                "final_dev_num_loss": float(final["dev_num_loss"]),
                "history_file": str(path),
            }
        )
    out = pd.DataFrame(rows)
    out_path = args.tables / "ssl_ablation_grid.csv"
    out.to_csv(out_path, index=False)
    return out


def write_report(args: argparse.Namespace, summary: pd.DataFrame) -> None:
    complete = summary[summary["status"] == "complete"].copy()
    lines = [
        "# SSL Ablation Grid Report",
        "",
        "This focused reconstruction ablation varies masking rate, embedding width, and encoder depth while preserving the 2016-2022 pretrain / 2023 development boundary.",
        "",
        f"- train sample per configuration: {args.max_train_per_year:,} records/year x 7 years",
        f"- development sample per configuration: {args.max_dev_rows:,}",
        f"- epochs per configuration: {args.epochs}",
        "",
        "## Final Development Reconstruction Loss",
        "",
        "| Tag | Mask | d_model | Layers | Dev loss | Dev cat | Dev num |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in complete.sort_values("final_dev_loss").to_dict("records"):
        lines.append(
            "| {tag} | {mask_rate:.2f} | {d_model} | {n_layers} | {final_dev_loss:.4f} | {final_dev_cat_loss:.4f} | {final_dev_num_loss:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a reconstruction-level ablation. The selected SSL configuration should still be propagated through embedding export, phenotype clustering, calibration, and final 2024 risk enrichment before manuscript claims are upgraded.",
            "",
            "## Output",
            "",
            f"- `{args.tables / 'ssl_ablation_grid.csv'}`",
        ]
    )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for config in CONFIGS:
        run_config(args, config)
    summary = summarize(args)
    write_report(args, summary)
    print(f"wrote {args.tables / 'ssl_ablation_grid.csv'}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
