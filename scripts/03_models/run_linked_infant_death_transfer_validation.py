#!/usr/bin/env python
"""Validate fixed SSL phenotypes against linked infant death outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMB = PROJECT_ROOT / "results" / "objects" / "linked_infant_death_2023_ssl_embeddings_full2016_2022_mask035_d48_l2_cuda.parquet"
DEFAULT_MODEL = PROJECT_ROOT / "results" / "objects" / "ssl_phenotype_model_full2016_2022_mask035_d48_l2_cuda.npz"
DEFAULT_OBJECT_DIR = PROJECT_ROOT / "results" / "objects"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "32_linked_infant_death_transfer_validation_report.md"

OUTCOMES = [
    "outcome_infant_death",
    "outcome_neonatal_death_lt28d",
    "outcome_early_neonatal_death_lt7d",
    "outcome_postneonatal_death_28d_1y",
]

OUTCOME_LABEL = {
    "outcome_infant_death": "Infant death",
    "outcome_neonatal_death_lt28d": "Neonatal death <28d",
    "outcome_early_neonatal_death_lt7d": "Early neonatal death <7d",
    "outcome_postneonatal_death_28d_1y": "Postneonatal death 28d-1y",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMB)
    parser.add_argument("--phenotype-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--objects", type=Path, default=DEFAULT_OBJECT_DIR)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-tag", default="full2016_2022_mask035_d48_l2_cuda")
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--predefined-high-risk-phenotype", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260525)
    return parser.parse_args()


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("ssl_emb_")]


def assign_phenotypes(x_raw: np.ndarray, model_npz: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    scaler_mean = model_npz["scaler_mean"]
    scaler_scale = model_npz["scaler_scale"]
    pca_components = model_npz["pca_components"]
    pca_mean = model_npz["pca_mean"]
    centroids = model_npz["centroids"]
    x_scaled = (x_raw - scaler_mean) / scaler_scale
    x_pca = (x_scaled - pca_mean) @ pca_components.T
    distances = ((x_pca[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(distances, axis=1)
    return labels.astype("int16"), x_pca


def phenotype_rate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phenotype, group in frame.groupby("phenotype", sort=True):
        row = {"phenotype": int(phenotype), "n": int(len(group)), "proportion": float(len(group) / len(frame))}
        for outcome in OUTCOMES:
            row[f"{outcome}_events"] = int(group[outcome].sum())
            row[f"{outcome}_rate"] = float(group[outcome].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_rate_ci(frame: pd.DataFrame, seed: int, iterations: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for phenotype, group in frame.groupby("phenotype", sort=True):
        for outcome in OUTCOMES:
            y = group[outcome].astype("int8").to_numpy()
            boot = []
            for _ in range(iterations):
                idx = rng.integers(0, len(y), size=len(y))
                boot.append(float(np.mean(y[idx])))
            low, high = np.percentile(np.asarray(boot), [2.5, 97.5])
            rows.append(
                {
                    "phenotype": int(phenotype),
                    "outcome": outcome,
                    "outcome_label": OUTCOME_LABEL[outcome],
                    "n": int(len(y)),
                    "events": int(np.sum(y)),
                    "event_rate": float(np.mean(y)),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "n_bootstrap_requested": int(iterations),
                }
            )
    return pd.DataFrame(rows)


def predefined_high_risk_rows(frame: pd.DataFrame, high_risk: int) -> pd.DataFrame:
    rows = []
    selected = frame["phenotype"].eq(high_risk)
    for outcome in OUTCOMES:
        y = frame[outcome].astype("int8")
        prevalence = float(y.mean())
        event_rate = float(y[selected].mean())
        rows.append(
            {
                "rule": f"phenotype_{high_risk}",
                "outcome": outcome,
                "outcome_label": OUTCOME_LABEL[outcome],
                "n_total": int(len(frame)),
                "n_selected": int(selected.sum()),
                "selected_fraction": float(selected.mean()),
                "events_total": int(y.sum()),
                "events_selected": int(y[selected].sum()),
                "prevalence": prevalence,
                "event_rate_selected": event_rate,
                "enrichment_over_prevalence": event_rate / prevalence if prevalence else np.nan,
                "event_capture_pct": 100 * int(y[selected].sum()) / int(y.sum()) if int(y.sum()) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.objects.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)
    if args.output_tag and args.report == DEFAULT_REPORT:
        args.report = args.report.with_name(f"{args.report.stem}_{args.output_tag}{args.report.suffix}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_tag}" if args.output_tag else ""

    frame = pd.read_parquet(args.embeddings)
    emb_cols = embedding_columns(frame)
    model_npz = np.load(args.phenotype_model)
    labels, x_pca = assign_phenotypes(frame[emb_cols].to_numpy(dtype=np.float32), model_npz)
    out = frame[[
        "source_year",
        "record_id",
        "linked_co_seqnum",
        "linked_co_yod",
        *OUTCOMES,
        "linked_age_at_death_days",
    ]].copy()
    out["phenotype"] = labels
    out["pc1"] = x_pca[:, 0]
    out["pc2"] = x_pca[:, 1] if x_pca.shape[1] > 1 else 0.0

    assignment_path = args.objects / f"linked_infant_death_phenotype_assignments{suffix}.parquet"
    rates_path = args.tables / f"linked_infant_death_phenotype_rates{suffix}.csv"
    ci_path = args.tables / f"linked_infant_death_phenotype_rate_bootstrap_ci{suffix}.csv"
    highrisk_path = args.tables / f"linked_infant_death_predefined_highrisk_enrichment{suffix}.csv"
    metadata_path = args.tables / f"linked_infant_death_transfer_metadata{suffix}.json"

    rates = phenotype_rate_rows(out)
    ci = bootstrap_rate_ci(out, args.seed + 17, args.bootstrap_iterations)
    highrisk = predefined_high_risk_rows(out, args.predefined_high_risk_phenotype)

    out.to_parquet(assignment_path, index=False)
    rates.to_csv(rates_path, index=False)
    ci.to_csv(ci_path, index=False)
    highrisk.to_csv(highrisk_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "embeddings": str(args.embeddings),
                "phenotype_model": str(args.phenotype_model),
                "rows": int(len(out)),
                "embedding_columns": len(emb_cols),
                "predefined_high_risk_phenotype": int(args.predefined_high_risk_phenotype),
                "bootstrap_iterations": int(args.bootstrap_iterations),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    infant_ci = ci[ci["outcome"].eq("outcome_infant_death")].sort_values("phenotype")
    lines = [
        "# Linked Infant Death Transfer Validation Report",
        "",
        "Fixed SSL phenotype centroids from the Natality development analysis were applied to the 2023 linked birth/infant death cohort. Infant death labels were not used to train the SSL encoder or the phenotype centroids.",
        "",
        f"- linked cohort rows: {len(out):,}",
        f"- predefined high-risk phenotype: {args.predefined_high_risk_phenotype}",
        "",
        "## Infant Death Rate by Fixed Phenotype",
        "",
        "| Phenotype | n | Cohort % | Infant deaths | Rate % | 95% CI % |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    rates_indexed = rates.set_index("phenotype")
    for row in infant_ci.to_dict("records"):
        phenotype = int(row["phenotype"])
        lines.append(
            "| {phenotype} | {n:,} | {prop:.2f} | {events:,} | {rate:.3f} | {low:.3f}-{high:.3f} |".format(
                phenotype=phenotype,
                n=int(row["n"]),
                prop=100 * float(rates_indexed.loc[phenotype, "proportion"]),
                events=int(row["events"]),
                rate=100 * float(row["event_rate"]),
                low=100 * float(row["ci_low"]),
                high=100 * float(row["ci_high"]),
            )
        )

    lines.extend(
        [
            "",
            "## Predefined High-Risk Phenotype Enrichment",
            "",
            "| Outcome | Selected fraction | Event rate selected % | Baseline % | Enrichment | Event capture % |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in highrisk.to_dict("records"):
        lines.append(
            "| {outcome_label} | {selected_fraction:.2%} | {sel:.3f} | {base:.3f} | {enrich:.2f} | {capture:.2f} |".format(
                outcome_label=row["outcome_label"],
                selected_fraction=float(row["selected_fraction"]),
                sel=100 * float(row["event_rate_selected"]),
                base=100 * float(row["prevalence"]),
                enrich=float(row["enrichment_over_prevalence"]),
                capture=float(row["event_capture_pct"]),
            )
        )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{assignment_path}`",
            f"- `{rates_path}`",
            f"- `{ci_path}`",
            f"- `{highrisk_path}`",
            f"- `{metadata_path}`",
        ]
    )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
