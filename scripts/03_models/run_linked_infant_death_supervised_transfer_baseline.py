#!/usr/bin/env python
"""Compare fixed SSL phenotypes with a supervised score on linked infant death."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "processed" / "natality_2016_2024_split_manifest.csv"
DEFAULT_LINKED = PROJECT_ROOT / "data" / "processed" / "linked_infant_death_2023_cohort.parquet"
DEFAULT_TRAIN_EMB = PROJECT_ROOT / "results" / "objects" / "masked_tabular_ssl_train_embeddings_full2016_2022_mask035_d48_l2_cuda_train500k.parquet"
DEFAULT_DEV_EMB = PROJECT_ROOT / "results" / "objects" / "masked_tabular_ssl_dev_embeddings_full2016_2022_mask035_d48_l2_cuda_full2023dev.parquet"
DEFAULT_ASSIGNMENTS = PROJECT_ROOT / "results" / "objects" / "linked_infant_death_phenotype_assignments_full2016_2022_mask035_d48_l2_cuda_full2023dev.parquet"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_OBJECT_DIR = PROJECT_ROOT / "results" / "objects"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "47_linked_infant_death_supervised_transfer_baseline.md"

TRAINING_ENDPOINT = "outcome_severe_neonatal_no_nicu"
LINKED_OUTCOMES = [
    "outcome_infant_death",
    "outcome_neonatal_death_lt28d",
    "outcome_early_neonatal_death_lt7d",
    "outcome_postneonatal_death_28d_1y",
]

OUTCOME_LABELS = {
    "outcome_infant_death": "Infant death",
    "outcome_neonatal_death_lt28d": "Neonatal death <28d",
    "outcome_early_neonatal_death_lt7d": "Early neonatal death <7d",
    "outcome_postneonatal_death_28d_1y": "Postneonatal death 28d-1y",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--linked-cohort", type=Path, default=DEFAULT_LINKED)
    parser.add_argument("--train-embeddings", type=Path, default=DEFAULT_TRAIN_EMB)
    parser.add_argument("--dev-embeddings", type=Path, default=DEFAULT_DEV_EMB)
    parser.add_argument("--phenotype-assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--objects", type=Path, default=DEFAULT_OBJECT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-tag", default="full2016_2022_mask035_d48_l2_cuda_full2023dev")
    parser.add_argument("--lgbm-estimators", type=int, default=300)
    parser.add_argument("--predefined-high-risk-phenotype", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260525)
    return parser.parse_args()


def input_columns_from_schema(path: Path) -> list[str]:
    names = pq.ParquetFile(path).schema.names
    return [
        column
        for column in names
        if column.startswith("input_") or column.startswith("missing_input_")
    ]


def load_train_exact(manifest: pd.DataFrame, train_embeddings: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frames = []
    order = train_embeddings[["source_year", "record_id"]].copy()
    order["source_year"] = order["source_year"].astype("int32")
    order["record_id"] = order["record_id"].astype("int64")
    for year, group in order.groupby("source_year", sort=True):
        path = Path(manifest.loc[manifest["year"].eq(int(year)), "path"].iloc[0])
        wanted_ids = set(group["record_id"].tolist())
        frame = pq.read_table(path, columns=["record_id"] + columns).to_pandas()
        frame["record_id"] = frame["record_id"].astype("int64")
        frame = frame[frame["record_id"].isin(wanted_ids)].copy()
        frame.insert(0, "source_year", int(year))
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    merged = order.merge(raw, on=["source_year", "record_id"], how="left")
    if merged[columns].isna().all(axis=1).any():
        missing = int(merged[columns].isna().all(axis=1).sum())
        raise RuntimeError(f"missing train records after merge: {missing}")
    return merged


def load_development(manifest: pd.DataFrame, dev_embeddings: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    order = dev_embeddings[["record_id"]].copy()
    order["record_id"] = order["record_id"].astype("int64")
    path = Path(manifest.loc[manifest["split"].eq("development"), "path"].iloc[0])
    frame = pq.read_table(path, columns=["record_id"] + columns).to_pandas()
    frame["record_id"] = frame["record_id"].astype("int64")
    merged = order.merge(frame, on="record_id", how="left")
    if merged[columns].isna().all(axis=1).any():
        missing = int(merged[columns].isna().all(axis=1).sum())
        raise RuntimeError(f"missing development records after merge: {missing}")
    return merged


def prepare_lgbm_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_bool_dtype(out[column]):
            out[column] = out[column].astype("int8")
        elif pd.api.types.is_object_dtype(out[column]) or isinstance(out[column].dtype, pd.StringDtype):
            out[column] = out[column].astype("category")
    return out


def make_lgbm(seed: int, n_estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=1000,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )


def clip_prob(prob: np.ndarray) -> np.ndarray:
    return np.clip(prob, 1e-6, 1 - 1e-6)


def fit_platt(y_dev: np.ndarray, p_dev: np.ndarray) -> LogisticRegression:
    logits = np.log(clip_prob(p_dev) / (1 - clip_prob(p_dev))).reshape(-1, 1)
    model = LogisticRegression(max_iter=1000)
    model.fit(logits, y_dev)
    return model


def apply_platt(model: LogisticRegression, prob: np.ndarray) -> np.ndarray:
    logits = np.log(clip_prob(prob) / (1 - clip_prob(prob))).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def selected_rows(frame: pd.DataFrame, selected: pd.Series, rule: str) -> list[dict[str, object]]:
    rows = []
    n_total = len(frame)
    n_selected = int(selected.sum())
    selected_fraction = n_selected / n_total if n_total else np.nan
    for outcome in LINKED_OUTCOMES:
        y = frame[outcome].astype("int8")
        events_total = int(y.sum())
        events_selected = int(y[selected].sum())
        prevalence = events_total / n_total if n_total else np.nan
        event_rate = events_selected / n_selected if n_selected else np.nan
        rows.append(
            {
                "rule": rule,
                "outcome": outcome,
                "outcome_label": OUTCOME_LABELS[outcome],
                "n_total": n_total,
                "n_selected": n_selected,
                "selected_fraction": selected_fraction,
                "events_total": events_total,
                "events_selected": events_selected,
                "prevalence": prevalence,
                "event_rate_selected": event_rate,
                "enrichment_over_prevalence": event_rate / prevalence if prevalence else np.nan,
                "event_capture_pct": 100 * events_selected / events_total if events_total else np.nan,
            }
        )
    return rows


def score_threshold_selection(prob: np.ndarray, fraction: float) -> np.ndarray:
    k = max(1, int(round(len(prob) * fraction)))
    order = np.argsort(-prob)
    selected = np.zeros(len(prob), dtype=bool)
    selected[order[:k]] = True
    return selected


def main() -> None:
    args = parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    args.objects.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_tag}" if args.output_tag else ""

    manifest = pd.read_csv(args.split_manifest)
    input_cols = input_columns_from_schema(Path(manifest.iloc[0]["path"]))
    raw_cols = input_cols + [TRAINING_ENDPOINT]

    print("load training/development identifiers", flush=True)
    train_emb = pd.read_parquet(args.train_embeddings, columns=["source_year", "record_id"])
    dev_emb = pd.read_parquet(args.dev_embeddings, columns=["record_id"])

    print("load exact training records", flush=True)
    train = load_train_exact(manifest, train_emb, raw_cols)
    print("load development records", flush=True)
    dev = load_development(manifest, dev_emb, raw_cols)

    print("fit all-input LightGBM severe-neonatal model", flush=True)
    model = make_lgbm(args.seed, args.lgbm_estimators)
    model.fit(
        prepare_lgbm_frame(train[input_cols]),
        train[TRAINING_ENDPOINT].astype("int8").to_numpy(),
        categorical_feature="auto",
    )
    print("fit Platt recalibration on 2023 development records", flush=True)
    p_dev_raw = model.predict_proba(prepare_lgbm_frame(dev[input_cols]))[:, 1]
    platt = fit_platt(dev[TRAINING_ENDPOINT].astype("int8").to_numpy(), p_dev_raw)

    print("score linked infant-death cohort", flush=True)
    linked_cols = ["source_year", "record_id", *input_cols, *LINKED_OUTCOMES]
    linked = pd.read_parquet(args.linked_cohort, columns=linked_cols)
    assignments = pd.read_parquet(args.phenotype_assignments, columns=["record_id", "phenotype"])
    linked = linked.merge(assignments, on="record_id", how="inner")
    p_linked_raw = model.predict_proba(prepare_lgbm_frame(linked[input_cols]))[:, 1]
    linked["pred_all_input_lgbm_severe_neonatal_platt"] = apply_platt(platt, p_linked_raw)

    phenotype_selected = linked["phenotype"].eq(args.predefined_high_risk_phenotype)
    phenotype_fraction = float(phenotype_selected.mean())
    rows = []
    rows.extend(selected_rows(linked, phenotype_selected, f"phenotype_{args.predefined_high_risk_phenotype}"))
    for fraction in [0.005, 0.01, 0.02, phenotype_fraction, 0.05, 0.10]:
        selected = pd.Series(
            score_threshold_selection(linked["pred_all_input_lgbm_severe_neonatal_platt"].to_numpy(), fraction),
            index=linked.index,
        )
        rows.extend(selected_rows(linked, selected, f"all_input_lgbm_top_{100*fraction:.3f}pct"))

    comparison = pd.DataFrame(rows)
    scores_path = args.objects / f"linked_infant_death_supervised_transfer_scores{suffix}.parquet"
    comparison_path = args.tables / f"linked_infant_death_supervised_transfer_baseline{suffix}.csv"
    metadata_path = args.tables / f"linked_infant_death_supervised_transfer_baseline_metadata{suffix}.json"
    linked[
        [
            "source_year",
            "record_id",
            "phenotype",
            *LINKED_OUTCOMES,
            "pred_all_input_lgbm_severe_neonatal_platt",
        ]
    ].to_parquet(scores_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "training_endpoint": TRAINING_ENDPOINT,
                "linked_outcomes": LINKED_OUTCOMES,
                "train_embeddings": str(args.train_embeddings),
                "development_embeddings": str(args.dev_embeddings),
                "linked_cohort": str(args.linked_cohort),
                "phenotype_assignments": str(args.phenotype_assignments),
                "input_columns": len(input_cols),
                "train_rows": int(len(train)),
                "development_rows": int(len(dev)),
                "linked_rows": int(len(linked)),
                "phenotype_high_risk": int(args.predefined_high_risk_phenotype),
                "phenotype_high_risk_fraction": phenotype_fraction,
                "lgbm_estimators": int(args.lgbm_estimators),
                "seed": int(args.seed),
                "interpretation": "The supervised score was trained for severe neonatal outcome, recalibrated in 2023, and transferred without infant-death labels to linked infant-death outcomes.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    infant = comparison[comparison["outcome"].eq("outcome_infant_death")].copy()
    lines = [
        "# Linked Infant Death Supervised Transfer Baseline",
        "",
        "All-input LightGBM was trained for the registry-defined severe neonatal endpoint using the same 2016-2022 sampled training records as the main incremental analysis, Platt recalibrated in 2023, and transferred to the linked birth/infant death cohort without using infant-death labels.",
        "",
        f"- training rows: {len(train):,}",
        f"- development rows: {len(dev):,}",
        f"- linked rows: {len(linked):,}",
        f"- fixed high-risk phenotype fraction: {phenotype_fraction:.4%}",
        "",
        "## Infant death enrichment",
        "",
        "| Rule | Selected % | Selected n | Infant death rate % | Enrichment | Death capture % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in infant.to_dict("records"):
        lines.append(
            "| {rule} | {pct:.3f} | {n:,} | {rate:.3f} | {enrich:.2f} | {capture:.2f} |".format(
                rule=row["rule"],
                pct=100 * float(row["selected_fraction"]),
                n=int(row["n_selected"]),
                rate=100 * float(row["event_rate_selected"]),
                enrich=float(row["enrichment_over_prevalence"]),
                capture=float(row["event_capture_pct"]),
            )
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{comparison_path}`",
            f"- `{metadata_path}`",
            f"- `{scores_path}`",
        ]
    )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {comparison_path}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
