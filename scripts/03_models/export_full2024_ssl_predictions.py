#!/usr/bin/env python
"""Export full-year 2024 SSL+phenotype predictions with streaming test batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEV_EMB = PROJECT_ROOT / "results" / "objects" / "masked_tabular_ssl_dev_embeddings_full2016_2022_mask035_d48_l2_cuda.parquet"
DEFAULT_TEST_EMB = PROJECT_ROOT / "results" / "objects" / "masked_tabular_ssl_test_embeddings_full2016_2022_mask035_d48_l2_cuda_full2024.parquet"
DEFAULT_DEV_ASSIGN = PROJECT_ROOT / "results" / "objects" / "ssl_phenotype_dev_assignments_full2016_2022_mask035_d48_l2_cuda_full2024.parquet"
DEFAULT_TEST_ASSIGN = PROJECT_ROOT / "results" / "objects" / "ssl_phenotype_test_assignments_full2016_2022_mask035_d48_l2_cuda_full2024.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "objects" / "ssl_plus_phenotype_predictions_full2016_2022_mask035_d48_l2_cuda_full2024.parquet"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"

ENDPOINTS = [
    "outcome_maternal_morbidity_core",
    "outcome_severe_neonatal_no_nicu",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-embeddings", type=Path, default=DEFAULT_DEV_EMB)
    parser.add_argument("--test-embeddings", type=Path, default=DEFAULT_TEST_EMB)
    parser.add_argument("--dev-assignments", type=Path, default=DEFAULT_DEV_ASSIGN)
    parser.add_argument("--test-assignments", type=Path, default=DEFAULT_TEST_ASSIGN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--output-tag", default="full2016_2022_mask035_d48_l2_cuda_full2024")
    return parser.parse_args()


def embedding_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if column.startswith("ssl_emb_")]


def phenotype_columns() -> list[str]:
    return ["phenotype_0", "phenotype_1", "phenotype_2"]


def make_design(frame: pd.DataFrame, emb_cols: list[str]) -> pd.DataFrame:
    phen = pd.get_dummies(frame["phenotype"].astype("category"), prefix="phenotype")
    phen = phen.reindex(columns=phenotype_columns(), fill_value=False)
    return pd.concat([frame[emb_cols].reset_index(drop=True), phen.astype("int8").reset_index(drop=True)], axis=1)


def clip_prob(prob: np.ndarray) -> np.ndarray:
    return np.clip(prob, 1e-6, 1 - 1e-6)


def fit_platt(y_cal: np.ndarray, p_cal: np.ndarray) -> LogisticRegression:
    logits = np.log(clip_prob(p_cal) / (1 - clip_prob(p_cal))).reshape(-1, 1)
    model = LogisticRegression(max_iter=1000)
    model.fit(logits, y_cal)
    return model


def apply_platt(model: LogisticRegression, prob: np.ndarray) -> np.ndarray:
    logits = np.log(clip_prob(prob) / (1 - clip_prob(prob))).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def fit_models(dev: pd.DataFrame, emb_cols: list[str], seed: int) -> dict[str, tuple[Pipeline, LogisticRegression]]:
    x_dev = make_design(dev, emb_cols)
    models: dict[str, tuple[Pipeline, LogisticRegression]] = {}
    for endpoint in ENDPOINTS:
        y = dev[endpoint].astype("int8").to_numpy()
        train_idx, cal_idx = train_test_split(np.arange(len(dev)), test_size=0.4, random_state=seed, stratify=y)
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logistic", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
            ]
        )
        model.fit(x_dev.iloc[train_idx], y[train_idx])
        p_cal = model.predict_proba(x_dev.iloc[cal_idx])[:, 1]
        platt = fit_platt(y[cal_idx], p_cal)
        models[endpoint] = (model, platt)
    return models


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    print("load development embeddings", flush=True)
    dev_emb = pd.read_parquet(args.dev_embeddings)
    emb_cols = embedding_columns(dev_emb.columns.tolist())
    dev_assign = pd.read_parquet(args.dev_assignments, columns=["record_id", "phenotype"])
    dev = dev_emb.merge(dev_assign, on="record_id", how="left")
    models = fit_models(dev, emb_cols, args.seed)
    del dev_emb, dev_assign, dev

    assignments = pd.read_parquet(args.test_assignments, columns=["record_id", "phenotype"])
    assignments["record_id"] = assignments["record_id"].astype("int64")
    assignment_index = assignments.set_index("record_id")

    columns = ["record_id", *ENDPOINTS, *emb_cols]
    parquet = pq.ParquetFile(args.test_embeddings)
    writer: pq.ParquetWriter | None = None
    rows = 0
    for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
        frame = batch.to_pandas()
        frame["record_id"] = frame["record_id"].astype("int64")
        phenotype = assignment_index.loc[frame["record_id"], "phenotype"].to_numpy()
        frame["phenotype"] = phenotype.astype("int16")
        x = make_design(frame, emb_cols)
        out = frame[["record_id", "phenotype", *ENDPOINTS]].copy()
        for endpoint, (model, platt) in models.items():
            raw_prob = model.predict_proba(x)[:, 1]
            out[f"pred_ssl_plus_phenotype_{endpoint}"] = apply_platt(platt, raw_prob).astype("float32")
        table = pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(args.output, table.schema, compression="zstd")
        else:
            table = table.cast(writer.schema)
        writer.write_table(table)
        rows += len(out)
        if rows % 1_000_000 < args.batch_size:
            print(f"wrote predictions for {rows:,} records", flush=True)
    if writer is not None:
        writer.close()
    metadata = {
        "dev_embeddings": str(args.dev_embeddings),
        "test_embeddings": str(args.test_embeddings),
        "dev_assignments": str(args.dev_assignments),
        "test_assignments": str(args.test_assignments),
        "output": str(args.output),
        "rows": int(rows),
        "embedding_columns": len(emb_cols),
        "model": "ssl_plus_phenotype_logistic_platt",
        "seed": int(args.seed),
    }
    metadata_path = args.tables / f"ssl_plus_phenotype_predictions_{args.output_tag}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
