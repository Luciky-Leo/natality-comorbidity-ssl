#!/usr/bin/env python
"""Export SSL embeddings for the linked birth/infant death cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from train_masked_tabular_ssl import (
    DEFAULT_OBJECT_DIR,
    Preprocessor,
    TabularMaskedTransformer,
    export_embeddings,
)
from export_masked_tabular_ssl_embeddings import load_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COHORT = PROJECT_ROOT / "data" / "processed" / "linked_infant_death_2023_cohort.parquet"
DEFAULT_CHECKPOINT = DEFAULT_OBJECT_DIR / "masked_tabular_ssl_encoder_full2016_2022_mask035_d48_l2_cuda.pt"
DEFAULT_PREPROCESSOR = DEFAULT_OBJECT_DIR / "masked_tabular_ssl_preprocessor_full2016_2022_mask035_d48_l2_cuda.json"
DEFAULT_OUTPUT = DEFAULT_OBJECT_DIR / "linked_infant_death_2023_ssl_embeddings_full2016_2022_mask035_d48_l2_cuda.parquet"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"

METADATA_COLS = [
    "source_year",
    "record_id",
    "linked_co_seqnum",
    "linked_co_yod",
    "outcome_infant_death",
    "outcome_neonatal_death_lt28d",
    "outcome_early_neonatal_death_lt7d",
    "outcome_postneonatal_death_28d_1y",
    "linked_age_at_death_days",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--preprocessor", type=Path, default=DEFAULT_PREPROCESSOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    prep = load_preprocessor(args.preprocessor)
    model_args = checkpoint["args"]
    model = TabularMaskedTransformer(
        cat_cardinalities=list(checkpoint["cat_cardinalities"]),
        n_numeric=len(prep.num_cols),
        d_model=int(model_args["d_model"]),
        n_heads=int(model_args["n_heads"]),
        n_layers=int(model_args["n_layers"]),
        dropout=float(model_args["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    columns = prep.cat_cols + prep.num_cols + METADATA_COLS
    print("load linked cohort", flush=True)
    frame = pd.read_parquet(args.cohort, columns=columns)
    print(f"rows={len(frame):,}, device={device}", flush=True)
    export_embeddings(
        model,
        frame,
        prep,
        args.output,
        args.batch_size,
        device,
        metadata_cols=METADATA_COLS,
    )
    metadata_path = args.tables / "linked_infant_death_2023_ssl_embedding_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "cohort": str(args.cohort),
                "checkpoint": str(args.checkpoint),
                "preprocessor": str(args.preprocessor),
                "output": str(args.output),
                "rows": int(len(frame)),
                "infant_deaths": int(frame["outcome_infant_death"].sum()),
                "device": str(device),
                "batch_size": int(args.batch_size),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
