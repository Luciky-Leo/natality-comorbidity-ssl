#!/usr/bin/env python
"""Export larger dev/test embeddings from an existing masked SSL checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from train_masked_tabular_ssl import (
    EXPORT_METADATA_COLS,
    DEFAULT_OBJECT_DIR,
    DEFAULT_SPLIT,
    Preprocessor,
    TabularMaskedTransformer,
    export_embeddings,
    input_columns_from_schema,
    load_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = DEFAULT_OBJECT_DIR / "masked_tabular_ssl_encoder.pt"
DEFAULT_PREPROCESSOR = DEFAULT_OBJECT_DIR / "masked_tabular_ssl_preprocessor.json"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--preprocessor", type=Path, default=DEFAULT_PREPROCESSOR)
    parser.add_argument("--objects", type=Path, default=DEFAULT_OBJECT_DIR)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--max-dev-rows", type=int, default=200_000)
    parser.add_argument("--max-test-rows", type=int, default=200_000)
    parser.add_argument("--max-train-per-year", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--output-tag", default="200k")
    parser.add_argument("--export-train", action="store_true")
    parser.add_argument("--export-dev", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-test", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_preprocessor(path: Path) -> Preprocessor:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Preprocessor(
        cat_cols=list(data["cat_cols"]),
        num_cols=list(data["num_cols"]),
        cat_maps={key: dict(value) for key, value in data["cat_maps"].items()},
        num_mean={key: float(value) for key, value in data["num_mean"].items()},
        num_std={key: float(value) for key, value in data["num_std"].items()},
    )


def main() -> None:
    args = parse_args()
    args.objects.mkdir(parents=True, exist_ok=True)
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

    manifest = pd.read_csv(args.split_manifest)
    input_cols = input_columns_from_schema(Path(manifest.iloc[0]["path"]))
    export_columns = input_cols + EXPORT_METADATA_COLS
    tag = f"_{args.output_tag}" if args.output_tag else ""

    train_path = None
    dev_path = None
    test_path = None
    train_rows = 0
    dev_rows = 0
    test_rows = 0

    if args.export_train:
        print("load train for embedding export")
        train = load_split(
            manifest,
            "train",
            export_columns,
            args.seed,
            max_per_year=args.max_train_per_year,
        )
        train_rows = int(len(train))
        train_path = args.objects / f"masked_tabular_ssl_train_embeddings{tag}.parquet"
        export_embeddings(
            model,
            train,
            prep,
            train_path,
            args.batch_size,
            device,
            metadata_cols=EXPORT_METADATA_COLS,
        )
        print(f"wrote {train_path}")

    if args.export_dev:
        print("load development for embedding export")
        dev = load_split(
            manifest,
            "development",
            export_columns,
            args.seed,
            max_rows=args.max_dev_rows,
        )
        dev_rows = int(len(dev))
        dev_path = args.objects / f"masked_tabular_ssl_dev_embeddings{tag}.parquet"
        export_embeddings(
            model,
            dev,
            prep,
            dev_path,
            args.batch_size,
            device,
            metadata_cols=EXPORT_METADATA_COLS,
        )
        print(f"wrote {dev_path}")

    if args.export_test:
        print("load test for embedding export")
        test = load_split(
            manifest,
            "test",
            export_columns,
            args.seed,
            max_rows=args.max_test_rows,
        )
        test_rows = int(len(test))
        test_path = args.objects / f"masked_tabular_ssl_test_embeddings{tag}.parquet"
        export_embeddings(
            model,
            test,
            prep,
            test_path,
            args.batch_size,
            device,
            metadata_cols=EXPORT_METADATA_COLS,
        )
        print(f"wrote {test_path}")

    metadata = {
        "checkpoint": str(args.checkpoint),
        "preprocessor": str(args.preprocessor),
        "train_embeddings": str(train_path) if train_path else None,
        "dev_embeddings": str(dev_path) if dev_path else None,
        "test_embeddings": str(test_path) if test_path else None,
        "train_rows": train_rows,
        "dev_rows": dev_rows,
        "test_rows": test_rows,
        "max_train_per_year": int(args.max_train_per_year),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "device": str(device),
    }
    metadata_path = args.tables / f"masked_tabular_ssl_embedding_export{tag}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
