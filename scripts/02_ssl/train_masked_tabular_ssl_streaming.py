#!/usr/bin/env python
"""Stream masked-tabular SSL training over multi-year Natality Parquet files.

This is the high-scale entry point for analyses that should not load all
2016-2022 records into pandas at once. It fits preprocessing statistics by
streaming training row groups, then trains the same tabular transformer in
row-group chunks.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn

from train_masked_tabular_ssl import (
    DEFAULT_OBJECT_DIR,
    DEFAULT_REPORT,
    DEFAULT_SPLIT,
    DEFAULT_TABLE_DIR,
    EXPORT_METADATA_COLS,
    Preprocessor,
    TabularMaskedTransformer,
    export_embeddings,
    input_columns_from_schema,
    load_split,
    make_masks,
    masked_loss,
    tagged_path,
    transform_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--objects", type=Path, default=DEFAULT_OBJECT_DIR)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--mask-rate", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-categories", type=int, default=128)
    parser.add_argument("--max-dev-rows", type=int, default=200_000)
    parser.add_argument("--max-test-rows", type=int, default=200_000)
    parser.add_argument(
        "--max-row-groups-per-year",
        type=int,
        default=None,
        help="Debug/smoke option. Omit for every row group in every training year.",
    )
    parser.add_argument(
        "--max-rows-per-row-group",
        type=int,
        default=None,
        help="Debug/smoke option. Omit to use every row in each selected row group.",
    )
    parser.add_argument("--output-tag", default="fullstream")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--export-embeddings", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def column_groups(path: Path, input_cols: list[str]) -> tuple[list[str], list[str]]:
    schema = pq.ParquetFile(path).schema_arrow
    cat_cols: list[str] = []
    num_cols: list[str] = []
    for column in input_cols:
        field = schema.field(column)
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            cat_cols.append(column)
        else:
            num_cols.append(column)
    return cat_cols, num_cols


def iter_train_chunks(
    manifest: pd.DataFrame,
    input_cols: list[str],
    seed: int,
    max_row_groups_per_year: int | None,
    max_rows_per_row_group: int | None,
    shuffle_row_groups: bool,
):
    rng = np.random.default_rng(seed)
    work_items: list[tuple[int, Path, int]] = []
    for row in manifest[manifest["split"] == "train"].to_dict("records"):
        path = Path(row["path"])
        pf = pq.ParquetFile(path)
        row_groups = list(range(pf.num_row_groups))
        if max_row_groups_per_year is not None:
            row_groups = row_groups[: max(0, max_row_groups_per_year)]
        for row_group in row_groups:
            work_items.append((int(row["year"]), path, row_group))
    if shuffle_row_groups:
        rng.shuffle(work_items)

    for year, path, row_group in work_items:
        table = pq.ParquetFile(path).read_row_group(row_group, columns=input_cols)
        frame = table.to_pandas()
        if max_rows_per_row_group is not None and len(frame) > max_rows_per_row_group:
            frame = frame.sample(
                n=max_rows_per_row_group,
                random_state=seed + year * 1000 + row_group,
            )
        yield year, row_group, frame.reset_index(drop=True)


def fit_preprocessor_streaming(
    manifest: pd.DataFrame,
    input_cols: list[str],
    seed: int,
    max_categories: int,
    max_row_groups_per_year: int | None,
    max_rows_per_row_group: int | None,
) -> tuple[Preprocessor, int]:
    first_path = Path(manifest.loc[manifest["split"] == "train", "path"].iloc[0])
    cat_cols, num_cols = column_groups(first_path, input_cols)
    counters = {column: Counter() for column in cat_cols}
    sums = {column: 0.0 for column in num_cols}
    sumsq = {column: 0.0 for column in num_cols}
    counts = {column: 0 for column in num_cols}
    total_rows = 0

    for _, _, frame in iter_train_chunks(
        manifest,
        input_cols,
        seed,
        max_row_groups_per_year,
        max_rows_per_row_group,
        shuffle_row_groups=False,
    ):
        total_rows += len(frame)
        for column in cat_cols:
            values = frame[column].astype("string").fillna("__MISSING__")
            counters[column].update(values.tolist())
        for column in num_cols:
            values = pd.to_numeric(frame[column], errors="coerce").astype("float64")
            valid = values.dropna()
            if len(valid):
                sums[column] += float(valid.sum())
                sumsq[column] += float(np.square(valid.to_numpy()).sum())
                counts[column] += int(len(valid))

    cat_maps: dict[str, dict[str, int]] = {}
    for column in cat_cols:
        values = ["__MISSING__"]
        values.extend(
            value
            for value, _ in counters[column].most_common(max_categories)
            if value != "__MISSING__"
        )
        cat_maps[column] = {
            value: idx for idx, value in enumerate(dict.fromkeys(values))
        }

    num_mean: dict[str, float] = {}
    num_std: dict[str, float] = {}
    for column in num_cols:
        if counts[column] == 0:
            mean = 0.0
            std = 1.0
        else:
            mean = sums[column] / counts[column]
            variance = max(sumsq[column] / counts[column] - mean * mean, 0.0)
            std = float(np.sqrt(variance))
            if not np.isfinite(std) or std < 1e-6:
                std = 1.0
        num_mean[column] = float(mean)
        num_std[column] = float(std)

    return Preprocessor(cat_cols, num_cols, cat_maps, num_mean, num_std), total_rows


def train_streaming_epoch(
    model: TabularMaskedTransformer,
    optimizer: torch.optim.Optimizer,
    manifest: pd.DataFrame,
    input_cols: list[str],
    prep: Preprocessor,
    cat_mask_ids: list[int],
    args: argparse.Namespace,
    epoch: int,
    device: torch.device,
) -> dict[str, float]:
    model.train(True)
    totals = {"loss": 0.0, "cat_loss": 0.0, "num_loss": 0.0, "n": 0}
    chunks_seen = 0
    rows_seen = 0
    for chunks_seen, (year, row_group, frame) in enumerate(iter_train_chunks(
        manifest,
        input_cols,
        args.seed + epoch,
        args.max_row_groups_per_year,
        args.max_rows_per_row_group,
        shuffle_row_groups=True,
    ), start=1):
        rows_seen += len(frame)
        cat_x, num_x = transform_frame(frame[prep.cat_cols + prep.num_cols], prep)
        order = np.random.default_rng(args.seed + epoch + len(frame)).permutation(len(frame))
        for start in range(0, len(frame), args.batch_size):
            idx = order[start : start + args.batch_size]
            cat_batch = torch.from_numpy(cat_x[idx].astype(np.int64)).to(device)
            num_batch = torch.from_numpy(num_x[idx].astype(np.float32)).to(device)
            cat_masked, num_masked, cat_mask, num_mask = make_masks(
                cat_batch, num_batch, cat_mask_ids, args.mask_rate, device
            )
            cat_logits, numeric_preds, _ = model(cat_masked, num_masked, cat_mask, num_mask)
            loss, cat_loss, num_loss = masked_loss(
                cat_logits, numeric_preds, cat_batch, num_batch, cat_mask, num_mask
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_n = len(idx)
            totals["loss"] += float(loss.detach().cpu()) * batch_n
            totals["cat_loss"] += cat_loss * batch_n
            totals["num_loss"] += num_loss * batch_n
            totals["n"] += batch_n
        if chunks_seen == 1 or chunks_seen % 10 == 0:
            running_loss = totals["loss"] / max(totals["n"], 1)
            print(
                f"epoch {epoch} progress: chunks={chunks_seen}, rows={rows_seen:,}, last_year={year}, row_group={row_group}, train_loss={running_loss:.4f}",
                flush=True,
            )
    out = {key: value / totals["n"] for key, value in totals.items() if key != "n"}
    out["rows"] = int(totals["n"])
    out["chunks"] = int(chunks_seen)
    return out


def evaluate_frame(
    model: TabularMaskedTransformer,
    frame: pd.DataFrame,
    prep: Preprocessor,
    cat_mask_ids: list[int],
    mask_rate: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.train(False)
    cat_x, num_x = transform_frame(frame[prep.cat_cols + prep.num_cols], prep)
    totals = {"loss": 0.0, "cat_loss": 0.0, "num_loss": 0.0, "n": 0}
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            cat_batch = torch.from_numpy(cat_x[start:stop].astype(np.int64)).to(device)
            num_batch = torch.from_numpy(num_x[start:stop].astype(np.float32)).to(device)
            cat_masked, num_masked, cat_mask, num_mask = make_masks(
                cat_batch, num_batch, cat_mask_ids, mask_rate, device
            )
            cat_logits, numeric_preds, _ = model(cat_masked, num_masked, cat_mask, num_mask)
            loss, cat_loss, num_loss = masked_loss(
                cat_logits, numeric_preds, cat_batch, num_batch, cat_mask, num_mask
            )
            batch_n = stop - start
            totals["loss"] += float(loss.detach().cpu()) * batch_n
            totals["cat_loss"] += cat_loss * batch_n
            totals["num_loss"] += num_loss * batch_n
            totals["n"] += batch_n
    return {key: value / totals["n"] for key, value in totals.items() if key != "n"}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.objects.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)
    report_path = tagged_path(args.report, args.output_tag)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = pd.read_csv(args.split_manifest)
    input_cols = input_columns_from_schema(Path(manifest.iloc[0]["path"]))
    print(f"input columns: {len(input_cols)}")
    print("fit streaming preprocessor", flush=True)
    prep, train_rows_seen = fit_preprocessor_streaming(
        manifest,
        input_cols,
        args.seed,
        args.max_categories,
        args.max_row_groups_per_year,
        args.max_rows_per_row_group,
    )
    print(
        f"streaming rows for preprocessor: {train_rows_seen:,}; categorical={len(prep.cat_cols)}, numeric={len(prep.num_cols)}",
        flush=True,
    )

    dev = load_split(
        manifest,
        "development",
        input_cols,
        args.seed,
        max_rows=args.max_dev_rows,
    )
    cat_cardinalities = [len(prep.cat_maps[column]) + 1 for column in prep.cat_cols]
    cat_mask_ids = list(cat_cardinalities)
    model = TabularMaskedTransformer(
        cat_cardinalities=cat_cardinalities,
        n_numeric=len(prep.num_cols),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_rows = []
    train_rows_per_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_streaming_epoch(
            model,
            optimizer,
            manifest,
            input_cols,
            prep,
            cat_mask_ids,
            args,
            epoch,
            device,
        )
        train_rows_per_epoch = int(round(train_metrics.get("rows", 0)))
        dev_metrics = evaluate_frame(
            model,
            dev,
            prep,
            cat_mask_ids,
            args.mask_rate,
            args.batch_size,
            device,
        )
        row = {
            "epoch": epoch,
            "train_rows_seen_preprocessor": int(train_rows_seen),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"dev_{key}": value for key, value in dev_metrics.items()},
        }
        history_rows.append(row)
        print(
            f"epoch {epoch}: train_rows={train_rows_per_epoch:,}, train_loss={row['train_loss']:.4f}, dev_loss={row['dev_loss']:.4f}",
            flush=True,
        )

    history_path = tagged_path(args.tables / "masked_tabular_ssl_history.csv", args.output_tag)
    prep_path = tagged_path(args.objects / "masked_tabular_ssl_preprocessor.json", args.output_tag)
    checkpoint_path = tagged_path(args.objects / "masked_tabular_ssl_encoder.pt", args.output_tag)
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    prep_path.write_text(json.dumps(prep.to_jsonable(), indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cat_cardinalities": cat_cardinalities,
            "cat_cols": prep.cat_cols,
            "num_cols": prep.num_cols,
            "args": vars(args),
        },
        checkpoint_path,
    )

    exported = []
    if args.export_embeddings:
        export_columns = input_cols + EXPORT_METADATA_COLS
        dev_export = load_split(
            manifest,
            "development",
            export_columns,
            args.seed,
            max_rows=args.max_dev_rows,
        )
        dev_embedding_path = tagged_path(
            args.objects / "masked_tabular_ssl_dev_embeddings.parquet",
            args.output_tag,
        )
        export_embeddings(
            model,
            dev_export,
            prep,
            dev_embedding_path,
            args.batch_size,
            device,
            metadata_cols=EXPORT_METADATA_COLS,
        )
        exported.append(dev_embedding_path)
        test = load_split(
            manifest,
            "test",
            export_columns,
            args.seed,
            max_rows=args.max_test_rows,
        )
        test_embedding_path = tagged_path(
            args.objects / "masked_tabular_ssl_test_embeddings.parquet",
            args.output_tag,
        )
        export_embeddings(
            model,
            test,
            prep,
            test_embedding_path,
            args.batch_size,
            device,
            metadata_cols=EXPORT_METADATA_COLS,
        )
        exported.append(test_embedding_path)

    final = history_rows[-1]
    lines = [
        "# Streaming Masked Tabular SSL Report",
        "",
        "Preprocessing and training stream through 2016-2022 row groups without loading all rows into memory.",
        "",
        "## Configuration",
        "",
        f"- device: {device}",
        f"- train rows seen by preprocessor: {train_rows_seen:,}",
        f"- row groups per year: {args.max_row_groups_per_year if args.max_row_groups_per_year is not None else 'all'}",
        f"- rows per row group: {args.max_rows_per_row_group if args.max_rows_per_row_group is not None else 'all'}",
        f"- development rows: {len(dev):,}",
        f"- categorical features: {len(prep.cat_cols)}",
        f"- numeric features: {len(prep.num_cols)}",
        f"- d_model: {args.d_model}",
        f"- layers: {args.n_layers}",
        f"- heads: {args.n_heads}",
        f"- mask rate: {args.mask_rate}",
        f"- epochs: {args.epochs}",
        "",
        "## Final Development Loss",
        "",
        f"- dev total loss: {final['dev_loss']:.6f}",
        f"- dev categorical reconstruction loss: {final['dev_cat_loss']:.6f}",
        f"- dev numeric reconstruction loss: {final['dev_num_loss']:.6f}",
        "",
        "## Outputs",
        "",
        f"- `{checkpoint_path}`",
        f"- `{prep_path}`",
        f"- `{history_path}`",
    ]
    for path in exported:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "A true full-scale run should leave row-group and row-count debug options unset. CPU-only full-scale pretraining is expected to be substantially slower than sampled prototype runs.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {checkpoint_path}")
    print(f"wrote {history_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
