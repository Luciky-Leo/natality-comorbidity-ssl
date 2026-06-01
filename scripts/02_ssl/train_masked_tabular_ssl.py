#!/usr/bin/env python
"""Train a masked tabular transformer encoder on Natality input variables.

The script fits preprocessing only on training years, trains a masked
reconstruction encoder on 2016-2022 samples, reports 2023 development loss, and
optionally exports unmasked record embeddings for development/test samples.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "processed" / "natality_2016_2024_split_manifest.csv"
DEFAULT_OBJECT_DIR = PROJECT_ROOT / "results" / "objects"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "12_masked_tabular_ssl_report.md"
EXPORT_METADATA_COLS = [
    "source_year",
    "record_id",
    "outcome_maternal_morbidity_core",
    "outcome_severe_neonatal_no_nicu",
]


@dataclass
class Preprocessor:
    cat_cols: list[str]
    num_cols: list[str]
    cat_maps: dict[str, dict[str, int]]
    num_mean: dict[str, float]
    num_std: dict[str, float]

    def to_jsonable(self) -> dict[str, object]:
        return {
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "cat_maps": self.cat_maps,
            "num_mean": self.num_mean,
            "num_std": self.num_std,
        }


class TabularMaskedTransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities: list[int],
        n_numeric: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cat_cardinalities = cat_cardinalities
        self.n_cat = len(cat_cardinalities)
        self.n_numeric = n_numeric
        self.n_features = self.n_cat + self.n_numeric
        self.d_model = d_model

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, d_model) for cardinality in cat_cardinalities]
        )
        self.numeric_projections = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(n_numeric)]
        )
        self.feature_embedding = nn.Embedding(self.n_features, d_model)
        self.mask_embedding = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_embedding, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.cat_heads = nn.ModuleList(
            [nn.Linear(d_model, cardinality) for cardinality in cat_cardinalities]
        )
        self.numeric_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(n_numeric)])

    def encode(
        self,
        cat_x: torch.Tensor,
        num_x: torch.Tensor,
        cat_mask: torch.Tensor | None = None,
        num_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        for i, embedding in enumerate(self.cat_embeddings):
            tokens.append(embedding(cat_x[:, i]))
        for i, projection in enumerate(self.numeric_projections):
            tokens.append(projection(num_x[:, i : i + 1]))
        token_tensor = torch.stack(tokens, dim=1)
        positions = torch.arange(self.n_features, device=token_tensor.device)
        token_tensor = token_tensor + self.feature_embedding(positions).unsqueeze(0)

        if cat_mask is not None or num_mask is not None:
            masks = []
            if self.n_cat:
                masks.append(cat_mask)
            if self.n_numeric:
                masks.append(num_mask)
            full_mask = torch.cat(masks, dim=1)
            token_tensor = token_tensor + full_mask.unsqueeze(-1).float() * self.mask_embedding

        encoded = self.norm(self.encoder(token_tensor))
        record_embedding = encoded.mean(dim=1)
        return encoded, record_embedding

    def forward(
        self,
        cat_x: torch.Tensor,
        num_x: torch.Tensor,
        cat_mask: torch.Tensor,
        num_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        encoded, record_embedding = self.encode(cat_x, num_x, cat_mask, num_mask)
        cat_logits = [
            head(encoded[:, i, :]) for i, head in enumerate(self.cat_heads)
        ]
        numeric_preds = [
            head(encoded[:, self.n_cat + i, :]).squeeze(-1)
            for i, head in enumerate(self.numeric_heads)
        ]
        return cat_logits, numeric_preds, record_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--objects", type=Path, default=DEFAULT_OBJECT_DIR)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-train-per-year", type=int, default=20_000)
    parser.add_argument("--max-dev-rows", type=int, default=50_000)
    parser.add_argument("--max-test-rows", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--mask-rate", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional suffix for checkpoints, histories, reports, and embeddings.",
    )
    parser.add_argument("--export-embeddings", action="store_true")
    return parser.parse_args()


def tagged_path(path: Path, tag: str) -> Path:
    clean = tag.strip().replace(" ", "_")
    if not clean:
        return path
    return path.with_name(f"{path.stem}_{clean}{path.suffix}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def input_columns_from_schema(path: Path) -> list[str]:
    names = pq.ParquetFile(path).schema.names
    return [
        column
        for column in names
        if column.startswith("input_") or column.startswith("missing_input_")
    ]


def load_split(
    manifest: pd.DataFrame,
    split: str,
    columns: list[str],
    seed: int,
    max_per_year: int | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    frames = []
    for row in manifest[manifest["split"] == split].to_dict("records"):
        path = Path(row["path"])
        frame = pq.read_table(path, columns=columns).to_pandas()
        if max_per_year is not None and len(frame) > max_per_year:
            frame = frame.sample(n=max_per_year, random_state=seed + int(row["year"]))
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if max_rows is not None and len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return data.reset_index(drop=True)


def fit_preprocessor(train: pd.DataFrame, max_categories: int = 128) -> Preprocessor:
    cat_cols = []
    num_cols = []
    for column in train.columns:
        if pd.api.types.is_object_dtype(train[column]) or isinstance(
            train[column].dtype, pd.StringDtype
        ):
            cat_cols.append(column)
        else:
            num_cols.append(column)

    cat_maps: dict[str, dict[str, int]] = {}
    for column in cat_cols:
        series = train[column].astype("string").fillna("__MISSING__")
        counts = series.value_counts(dropna=False).head(max_categories)
        values = ["__MISSING__"] + [value for value in counts.index if value != "__MISSING__"]
        cat_maps[column] = {value: idx for idx, value in enumerate(dict.fromkeys(values))}

    num_mean: dict[str, float] = {}
    num_std: dict[str, float] = {}
    for column in num_cols:
        values = pd.to_numeric(train[column], errors="coerce").astype("float32")
        mean = float(values.mean(skipna=True))
        std = float(values.std(skipna=True))
        if not np.isfinite(mean):
            mean = 0.0
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        num_mean[column] = mean
        num_std[column] = std

    return Preprocessor(cat_cols, num_cols, cat_maps, num_mean, num_std)


def transform_frame(frame: pd.DataFrame, prep: Preprocessor) -> tuple[np.ndarray, np.ndarray]:
    cat_arrays = []
    for column in prep.cat_cols:
        mapping = prep.cat_maps[column]
        unk = len(mapping)
        values = frame[column].astype("string").fillna("__MISSING__")
        cat_arrays.append(values.map(mapping).fillna(unk).astype("int64").to_numpy())
    if cat_arrays:
        cat_x = np.stack(cat_arrays, axis=1)
    else:
        cat_x = np.zeros((len(frame), 0), dtype=np.int64)

    num_arrays = []
    for column in prep.num_cols:
        values = pd.to_numeric(frame[column], errors="coerce").astype("float32")
        standardized = ((values - prep.num_mean[column]) / prep.num_std[column]).fillna(0.0)
        num_arrays.append(standardized.to_numpy(dtype=np.float32))
    if num_arrays:
        num_x = np.stack(num_arrays, axis=1)
    else:
        num_x = np.zeros((len(frame), 0), dtype=np.float32)
    return cat_x, num_x


def make_loader(
    cat_x: np.ndarray,
    num_x: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(cat_x.astype(np.int64)),
        torch.from_numpy(num_x.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def make_masks(
    cat_x: torch.Tensor,
    num_x: torch.Tensor,
    cat_mask_ids: list[int],
    mask_rate: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cat_x = cat_x.to(device)
    num_x = num_x.to(device)
    cat_mask = torch.rand(cat_x.shape, device=device) < mask_rate if cat_x.numel() else torch.zeros_like(cat_x, dtype=torch.bool)
    num_mask = torch.rand(num_x.shape, device=device) < mask_rate if num_x.numel() else torch.zeros_like(num_x, dtype=torch.bool)

    cat_masked = cat_x.clone()
    for i, mask_id in enumerate(cat_mask_ids):
        cat_masked[:, i] = torch.where(cat_mask[:, i], torch.full_like(cat_masked[:, i], mask_id), cat_masked[:, i])
    num_masked = num_x.clone()
    num_masked[num_mask] = 0.0
    return cat_masked, num_masked, cat_mask, num_mask


def masked_loss(
    cat_logits: list[torch.Tensor],
    numeric_preds: list[torch.Tensor],
    cat_targets: torch.Tensor,
    num_targets: torch.Tensor,
    cat_mask: torch.Tensor,
    num_mask: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    losses = []
    cat_loss_value = 0.0
    num_loss_value = 0.0
    cat_terms = 0
    num_terms = 0

    for i, logits in enumerate(cat_logits):
        mask = cat_mask[:, i]
        if mask.any():
            loss = F.cross_entropy(logits[mask], cat_targets[:, i][mask])
            losses.append(loss)
            cat_loss_value += float(loss.detach().cpu())
            cat_terms += 1
    for i, pred in enumerate(numeric_preds):
        mask = num_mask[:, i]
        if mask.any():
            loss = F.mse_loss(pred[mask], num_targets[:, i][mask])
            losses.append(loss)
            num_loss_value += float(loss.detach().cpu())
            num_terms += 1
    if not losses:
        return torch.tensor(0.0, device=cat_targets.device), 0.0, 0.0
    total = torch.stack(losses).mean()
    return total, cat_loss_value / max(cat_terms, 1), num_loss_value / max(num_terms, 1)


def run_epoch(
    model: TabularMaskedTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    cat_mask_ids: list[int],
    mask_rate: float,
    device: torch.device,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    totals = {"loss": 0.0, "cat_loss": 0.0, "num_loss": 0.0, "n": 0}
    for cat_batch, num_batch in loader:
        cat_batch = cat_batch.to(device)
        num_batch = num_batch.to(device)
        cat_masked, num_masked, cat_mask, num_mask = make_masks(
            cat_batch, num_batch, cat_mask_ids, mask_rate, device
        )
        with torch.set_grad_enabled(train_mode):
            cat_logits, numeric_preds, _ = model(cat_masked, num_masked, cat_mask, num_mask)
            loss, cat_loss, num_loss = masked_loss(
                cat_logits, numeric_preds, cat_batch, num_batch, cat_mask, num_mask
            )
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_n = len(cat_batch)
        totals["loss"] += float(loss.detach().cpu()) * batch_n
        totals["cat_loss"] += cat_loss * batch_n
        totals["num_loss"] += num_loss * batch_n
        totals["n"] += batch_n
    return {key: value / totals["n"] for key, value in totals.items() if key != "n"}


def export_embeddings(
    model: TabularMaskedTransformer,
    frame: pd.DataFrame,
    prep: Preprocessor,
    output_path: Path,
    batch_size: int,
    device: torch.device,
    metadata_cols: list[str] | None = None,
) -> None:
    cat_x, num_x = transform_frame(frame[prep.cat_cols + prep.num_cols], prep)
    loader = make_loader(cat_x, num_x, batch_size=batch_size, shuffle=False)
    embeddings = []
    model.eval()
    with torch.no_grad():
        for cat_batch, num_batch in loader:
            cat_batch = cat_batch.to(device)
            num_batch = num_batch.to(device)
            cat_mask = torch.zeros_like(cat_batch, dtype=torch.bool, device=device)
            num_mask = torch.zeros_like(num_batch, dtype=torch.bool, device=device)
            _, record_embedding = model.encode(cat_batch, num_batch, cat_mask, num_mask)
            embeddings.append(record_embedding.cpu().numpy())
    arr = np.concatenate(embeddings, axis=0)
    columns = [f"ssl_emb_{i:03d}" for i in range(arr.shape[1])]
    embedding_frame = pd.DataFrame(arr, columns=columns)
    if metadata_cols:
        metadata = frame[metadata_cols].reset_index(drop=True)
        embedding_frame = pd.concat([metadata, embedding_frame], axis=1)
    embedding_frame.to_parquet(output_path, index=False)


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
    print("load train")
    train = load_split(
        manifest,
        "train",
        input_cols,
        args.seed,
        max_per_year=args.max_train_per_year,
    )
    print(f"train rows: {len(train):,}")
    print("load development")
    dev = load_split(
        manifest,
        "development",
        input_cols,
        args.seed,
        max_rows=args.max_dev_rows,
    )
    print(f"dev rows: {len(dev):,}")

    prep = fit_preprocessor(train)
    cat_train, num_train = transform_frame(train, prep)
    cat_dev, num_dev = transform_frame(dev, prep)
    cat_cardinalities = [len(prep.cat_maps[column]) + 1 for column in prep.cat_cols]
    cat_mask_ids = list(cat_cardinalities)
    print(
        f"categorical={len(prep.cat_cols)}, numeric={len(prep.num_cols)}, train={cat_train.shape[0]:,}, dev={cat_dev.shape[0]:,}"
    )

    train_loader = make_loader(cat_train, num_train, args.batch_size, shuffle=True)
    dev_loader = make_loader(cat_dev, num_dev, args.batch_size, shuffle=False)
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
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, cat_mask_ids, args.mask_rate, device)
        dev_metrics = run_epoch(model, dev_loader, None, cat_mask_ids, args.mask_rate, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"dev_{k}": v for k, v in dev_metrics.items()}}
        history_rows.append(row)
        print(
            f"epoch {epoch}: train_loss={row['train_loss']:.4f}, dev_loss={row['dev_loss']:.4f}, dev_cat={row['dev_cat_loss']:.4f}, dev_num={row['dev_num_loss']:.4f}",
            flush=True,
        )

    history = pd.DataFrame(history_rows)
    history_path = tagged_path(args.tables / "masked_tabular_ssl_history.csv", args.output_tag)
    history.to_csv(history_path, index=False)
    prep_path = tagged_path(args.objects / "masked_tabular_ssl_preprocessor.json", args.output_tag)
    prep_path.write_text(json.dumps(prep.to_jsonable(), indent=2), encoding="utf-8")
    checkpoint_path = tagged_path(args.objects / "masked_tabular_ssl_encoder.pt", args.output_tag)
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
        dev_embedding_path = tagged_path(args.objects / "masked_tabular_ssl_dev_embeddings.parquet", args.output_tag)
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
        print(f"wrote {dev_embedding_path}")

        print("load test for embedding export")
        test = load_split(
            manifest,
            "test",
            export_columns,
            args.seed,
            max_rows=args.max_test_rows,
        )
        test_embedding_path = tagged_path(args.objects / "masked_tabular_ssl_test_embeddings.parquet", args.output_tag)
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
        print(f"wrote {test_embedding_path}")

    final = history_rows[-1]
    report_lines = [
        "# Masked Tabular SSL Report",
        "",
        "Pretraining split: 2016-2022 only.",
        "Development split: 2023 only.",
        "Final test split: not used for model fitting or hyperparameter selection.",
        "",
        "## Configuration",
        "",
        f"- Python/PyTorch device: {device}",
        f"- train rows: {len(train):,}",
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
        report_lines.append(f"- `{path}`")
    report_lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is the first masked reconstruction encoder. It establishes the SSL training pipeline; downstream phenotype clustering and risk-model comparison must be run separately and should use 2023 for model selection before final 2024 evaluation.",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"wrote {checkpoint_path}")
    print(f"wrote {history_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
