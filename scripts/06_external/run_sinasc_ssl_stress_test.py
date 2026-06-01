#!/usr/bin/env python
"""Run an independent SINASC masked-tabular SSL registry stress-test."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "data" / "processed" / "sinasc" / "sinasc_2023_2024_split_manifest.csv"
OBJECT_DIR = PROJECT_ROOT / "results" / "objects"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
DOC_DIR = PROJECT_ROOT / "docs"
SSL_LIB_PATH = PROJECT_ROOT / "scripts" / "02_ssl" / "train_masked_tabular_ssl.py"

ENDPOINTS = [
    "outcome_sinasc_severe_birth_status",
    "outcome_sinasc_broad_birth_status",
    "outcome_low_apgar5_lt7",
    "outcome_congenital_anomaly",
]

PROFILE_FEATURES = [
    "input_IDADEMAE",
    "input_CONSPRENAT",
    "input_QTDPARTCES",
    "input_prior_cesarean",
    "input_multiple_gestation",
    "input_GRAVIDEZ",
    "input_PARIDADE",
    "input_ESCMAE2010",
    "input_RACACORMAE",
    "input_MESPRENAT",
]


def load_ssl_lib():
    spec = importlib.util.spec_from_file_location("masked_tabular_ssl_lib", SSL_LIB_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SSL_LIB_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-tag", default="sinasc_2023train_2024test_ssl")
    parser.add_argument("--max-train-rows", type=int, default=500_000)
    parser.add_argument("--max-dev-rows", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--mask-rate", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260528)
    return parser.parse_args()


def tag_path(base: Path, tag: str) -> Path:
    return base.with_name(f"{base.stem}_{tag}{base.suffix}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def input_columns(path: Path) -> list[str]:
    names = pq.ParquetFile(path).schema.names
    return [name for name in names if name.startswith("input_") or name.startswith("missing_input_")]


def load_frame(path: Path, columns: list[str], max_rows: int | None, seed: int) -> pd.DataFrame:
    frame = pq.read_table(path, columns=columns).to_pandas()
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return frame.reset_index(drop=True)


def split_2023(path: Path, columns: list[str], train_n: int, dev_n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pq.read_table(path, columns=columns).to_pandas()
    rng = np.random.default_rng(seed)
    total_n = min(len(frame), train_n + dev_n)
    idx = rng.choice(len(frame), size=total_n, replace=False)
    train_idx = idx[: min(train_n, total_n)]
    dev_idx = idx[min(train_n, total_n) :]
    return frame.iloc[train_idx].reset_index(drop=True), frame.iloc[dev_idx].reset_index(drop=True)


def embedding_cols(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col.startswith("ssl_emb_")]


def fit_kmeans(dev_emb: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, Pipeline, KMeans, pd.DataFrame]:
    cols = embedding_cols(dev_emb)
    x = dev_emb[cols].to_numpy(dtype=np.float32)
    n_components = min(args.pca_components, x.shape[1])
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=args.seed)
    x_reduced = pca.fit_transform(scaler.fit_transform(x))
    rows = []
    models: dict[int, KMeans] = {}
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(x_reduced), size=min(10000, len(x_reduced)), replace=False)
    for k in range(args.k_min, args.k_max + 1):
        model = KMeans(n_clusters=k, random_state=args.seed, n_init=20)
        labels = model.fit_predict(x_reduced)
        models[k] = model
        counts = np.bincount(labels, minlength=k)
        rows.append(
            {
                "k": k,
                "silhouette_sample": float(silhouette_score(x_reduced[sample_idx], labels[sample_idx])),
                "davies_bouldin": float(davies_bouldin_score(x_reduced, labels)),
                "min_cluster_prop": float(counts.min() / counts.sum()),
                "cluster_sizes": ";".join(str(int(v)) for v in counts),
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            }
        )
    selection = pd.DataFrame(rows)
    candidate = selection[selection["min_cluster_prop"].ge(0.03)]
    if candidate.empty:
        candidate = selection
    best_k = int(candidate.sort_values(["silhouette_sample", "min_cluster_prop"], ascending=False).iloc[0]["k"])
    pipeline = Pipeline([("scaler", scaler), ("pca", pca)])
    dev_labels = models[best_k].labels_
    dev_out = dev_emb.copy()
    dev_out["phenotype"] = dev_labels.astype(int)
    return dev_out, pipeline, models[best_k], selection


def assign_labels(emb: pd.DataFrame, pipeline: Pipeline, model: KMeans) -> pd.DataFrame:
    x = emb[embedding_cols(emb)].to_numpy(dtype=np.float32)
    reduced = pipeline.transform(x)
    labels = np.argmin(model.transform(reduced), axis=1)
    out = emb.copy()
    out["phenotype"] = labels.astype(int)
    return out


def topk_metrics(y: np.ndarray, prob: np.ndarray, fraction: float = 0.01) -> dict[str, float]:
    n = len(y)
    k = max(1, int(round(n * fraction)))
    order = np.argsort(prob)[::-1][:k]
    prevalence = float(y.mean())
    event_rate = float(y[order].mean())
    return {
        "top_fraction": fraction,
        "top_n": k,
        "top_event_rate": event_rate,
        "top_enrichment_over_prevalence": event_rate / prevalence if prevalence > 0 else np.nan,
        "top_event_capture_pct": float(y[order].sum() / y.sum() * 100) if y.sum() > 0 else np.nan,
    }


def fit_platt(y_cal: np.ndarray, p_cal: np.ndarray) -> LogisticRegression:
    p_cal = np.clip(p_cal, 1e-6, 1 - 1e-6)
    logits = np.log(p_cal / (1 - p_cal)).reshape(-1, 1)
    model = LogisticRegression(max_iter=1000)
    model.fit(logits, y_cal)
    return model


def apply_platt(model: LogisticRegression, prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    logits = np.log(prob / (1 - prob)).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def metric_row(endpoint: str, model_name: str, y: np.ndarray, prob: np.ndarray, n_train: int) -> dict[str, object]:
    top = topk_metrics(y, prob)
    row = {
        "endpoint": endpoint,
        "model": model_name,
        "n_train": n_train,
        "n_test": int(len(y)),
        "events_test": int(y.sum()),
        "prevalence_test": float(y.mean()),
        "auroc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else np.nan,
        "auprc": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }
    row["auprc_over_prevalence"] = row["auprc"] / row["prevalence_test"] if row["prevalence_test"] > 0 else np.nan
    row.update(top)
    return row


def evaluate_ssl_models(dev: pd.DataFrame, test: pd.DataFrame, endpoints: list[str], seed: int) -> pd.DataFrame:
    rows = []
    emb_cols = embedding_cols(dev)
    for endpoint in endpoints:
        dev_ep = dev.dropna(subset=[endpoint]).reset_index(drop=True)
        test_ep = test.dropna(subset=[endpoint]).reset_index(drop=True)
        y_dev = dev_ep[endpoint].astype("int8").to_numpy()
        y_test = test_ep[endpoint].astype("int8").to_numpy()
        train_idx, cal_idx = train_test_split(np.arange(len(dev_ep)), test_size=0.40, random_state=seed, stratify=y_dev)

        feature_sets = {
            "ssl_embedding_logit": emb_cols,
            "ssl_plus_phenotype_logit": emb_cols + ["phenotype"],
            "phenotype_only_logit": ["phenotype"],
        }
        for model_name, cols in feature_sets.items():
            x_dev = dev_ep[cols].to_numpy(dtype=np.float32)
            x_test = test_ep[cols].to_numpy(dtype=np.float32)
            clf = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
                ]
            )
            clf.fit(x_dev[train_idx], y_dev[train_idx])
            p_cal = clf.predict_proba(x_dev[cal_idx])[:, 1]
            platt = fit_platt(y_dev[cal_idx], p_cal)
            prob = apply_platt(platt, clf.predict_proba(x_test)[:, 1])
            rows.append(metric_row(endpoint, model_name, y_test, prob, len(train_idx)))
    return pd.DataFrame(rows)


def make_input_design(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    inputs = [col for col in train.columns if col.startswith("input_") or col.startswith("missing_input_")]
    for frame in (train, test):
        for col in inputs:
            if not pd.api.types.is_numeric_dtype(frame[col]) and not pd.api.types.is_bool_dtype(frame[col]):
                frame[col] = frame[col].astype("category")
    return train[inputs], test[inputs], inputs


def evaluate_lightgbm_inputs(dev: pd.DataFrame, test: pd.DataFrame, endpoints: list[str], seed: int) -> pd.DataFrame:
    if lgb is None:
        return pd.DataFrame()
    rows = []
    for endpoint in endpoints:
        dev_ep = dev.dropna(subset=[endpoint]).reset_index(drop=True)
        test_ep = test.dropna(subset=[endpoint]).reset_index(drop=True)
        y_dev = dev_ep[endpoint].astype("int8").to_numpy()
        y_test = test_ep[endpoint].astype("int8").to_numpy()
        train_idx, cal_idx = train_test_split(np.arange(len(dev_ep)), test_size=0.40, random_state=seed, stratify=y_dev)
        x_dev, x_test, inputs = make_input_design(dev_ep.copy(), test_ep.copy())
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
            verbosity=-1,
        )
        clf.fit(x_dev.iloc[train_idx], y_dev[train_idx], categorical_feature="auto")
        p_cal = clf.predict_proba(x_dev.iloc[cal_idx])[:, 1]
        platt = fit_platt(y_dev[cal_idx], p_cal)
        prob = apply_platt(platt, clf.predict_proba(x_test)[:, 1])
        rows.append(metric_row(endpoint, "sinasc_overlap_inputs_lightgbm", y_test, prob, len(train_idx)))
    return pd.DataFrame(rows)


def phenotype_rate_table(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for phenotype, group in frame.groupby("phenotype", sort=True):
        row = {
            "split": split,
            "phenotype": int(phenotype),
            "n": int(len(group)),
            "proportion": float(len(group) / len(frame)),
        }
        for endpoint in ENDPOINTS:
            known = group[endpoint].dropna().astype("int8")
            row[f"{endpoint}_known_n"] = int(len(known))
            row[f"{endpoint}_positive_n"] = int(known.sum())
            row[f"{endpoint}_rate"] = float(known.mean()) if len(known) else np.nan
        for feature in ["input_IDADEMAE", "input_CONSPRENAT", "input_QTDPARTCES", "input_prior_cesarean", "input_multiple_gestation"]:
            row[f"{feature}_mean"] = float(pd.to_numeric(group[feature], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_figure(metrics: pd.DataFrame, pheno_rates: pd.DataFrame, selection: pd.DataFrame, history: pd.DataFrame, path: Path) -> None:
    model_labels = {
        "sinasc_overlap_inputs_lightgbm": "Inputs\nLightGBM",
        "ssl_embedding_logit": "SSL\nlogit",
        "ssl_plus_phenotype_logit": "SSL + phenotype\nlogit",
        "phenotype_only_logit": "Phenotype\nonly",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    ax = axes[0, 0]
    hist = history
    ax.plot(hist["epoch"], hist["train_loss"], marker="o", label="Train")
    ax.plot(hist["epoch"], hist["dev_loss"], marker="o", label="Development")
    ax.set_title("Masked reconstruction loss", loc="left")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    test_rates = pheno_rates[pheno_rates["split"].eq("test")].sort_values("phenotype")
    x = np.arange(len(test_rates))
    ax.bar(x - 0.18, test_rates["outcome_sinasc_severe_birth_status_rate"] * 100, width=0.36, label="Severe")
    ax.bar(x + 0.18, test_rates["outcome_sinasc_broad_birth_status_rate"] * 100, width=0.36, label="Broad")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{int(v)}" for v in test_rates["phenotype"]])
    ax.set_title("2024 outcome rate by phenotype", loc="left")
    ax.set_ylabel("Event rate (%)")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    plot_metrics = metrics[metrics["endpoint"].isin(["outcome_sinasc_severe_birth_status", "outcome_sinasc_broad_birth_status"])]
    pivot = plot_metrics.pivot_table(index="model", columns="endpoint", values="auprc", aggfunc="first")
    pivot = pivot.reindex(["sinasc_overlap_inputs_lightgbm", "ssl_embedding_logit", "ssl_plus_phenotype_logit", "phenotype_only_logit"])
    pivot.index = [model_labels.get(item, item) for item in pivot.index]
    pivot.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
    ax.set_title("2024 AUPRC stress-test", loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("AUPRC")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(["Severe", "Broad"], frameon=False)

    ax = axes[1, 1]
    ax.plot(selection["k"], selection["silhouette_sample"], marker="o", label="Silhouette")
    ax2 = ax.twinx()
    ax2.plot(selection["k"], selection["min_cluster_prop"], marker="s", color="#F58518", label="Min cluster prop.")
    ax.set_title("Development-only cluster diagnostics", loc="left")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    ax2.set_ylabel("Min cluster proportion")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="best")

    for a in axes.flat:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    ssl_lib = load_ssl_lib()
    OBJECT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    train_path = Path(manifest.loc[manifest["split"].eq("development"), "path"].iloc[0])
    test_path = Path(manifest.loc[manifest["split"].eq("test"), "path"].iloc[0])
    inputs = input_columns(train_path)
    metadata_cols = ["source_year", "record_id"] + ENDPOINTS + PROFILE_FEATURES
    columns = list(dict.fromkeys(inputs + metadata_cols))

    print("load/split 2023", flush=True)
    train, dev = split_2023(train_path, columns, args.max_train_rows, args.max_dev_rows, args.seed)
    print(f"train={len(train):,} dev={len(dev):,}", flush=True)
    prep = ssl_lib.fit_preprocessor(train[inputs])
    cat_train, num_train = ssl_lib.transform_frame(train[inputs], prep)
    cat_dev, num_dev = ssl_lib.transform_frame(dev[inputs], prep)
    cat_cardinalities = [len(prep.cat_maps[col]) + 1 for col in prep.cat_cols]
    cat_mask_ids = list(cat_cardinalities)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ssl_lib.TabularMaskedTransformer(
        cat_cardinalities=cat_cardinalities,
        n_numeric=len(prep.num_cols),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = ssl_lib.make_loader(cat_train, num_train, args.batch_size, shuffle=True)
    dev_loader = ssl_lib.make_loader(cat_dev, num_dev, args.batch_size, shuffle=False)

    history_rows = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = ssl_lib.run_epoch(model, train_loader, optimizer, cat_mask_ids, args.mask_rate, device)
        dev_metrics = ssl_lib.run_epoch(model, dev_loader, None, cat_mask_ids, args.mask_rate, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"dev_{k}": v for k, v in dev_metrics.items()}}
        history_rows.append(row)
        print(f"epoch {epoch}: train_loss={row['train_loss']:.4f} dev_loss={row['dev_loss']:.4f}", flush=True)
    history = pd.DataFrame(history_rows)

    checkpoint_path = tag_path(OBJECT_DIR / "sinasc_masked_tabular_ssl_encoder.pt", args.output_tag)
    prep_path = tag_path(OBJECT_DIR / "sinasc_masked_tabular_ssl_preprocessor.json", args.output_tag)
    history_path = tag_path(TABLE_DIR / "sinasc_masked_tabular_ssl_history.csv", args.output_tag)
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
    prep_path.write_text(json.dumps(prep.to_jsonable(), indent=2), encoding="utf-8")
    history.to_csv(history_path, index=False)

    print("export dev embeddings", flush=True)
    dev_emb_path = tag_path(OBJECT_DIR / "sinasc_ssl_dev_embeddings.parquet", args.output_tag)
    ssl_lib.export_embeddings(model, dev, prep, dev_emb_path, args.batch_size, device, metadata_cols=metadata_cols)

    print("load 2024 and export embeddings", flush=True)
    test = load_frame(test_path, columns, None, args.seed)
    test_emb_path = tag_path(OBJECT_DIR / "sinasc_ssl_test_embeddings.parquet", args.output_tag)
    ssl_lib.export_embeddings(model, test, prep, test_emb_path, args.batch_size, device, metadata_cols=metadata_cols)

    dev_emb = pd.read_parquet(dev_emb_path)
    test_emb = pd.read_parquet(test_emb_path)
    dev_labeled, pca_pipeline, kmeans, selection = fit_kmeans(dev_emb, args)
    test_labeled = assign_labels(test_emb, pca_pipeline, kmeans)
    dev_labeled_path = tag_path(OBJECT_DIR / "sinasc_ssl_dev_embeddings_phenotypes.parquet", args.output_tag)
    test_labeled_path = tag_path(OBJECT_DIR / "sinasc_ssl_test_embeddings_phenotypes.parquet", args.output_tag)
    dev_labeled.to_parquet(dev_labeled_path, index=False)
    test_labeled.to_parquet(test_labeled_path, index=False)

    print("evaluate SSL models", flush=True)
    ssl_metrics = evaluate_ssl_models(dev_labeled, test_labeled, ENDPOINTS, args.seed)
    print("evaluate input LightGBM baseline", flush=True)
    input_metrics = evaluate_lightgbm_inputs(dev, test, ENDPOINTS, args.seed)
    metrics = pd.concat([input_metrics, ssl_metrics], ignore_index=True)

    pheno_rates = pd.concat(
        [
            phenotype_rate_table(dev_labeled, "development"),
            phenotype_rate_table(test_labeled, "test"),
        ],
        ignore_index=True,
    )
    metrics_path = tag_path(TABLE_DIR / "sinasc_ssl_stress_test_metrics.csv", args.output_tag)
    pheno_path = tag_path(TABLE_DIR / "sinasc_ssl_phenotype_rates.csv", args.output_tag)
    selection_path = tag_path(TABLE_DIR / "sinasc_ssl_cluster_selection.csv", args.output_tag)
    metadata_path = tag_path(TABLE_DIR / "sinasc_ssl_stress_test_metadata.json", args.output_tag)
    figure_path = tag_path(FIGURE_DIR / "sinasc_ssl_stress_test_summary.png", args.output_tag)
    report_path = tag_path(DOC_DIR / "45_sinasc_ssl_stress_test_report.md", args.output_tag)
    metrics.to_csv(metrics_path, index=False)
    pheno_rates.to_csv(pheno_path, index=False)
    selection.to_csv(selection_path, index=False)
    build_figure(metrics, pheno_rates, selection, history, figure_path)

    best_k = int(kmeans.n_clusters)
    severe_best = metrics[metrics["endpoint"].eq("outcome_sinasc_severe_birth_status")].sort_values("auprc", ascending=False).head(1)
    high_pheno = (
        pheno_rates[pheno_rates["split"].eq("test")]
        .sort_values("outcome_sinasc_severe_birth_status_rate", ascending=False)
        .head(1)
        .iloc[0]
    )
    metadata = {
        "boundary": "Independent SINASC registry stress-test; not direct external validation of the U.S. Natality encoder.",
        "train_year": 2023,
        "test_year": 2024,
        "train_rows_sampled": len(train),
        "development_rows_sampled": len(dev),
        "test_rows_full": len(test),
        "d_model": args.d_model,
        "epochs": args.epochs,
        "mask_rate": args.mask_rate,
        "selected_k": best_k,
        "outputs": {
            "checkpoint": str(checkpoint_path),
            "preprocessor": str(prep_path),
            "history": str(history_path),
            "dev_embeddings": str(dev_emb_path),
            "test_embeddings": str(test_emb_path),
            "dev_phenotypes": str(dev_labeled_path),
            "test_phenotypes": str(test_labeled_path),
            "metrics": str(metrics_path),
            "phenotype_rates": str(pheno_path),
            "cluster_selection": str(selection_path),
            "figure": str(figure_path.with_suffix(".png")),
            "report": str(report_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lines = [
        "# SINASC independent registry SSL stress-test",
        "",
        "## Boundary",
        "",
        "This analysis trains and evaluates a separate masked tabular SSL model within Brazil SINASC public-use birth records. It is an independent registry workflow stress-test, not direct external validation of the U.S. Natality encoder.",
        "",
        "## Data split",
        "",
        f"- 2023 sampled SSL training rows: {len(train):,}",
        f"- 2023 development/calibration rows: {len(dev):,}",
        f"- 2024 full temporal test rows: {len(test):,}",
        "",
        "## SSL training",
        "",
        f"- Device: {device}",
        f"- Input features including missingness flags: {len(inputs)}",
        f"- Categorical features: {len(prep.cat_cols)}",
        f"- Numeric features: {len(prep.num_cols)}",
        f"- Final development reconstruction loss: {history.iloc[-1]['dev_loss']:.4f}",
        f"- Selected phenotype count k: {best_k}",
        "",
        "## Main 2024 stress-test result",
        "",
    ]
    if not severe_best.empty:
        row = severe_best.iloc[0]
        lines.append(
            f"- Best severe birth-status AUPRC model: {row['model']} (AUPRC {row['auprc']:.4f}, "
            f"AUPRC/prevalence {row['auprc_over_prevalence']:.2f}, top 1% enrichment {row['top_enrichment_over_prevalence']:.2f})."
        )
    lines.append(
        f"- Highest-risk phenotype for severe birth-status outcome: P{int(high_pheno['phenotype'])}, "
        f"n={int(high_pheno['n']):,}, severe event rate={100*high_pheno['outcome_sinasc_severe_birth_status_rate']:.2f}%."
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "SINASC supports an independent stress-test of the workflow using overlapping registry variables and birth-status endpoints. The result should be described as transport of the analytical framework to an independent public registry. It should not be described as transport of the trained U.S. model, because the SINASC input schema lacks several U.S. Natality feature families.",
            "",
            "## Outputs",
            "",
            f"- Metrics: `{metrics_path}`",
            f"- Phenotype rates: `{pheno_path}`",
            f"- Cluster selection: `{selection_path}`",
            f"- Figure: `{figure_path.with_suffix('.png')}`",
            f"- Metadata: `{metadata_path}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(metrics_path)
    print(pheno_path)
    print(selection_path)
    print(figure_path.with_suffix(".png"))
    print(report_path)


if __name__ == "__main__":
    main()
