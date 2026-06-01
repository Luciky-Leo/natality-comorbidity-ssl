#!/usr/bin/env python
"""Run reviewer-requested subgroup, utility, and interpretation extensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "processed" / "natality_2016_2024_split_manifest.csv"
DEFAULT_DEV_EMB = PROJECT_ROOT / "results" / "objects" / "masked_tabular_ssl_dev_embeddings_200k.parquet"
DEFAULT_TEST_EMB = PROJECT_ROOT / "results" / "objects" / "masked_tabular_ssl_test_embeddings_200k.parquet"
DEFAULT_DEV_ASSIGN = PROJECT_ROOT / "results" / "objects" / "ssl_phenotype_dev_assignments_200k.parquet"
DEFAULT_TEST_ASSIGN = PROJECT_ROOT / "results" / "objects" / "ssl_phenotype_test_assignments_200k.parquet"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_OBJECT_DIR = PROJECT_ROOT / "results" / "objects"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "26_cns_extension_analysis_report.md"

PRIMARY_ENDPOINTS = [
    "outcome_maternal_morbidity_core",
    "outcome_severe_neonatal_no_nicu",
]

ENDPOINT_LABEL = {
    "outcome_maternal_morbidity_core": "Maternal core morbidity",
    "outcome_severe_neonatal_no_nicu": "Severe neonatal outcome",
}

COMORBIDITY_FEATURE_BASES = [
    "MAGER",
    "MAGER9",
    "BMI",
    "BMI_R",
    "WTGAIN",
    "WTGAIN_REC",
    "CIG0_R",
    "CIG1_R",
    "CIG2_R",
    "CIG3_R",
    "CIG_REC",
    "RF_PDIAB",
    "RF_GDIAB",
    "RF_PHYPE",
    "RF_GHYPE",
    "RF_EHYPE",
    "RF_PPTERM",
    "RF_INFTR",
    "RF_FEDRG",
    "RF_ARTEC",
    "RF_CESAR",
    "RF_CESARN",
    "NO_RISKS",
    "IP_GON",
    "IP_SYPH",
    "IP_CHLAM",
    "IP_HEPB",
    "IP_HEPC",
    "NO_INFEC",
    "DPLURAL",
]

SUBGROUP_RAW_COLUMNS = [
    "input_MAGER",
    "input_BMI",
    "input_PREVIS",
    "input_WTGAIN",
    "input_MRACEHISP",
    "input_DMAR",
    "input_MEDUC",
    "input_WIC",
    "input_PRECARE5",
    "input_RF_PDIAB",
    "input_RF_GDIAB",
    "input_RF_PHYPE",
    "input_RF_GHYPE",
    "input_RF_EHYPE",
    "input_RF_PPTERM",
    "input_RF_INFTR",
    "input_RF_FEDRG",
    "input_RF_ARTEC",
    "input_RF_CESAR",
    "input_CIG_REC",
    "input_DPLURAL",
    "input_IP_CHLAM",
    "input_IP_HEPB",
    "input_IP_HEPC",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--dev-embeddings", type=Path, default=DEFAULT_DEV_EMB)
    parser.add_argument("--test-embeddings", type=Path, default=DEFAULT_TEST_EMB)
    parser.add_argument("--dev-assignments", type=Path, default=DEFAULT_DEV_ASSIGN)
    parser.add_argument("--test-assignments", type=Path, default=DEFAULT_TEST_ASSIGN)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--objects", type=Path, default=DEFAULT_OBJECT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-train-per-year", type=int, default=500_000)
    parser.add_argument("--lgbm-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--reuse-predictions", action="store_true")
    parser.add_argument(
        "--output-tag",
        default="200k",
        help="Suffix for output files, for example full2016_2022_mask035_d48_l2_cuda.",
    )
    return parser.parse_args()


def tagged_name(stem: str, tag: str, suffix: str) -> str:
    return f"{stem}_{tag}.{suffix}"


def feature_sets_from_schema(path: Path) -> dict[str, list[str]]:
    names = pq.ParquetFile(path).schema.names
    all_inputs = [column for column in names if column.startswith("input_") or column.startswith("missing_input_")]
    base_names = {f"input_{name}" for name in COMORBIDITY_FEATURE_BASES}
    missing_names = {f"missing_input_{name}" for name in COMORBIDITY_FEATURE_BASES}
    comorbidity = [column for column in all_inputs if column in base_names or column in missing_names]
    return {"all_inputs": all_inputs, "comorbidity_only": comorbidity}


def load_train(manifest: pd.DataFrame, columns: list[str], seed: int, max_per_year: int) -> pd.DataFrame:
    frames = []
    for row in manifest[manifest["split"] == "train"].to_dict("records"):
        frame = pq.read_table(Path(row["path"]), columns=columns).to_pandas()
        if len(frame) > max_per_year:
            frame = frame.sample(n=max_per_year, random_state=seed + int(row["year"]))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_exact_records(manifest: pd.DataFrame, split: str, columns: list[str], id_source: Path, keep_id: bool = False) -> pd.DataFrame:
    wanted = pd.read_parquet(id_source, columns=["record_id"])
    wanted["record_id"] = wanted["record_id"].astype("int64")
    wanted_ids = set(wanted["record_id"].tolist())
    path = Path(manifest.loc[manifest["split"] == split, "path"].iloc[0])
    frame = pq.read_table(path, columns=["record_id"] + columns).to_pandas()
    frame["record_id"] = frame["record_id"].astype("int64")
    frame = frame[frame["record_id"].isin(wanted_ids)].copy()
    frame = wanted.merge(frame, on="record_id", how="left")
    if frame[columns].isna().all(axis=1).any():
        missing = int(frame[columns].isna().all(axis=1).sum())
        raise RuntimeError(f"missing exact records after merge: {missing}")
    return frame if keep_id else frame.drop(columns=["record_id"])


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


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("ssl_emb_")]


def make_ssl_design(frame: pd.DataFrame, emb_cols: list[str]) -> pd.DataFrame:
    phen = pd.get_dummies(frame["phenotype"].astype("category"), prefix="phenotype")
    return pd.concat([frame[emb_cols].reset_index(drop=True), phen.reset_index(drop=True)], axis=1)


def ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return np.nan
    order = np.argsort(prob)
    bins = np.array_split(order, n_bins)
    total = 0.0
    for idx in bins:
        if len(idx) == 0:
            continue
        total += len(idx) / len(y_true) * abs(float(np.mean(y_true[idx])) - float(np.mean(prob[idx])))
    return float(total)


def metric_snapshot(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    prevalence = float(np.mean(y_true)) if len(y_true) else np.nan
    if len(np.unique(y_true)) < 2:
        return {
            "auroc": np.nan,
            "auprc": np.nan,
            "auprc_over_prevalence": np.nan,
            "brier": np.nan,
            "ece_10": np.nan,
        }
    auprc = float(average_precision_score(y_true, prob))
    return {
        "auroc": float(roc_auc_score(y_true, prob)),
        "auprc": auprc,
        "auprc_over_prevalence": float(auprc / prevalence) if prevalence > 0 else np.nan,
        "brier": float(brier_score_loss(y_true, prob)),
        "ece_10": ece(y_true, prob),
    }


def top_k_rows(y_true: np.ndarray, prob: np.ndarray, fractions: list[float]) -> list[dict[str, float]]:
    rows = []
    prevalence = float(np.mean(y_true))
    order = np.argsort(-prob)
    n = len(y_true)
    total_events = int(np.sum(y_true))
    for frac in fractions:
        k = max(1, int(round(n * frac)))
        idx = order[:k]
        events = int(np.sum(y_true[idx]))
        event_rate = events / k
        rows.append(
            {
                "top_fraction": frac,
                "n_selected": int(k),
                "events_selected": events,
                "event_rate": float(event_rate),
                "precision": float(event_rate),
                "recall": float(events / total_events) if total_events > 0 else np.nan,
                "enrichment_over_prevalence": float(event_rate / prevalence) if prevalence > 0 else np.nan,
                "number_needed_to_evaluate": float(1 / event_rate) if event_rate > 0 else np.inf,
            }
        )
    return rows


def decision_curve_rows(y_true: np.ndarray, prob: np.ndarray, thresholds: list[float]) -> list[dict[str, float]]:
    rows = []
    y = y_true.astype(bool)
    n = len(y)
    prevalence = float(np.mean(y_true))
    for threshold in thresholds:
        pred = prob >= threshold
        tp = int(np.sum(pred & y))
        fp = int(np.sum(pred & ~y))
        weight = threshold / (1 - threshold)
        model_nb = tp / n - fp / n * weight
        treat_all_nb = prevalence - (1 - prevalence) * weight
        rows.append(
            {
                "threshold": float(threshold),
                "net_benefit": float(model_nb),
                "treat_all_net_benefit": float(treat_all_nb),
                "treat_none_net_benefit": 0.0,
                "n_flagged": int(np.sum(pred)),
                "flagged_fraction": float(np.mean(pred)),
            }
        )
    return rows


def as_yes(series: pd.Series) -> pd.Series:
    return series.astype("string").str.upper().eq("Y").fillna(False)


def load_subgroup_frame(manifest: pd.DataFrame, test_embeddings: Path) -> pd.DataFrame:
    raw = load_exact_records(manifest, "test", SUBGROUP_RAW_COLUMNS, test_embeddings, keep_id=True)
    out = pd.DataFrame({"record_id": raw["record_id"].astype("int64")})
    age = pd.to_numeric(raw["input_MAGER"], errors="coerce")
    bmi = pd.to_numeric(raw["input_BMI"], errors="coerce")
    plurality = pd.to_numeric(raw["input_DPLURAL"], errors="coerce")
    out["age_group"] = pd.cut(age, bins=[-np.inf, 19, 34, np.inf], labels=["<20", "20-34", ">=35"]).astype("string").fillna("Unknown")
    out["bmi_group"] = pd.cut(bmi, bins=[-np.inf, 24.9, 29.9, np.inf], labels=["<25", "25-29.9", ">=30"]).astype("string").fillna("Unknown")
    out["race_ethnicity_code"] = "MRACEHISP_" + raw["input_MRACEHISP"].astype("string").fillna("Unknown")
    out["diabetes"] = np.where(as_yes(raw["input_RF_PDIAB"]) | as_yes(raw["input_RF_GDIAB"]), "Any diabetes", "No diabetes")
    out["hypertensive_disorder"] = np.where(
        as_yes(raw["input_RF_PHYPE"]) | as_yes(raw["input_RF_GHYPE"]) | as_yes(raw["input_RF_EHYPE"]),
        "Any hypertensive disorder",
        "No hypertensive disorder",
    )
    out["infertility_art"] = np.where(
        as_yes(raw["input_RF_INFTR"]) | as_yes(raw["input_RF_FEDRG"]) | as_yes(raw["input_RF_ARTEC"]),
        "Infertility/ART",
        "No infertility/ART",
    )
    out["plurality"] = np.where(plurality > 1, "Multiple", "Singleton")
    out["prior_cesarean"] = np.where(as_yes(raw["input_RF_CESAR"]), "Prior cesarean", "No prior cesarean")
    out["prior_preterm"] = np.where(as_yes(raw["input_RF_PPTERM"]), "Prior preterm", "No prior preterm")
    out["smoking"] = np.where(as_yes(raw["input_CIG_REC"]), "Smoking", "No smoking")
    out["chlamydia"] = np.where(as_yes(raw["input_IP_CHLAM"]), "Chlamydia", "No chlamydia")
    return out


def fit_ssl_predictions(args: argparse.Namespace) -> pd.DataFrame:
    dev_emb = pd.read_parquet(args.dev_embeddings)
    test_emb = pd.read_parquet(args.test_embeddings)
    dev_assign = pd.read_parquet(args.dev_assignments, columns=["record_id", "phenotype"])
    test_assign = pd.read_parquet(args.test_assignments, columns=["record_id", "phenotype"])
    dev = dev_emb.merge(dev_assign, on="record_id", how="left")
    test = test_emb.merge(test_assign, on="record_id", how="left")
    emb_cols = embedding_columns(dev)
    x_dev = make_ssl_design(dev, emb_cols)
    x_test = make_ssl_design(test, emb_cols)
    predictions = test[["record_id", "phenotype"] + PRIMARY_ENDPOINTS].copy()
    coefficient_rows = []
    for endpoint in PRIMARY_ENDPOINTS:
        y = dev[endpoint].astype("int8")
        train_idx, cal_idx = train_test_split(np.arange(len(dev)), test_size=0.4, random_state=args.seed, stratify=y)
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logistic", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
            ]
        )
        model.fit(x_dev.iloc[train_idx], y.iloc[train_idx].to_numpy())
        p_cal_raw = model.predict_proba(x_dev.iloc[cal_idx])[:, 1]
        p_test_raw = model.predict_proba(x_test)[:, 1]
        platt = fit_platt(y.iloc[cal_idx].to_numpy(), p_cal_raw)
        predictions[f"pred_ssl_plus_phenotype_{endpoint}"] = apply_platt(platt, p_test_raw)
        names = list(x_dev.columns)
        coefs = model.named_steps["logistic"].coef_[0]
        for name, coef in zip(names, coefs):
            coefficient_rows.append(
                {
                    "endpoint": endpoint,
                    "model": "ssl_plus_phenotype_logistic",
                    "feature": name,
                    "coefficient": float(coef),
                    "abs_coefficient": float(abs(coef)),
                }
            )
    pd.DataFrame(coefficient_rows).sort_values(["endpoint", "abs_coefficient"], ascending=[True, False]).to_csv(
        args.tables / tagged_name("cns_ssl_logistic_coefficients", args.output_tag, "csv"),
        index=False,
    )
    return predictions


def fit_lightgbm_predictions(args: argparse.Namespace, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_sets = feature_sets_from_schema(Path(manifest.iloc[0]["path"]))
    features = feature_sets["all_inputs"]
    columns = sorted(set(features + PRIMARY_ENDPOINTS))
    print("load LightGBM train", flush=True)
    train = load_train(manifest, columns, args.seed, args.max_train_per_year)
    print(f"train rows: {len(train):,}", flush=True)
    print("load LightGBM development/test", flush=True)
    dev = load_exact_records(manifest, "development", columns, args.dev_embeddings, keep_id=False)
    test = load_exact_records(manifest, "test", columns, args.test_embeddings, keep_id=True)
    predictions = test[["record_id"]].copy()
    importance_rows = []
    for endpoint_idx, endpoint in enumerate(PRIMARY_ENDPOINTS):
        print(f"fit LightGBM all_inputs {endpoint}", flush=True)
        y_train = train[endpoint].astype("int8").to_numpy()
        y_dev = dev[endpoint].astype("int8").to_numpy()
        x_train = prepare_lgbm_frame(train[features])
        x_dev = prepare_lgbm_frame(dev[features])
        x_test = prepare_lgbm_frame(test[features])
        model = make_lgbm(args.seed + endpoint_idx, args.lgbm_estimators)
        model.fit(x_train, y_train, categorical_feature="auto")
        p_dev_raw = model.predict_proba(x_dev)[:, 1]
        p_test_raw = model.predict_proba(x_test)[:, 1]
        platt = fit_platt(y_dev, p_dev_raw)
        predictions[f"pred_lgbm_all_inputs_{endpoint}"] = apply_platt(platt, p_test_raw)
        booster = model.booster_
        gain = booster.feature_importance(importance_type="gain")
        split = booster.feature_importance(importance_type="split")
        for feature, gain_value, split_value in zip(features, gain, split):
            importance_rows.append(
                {
                    "endpoint": endpoint,
                    "model": "lightgbm_all_inputs",
                    "feature": feature,
                    "gain_importance": float(gain_value),
                    "split_importance": float(split_value),
                    "feature_family": feature_family(feature),
                }
            )
    importance = pd.DataFrame(importance_rows)
    return predictions, importance


def feature_family(feature: str) -> str:
    name = feature.replace("missing_", "")
    if feature.startswith("missing_"):
        return "Missingness"
    if any(key in name for key in ["MAGER", "MRACE", "MHISP", "DMAR", "MEDUC", "RESTATUS", "MBSTATE"]):
        return "Demographics"
    if any(key in name for key in ["BMI", "WTGAIN", "PWgt", "M_Ht"]):
        return "Anthropometry"
    if any(key in name for key in ["PRECARE", "PREVIS", "WIC", "PAY"]):
        return "Prenatal care"
    if any(key in name for key in ["PDIAB", "GDIAB", "PHYPE", "GHYPE", "EHYPE"]):
        return "Diabetes / hypertension"
    if any(key in name for key in ["INFTR", "FEDRG", "ARTEC"]):
        return "Infertility / ART"
    if any(key in name for key in ["CIG", "IP_", "INFEC"]):
        return "Smoking / infection"
    if any(key in name for key in ["LBO", "TBO", "ILLB", "PPTERM", "CESAR", "DPLURAL", "SEX"]):
        return "Obstetric history"
    return "Other"


def build_predictions(args: argparse.Namespace, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_path = args.objects / tagged_name("cns_model_predictions", args.output_tag, "parquet")
    importance_path = args.tables / tagged_name("cns_lightgbm_gain_importance", args.output_tag, "csv")
    if args.reuse_predictions and prediction_path.exists() and importance_path.exists():
        return pd.read_parquet(prediction_path), pd.read_csv(importance_path)
    ssl_pred = fit_ssl_predictions(args)
    lgbm_pred, importance = fit_lightgbm_predictions(args, manifest)
    subgroups = load_subgroup_frame(manifest, args.test_embeddings)
    predictions = ssl_pred.merge(lgbm_pred, on="record_id", how="left").merge(subgroups, on="record_id", how="left")
    predictions.to_parquet(prediction_path, index=False)
    importance.to_csv(importance_path, index=False)
    family = (
        importance.groupby(["endpoint", "model", "feature_family"], as_index=False)["gain_importance"]
        .sum()
        .sort_values(["endpoint", "gain_importance"], ascending=[True, False])
    )
    family["gain_fraction"] = family.groupby("endpoint")["gain_importance"].transform(lambda x: x / x.sum())
    family.to_csv(args.tables / tagged_name("cns_feature_family_importance", args.output_tag, "csv"), index=False)
    return predictions, importance


def model_specs() -> list[tuple[str, str]]:
    return [
        ("SSL + phenotype", "pred_ssl_plus_phenotype"),
        ("LightGBM all inputs", "pred_lgbm_all_inputs"),
    ]


def run_subgroup_metrics(predictions: pd.DataFrame, tables: Path, tag: str) -> pd.DataFrame:
    subgroup_vars = [
        "age_group",
        "bmi_group",
        "race_ethnicity_code",
        "diabetes",
        "hypertensive_disorder",
        "infertility_art",
        "plurality",
        "prior_cesarean",
        "prior_preterm",
        "smoking",
    ]
    rows = []
    for endpoint in PRIMARY_ENDPOINTS:
        y_all = predictions[endpoint].astype("int8").to_numpy()
        for model_label, prefix in model_specs():
            prob_all = predictions[f"{prefix}_{endpoint}"].to_numpy(dtype=float)
            for subgroup_var in subgroup_vars:
                for subgroup, group in predictions.groupby(subgroup_var, dropna=False):
                    y = group[endpoint].astype("int8").to_numpy()
                    prob = group[f"{prefix}_{endpoint}"].to_numpy(dtype=float)
                    snapshot = metric_snapshot(y, prob)
                    top1 = top_k_rows(y, prob, [0.01])[0] if len(group) >= 100 else {}
                    rows.append(
                        {
                            "endpoint": endpoint,
                            "endpoint_label": ENDPOINT_LABEL[endpoint],
                            "model": model_label,
                            "subgroup_variable": subgroup_var,
                            "subgroup": str(subgroup),
                            "n": int(len(group)),
                            "events": int(np.sum(y)),
                            "prevalence": float(np.mean(y)) if len(y) else np.nan,
                            **snapshot,
                            "top1_event_rate": top1.get("event_rate", np.nan),
                            "top1_enrichment_over_prevalence": top1.get("enrichment_over_prevalence", np.nan),
                            "overall_prevalence": float(np.mean(y_all)),
                            "overall_model_auprc": metric_snapshot(y_all, prob_all)["auprc"],
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(tables / tagged_name("cns_subgroup_metrics", tag, "csv"), index=False)
    return out


def run_topk_utility(predictions: pd.DataFrame, tables: Path, tag: str) -> pd.DataFrame:
    rows = []
    for endpoint in PRIMARY_ENDPOINTS:
        y = predictions[endpoint].astype("int8").to_numpy()
        for model_label, prefix in model_specs():
            prob = predictions[f"{prefix}_{endpoint}"].to_numpy(dtype=float)
            for row in top_k_rows(y, prob, [0.005, 0.01, 0.02, 0.05]):
                row.update({"endpoint": endpoint, "endpoint_label": ENDPOINT_LABEL[endpoint], "model": model_label})
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(tables / tagged_name("cns_topk_utility", tag, "csv"), index=False)
    return out


def run_decision_curves(predictions: pd.DataFrame, tables: Path, tag: str) -> pd.DataFrame:
    threshold_map = {
        "outcome_maternal_morbidity_core": [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05],
        "outcome_severe_neonatal_no_nicu": [0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20],
    }
    rows = []
    for endpoint in PRIMARY_ENDPOINTS:
        y = predictions[endpoint].astype("int8").to_numpy()
        for model_label, prefix in model_specs():
            prob = predictions[f"{prefix}_{endpoint}"].to_numpy(dtype=float)
            for row in decision_curve_rows(y, prob, threshold_map[endpoint]):
                row.update({"endpoint": endpoint, "endpoint_label": ENDPOINT_LABEL[endpoint], "model": model_label})
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(tables / tagged_name("cns_decision_curve", tag, "csv"), index=False)
    return out


def phenotype_standardized_profiles(predictions: pd.DataFrame, manifest: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    raw = load_exact_records(manifest, "test", SUBGROUP_RAW_COLUMNS, args.test_embeddings, keep_id=True)
    data = predictions[["record_id", "phenotype"]].merge(raw, on="record_id", how="left")
    feature_defs = {
        "Age": (pd.to_numeric(data["input_MAGER"], errors="coerce"), "numeric"),
        "BMI": (pd.to_numeric(data["input_BMI"], errors="coerce"), "numeric"),
        "Prenatal visits": (pd.to_numeric(data["input_PREVIS"], errors="coerce"), "numeric"),
        "Weight gain": (pd.to_numeric(data["input_WTGAIN"], errors="coerce"), "numeric"),
        "GDM": (as_yes(data["input_RF_GDIAB"]).astype(float), "binary"),
        "Hypertension": ((as_yes(data["input_RF_PHYPE"]) | as_yes(data["input_RF_GHYPE"]) | as_yes(data["input_RF_EHYPE"])).astype(float), "binary"),
        "Prior preterm": (as_yes(data["input_RF_PPTERM"]).astype(float), "binary"),
        "Prior cesarean": (as_yes(data["input_RF_CESAR"]).astype(float), "binary"),
        "Smoking": (as_yes(data["input_CIG_REC"]).astype(float), "binary"),
        "Infertility/ART": ((as_yes(data["input_RF_INFTR"]) | as_yes(data["input_RF_FEDRG"]) | as_yes(data["input_RF_ARTEC"])).astype(float), "binary"),
        "Multiple gestation": ((pd.to_numeric(data["input_DPLURAL"], errors="coerce") > 1).astype(float), "binary"),
        "Chlamydia": (as_yes(data["input_IP_CHLAM"]).astype(float), "binary"),
    }
    rows = []
    for label, (values, kind) in feature_defs.items():
        overall = float(np.nanmean(values))
        if kind == "numeric":
            scale = float(np.nanstd(values, ddof=1))
        else:
            scale = float(np.sqrt(max(overall * (1 - overall), 1e-9)))
        for phenotype, idx in data.groupby("phenotype").groups.items():
            phenotype_values = values.loc[idx]
            mean = float(np.nanmean(phenotype_values))
            rows.append(
                {
                    "phenotype": int(phenotype),
                    "feature": label,
                    "feature_type": kind,
                    "value": mean,
                    "overall_value": overall,
                    "standardized_difference": float((mean - overall) / scale) if scale > 0 else np.nan,
                    "n_nonmissing": int(phenotype_values.notna().sum()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(args.tables / tagged_name("cns_phenotype_standardized_profiles", args.output_tag, "csv"), index=False)
    return out


def write_report(
    args: argparse.Namespace,
    predictions: pd.DataFrame,
    subgroup: pd.DataFrame,
    utility: pd.DataFrame,
    decision: pd.DataFrame,
    importance: pd.DataFrame,
    profile: pd.DataFrame,
) -> None:
    lines = [
        "# CNS Reviewer Extension Analysis Report",
        "",
        f"This report adds reviewer-requested robustness, clinical utility, and interpretability outputs on the matched 2024 test sample for output tag `{args.output_tag}`.",
        "",
        "## Outputs",
        "",
        f"- `results/objects/{tagged_name('cns_model_predictions', args.output_tag, 'parquet')}`",
        f"- `results/tables/{tagged_name('cns_subgroup_metrics', args.output_tag, 'csv')}`",
        f"- `results/tables/{tagged_name('cns_topk_utility', args.output_tag, 'csv')}`",
        f"- `results/tables/{tagged_name('cns_decision_curve', args.output_tag, 'csv')}`",
        f"- `results/tables/{tagged_name('cns_lightgbm_gain_importance', args.output_tag, 'csv')}`",
        f"- `results/tables/{tagged_name('cns_feature_family_importance', args.output_tag, 'csv')}`",
        f"- `results/tables/{tagged_name('cns_phenotype_standardized_profiles', args.output_tag, 'csv')}`",
        f"- `results/tables/{tagged_name('cns_ssl_logistic_coefficients', args.output_tag, 'csv')}`",
        "",
        "## Prediction Table",
        "",
        f"- Rows: {len(predictions):,}",
        f"- Maternal events: {int(predictions['outcome_maternal_morbidity_core'].sum()):,}",
        f"- Neonatal events: {int(predictions['outcome_severe_neonatal_no_nicu'].sum()):,}",
        "",
        "## Top-k Utility Snapshot",
        "",
        "| Endpoint | Model | Top fraction | Event rate | Recall | Enrichment | NNE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    snap = utility[utility["top_fraction"].isin([0.01, 0.05])].copy()
    for row in snap.to_dict("records"):
        lines.append(
            "| {endpoint_label} | {model} | {top_fraction:.1%} | {event_rate:.3f} | {recall:.3f} | {enrichment_over_prevalence:.2f} | {number_needed_to_evaluate:.1f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- Subgroup metrics are descriptive robustness analyses on the matched 2024 sample, not prospective fairness validation.",
            "- Decision curves use retrospective calibrated probabilities and should not be interpreted as deployment-ready clinical utility.",
            "- LightGBM feature importance is gain-based, not SHAP; SHAP was not available in the current Python environment.",
            "",
            "## Dimensions",
            "",
            f"- Subgroup rows: {len(subgroup):,}",
            f"- Decision-curve rows: {len(decision):,}",
            f"- LightGBM importance rows: {len(importance):,}",
            f"- Phenotype profile rows: {len(profile):,}",
        ]
    )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    args.objects.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.split_manifest)
    predictions, importance = build_predictions(args, manifest)
    subgroup = run_subgroup_metrics(predictions, args.tables, args.output_tag)
    utility = run_topk_utility(predictions, args.tables, args.output_tag)
    decision = run_decision_curves(predictions, args.tables, args.output_tag)
    profile = phenotype_standardized_profiles(predictions, manifest, args)
    metadata = {
        "prediction_rows": int(len(predictions)),
        "subgroup_metric_rows": int(len(subgroup)),
        "topk_utility_rows": int(len(utility)),
        "decision_curve_rows": int(len(decision)),
        "importance_rows": int(len(importance)),
        "phenotype_profile_rows": int(len(profile)),
        "max_train_per_year": int(args.max_train_per_year),
        "lgbm_estimators": int(args.lgbm_estimators),
        "seed": int(args.seed),
        "output_tag": args.output_tag,
    }
    (args.tables / tagged_name("cns_extension_metadata", args.output_tag, "json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(args, predictions, subgroup, utility, decision, importance, profile)
    print(f"wrote {args.objects / tagged_name('cns_model_predictions', args.output_tag, 'parquet')}")
    print(f"wrote {args.tables / tagged_name('cns_subgroup_metrics', args.output_tag, 'csv')}")
    print(f"wrote {args.tables / tagged_name('cns_topk_utility', args.output_tag, 'csv')}")
    print(f"wrote {args.tables / tagged_name('cns_decision_curve', args.output_tag, 'csv')}")
    print(f"wrote {args.tables / tagged_name('cns_lightgbm_gain_importance', args.output_tag, 'csv')}")
    print(f"wrote {args.tables / tagged_name('cns_phenotype_standardized_profiles', args.output_tag, 'csv')}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
