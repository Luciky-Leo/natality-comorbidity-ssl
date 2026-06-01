#!/usr/bin/env python
"""Full-2024 SSL+phenotype utility, subgroup, and phenotype profile analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = PROJECT_ROOT / "results" / "objects" / "ssl_plus_phenotype_predictions_full2016_2022_mask035_d48_l2_cuda_full2024.parquet"
DEFAULT_COHORT = PROJECT_ROOT / "data" / "processed" / "nat2024_analytic_cohort.parquet"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "40_full2024_ssl_utility_profiles_report.md"

TAG = "full2016_2022_mask035_d48_l2_cuda_full2024"

ENDPOINTS = [
    "outcome_maternal_morbidity_core",
    "outcome_severe_neonatal_no_nicu",
]

ENDPOINT_LABEL = {
    "outcome_maternal_morbidity_core": "Maternal core morbidity",
    "outcome_severe_neonatal_no_nicu": "Severe neonatal outcome",
}

RAW_COLUMNS = [
    "record_id",
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
    "outcome_gest_weeks_combined",
    "outcome_birthweight_g",
    "outcome_preterm_lt37",
    "outcome_low_birthweight_lt2500g",
    "outcome_very_low_birthweight_lt1500g",
    "outcome_low_apgar5_lt7",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-tag", default=TAG)
    return parser.parse_args()


def tagged(stem: str, tag: str) -> str:
    return f"{stem}_{tag}.csv"


def as_yes(series: pd.Series) -> pd.Series:
    return series.astype("string").str.upper().eq("Y").fillna(False)


def ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
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
        return {"auroc": np.nan, "auprc": np.nan, "auprc_over_prevalence": np.nan, "brier": np.nan, "ece_10": np.nan}
    auprc = float(average_precision_score(y_true, prob))
    return {
        "auroc": float(roc_auc_score(y_true, prob)),
        "auprc": auprc,
        "auprc_over_prevalence": auprc / prevalence if prevalence else np.nan,
        "brier": float(brier_score_loss(y_true, prob)),
        "ece_10": ece(y_true, prob),
    }


def top_k_rows(y_true: np.ndarray, prob: np.ndarray, fractions: list[float]) -> list[dict[str, float]]:
    rows = []
    prevalence = float(np.mean(y_true))
    total_events = int(np.sum(y_true))
    order = np.argsort(-prob)
    n = len(y_true)
    for fraction in fractions:
        k = max(1, int(round(n * fraction)))
        idx = order[:k]
        events = int(np.sum(y_true[idx]))
        event_rate = events / k
        rows.append(
            {
                "top_fraction": fraction,
                "n_selected": int(k),
                "events_selected": events,
                "event_rate": float(event_rate),
                "precision": float(event_rate),
                "recall": float(events / total_events) if total_events else np.nan,
                "enrichment_over_prevalence": float(event_rate / prevalence) if prevalence else np.nan,
                "number_needed_to_evaluate": float(1 / event_rate) if event_rate else np.inf,
                "baseline_prevalence": prevalence,
                "total_events": total_events,
            }
        )
    return rows


def decision_curve_rows(y_true: np.ndarray, prob: np.ndarray, thresholds: list[float]) -> list[dict[str, float]]:
    rows = []
    y = y_true.astype(bool)
    n = len(y)
    prevalence = float(np.mean(y_true))
    for threshold in thresholds:
        selected = prob >= threshold
        tp = int(np.sum(selected & y))
        fp = int(np.sum(selected & ~y))
        weight = threshold / (1 - threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "net_benefit": float(tp / n - fp / n * weight),
                "treat_all_net_benefit": float(prevalence - (1 - prevalence) * weight),
                "treat_none_net_benefit": 0.0,
                "n_flagged": int(np.sum(selected)),
                "flagged_fraction": float(np.mean(selected)),
            }
        )
    return rows


def add_subgroups(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    age = pd.to_numeric(out["input_MAGER"], errors="coerce")
    bmi = pd.to_numeric(out["input_BMI"], errors="coerce")
    plurality = pd.to_numeric(out["input_DPLURAL"], errors="coerce")
    out["age_group"] = pd.cut(age, bins=[-np.inf, 19, 34, np.inf], labels=["<20", "20-34", ">=35"]).astype("string").fillna("Unknown")
    out["bmi_group"] = pd.cut(bmi, bins=[-np.inf, 24.9, 29.9, np.inf], labels=["<25", "25-29.9", ">=30"]).astype("string").fillna("Unknown")
    out["race_ethnicity_code"] = "MRACEHISP_" + out["input_MRACEHISP"].astype("string").fillna("Unknown")
    out["diabetes"] = np.where(as_yes(out["input_RF_PDIAB"]) | as_yes(out["input_RF_GDIAB"]), "Any diabetes", "No diabetes")
    out["hypertensive_disorder"] = np.where(
        as_yes(out["input_RF_PHYPE"]) | as_yes(out["input_RF_GHYPE"]) | as_yes(out["input_RF_EHYPE"]),
        "Any hypertensive disorder",
        "No hypertensive disorder",
    )
    out["infertility_art"] = np.where(
        as_yes(out["input_RF_INFTR"]) | as_yes(out["input_RF_FEDRG"]) | as_yes(out["input_RF_ARTEC"]),
        "Infertility/ART",
        "No infertility/ART",
    )
    out["plurality"] = np.where(plurality > 1, "Multiple", "Singleton")
    out["prior_cesarean"] = np.where(as_yes(out["input_RF_CESAR"]), "Prior cesarean", "No prior cesarean")
    out["prior_preterm"] = np.where(as_yes(out["input_RF_PPTERM"]), "Prior preterm", "No prior preterm")
    out["smoking"] = np.where(as_yes(out["input_CIG_REC"]), "Smoking", "No smoking")
    out["chlamydia"] = np.where(as_yes(out["input_IP_CHLAM"]), "Chlamydia", "No chlamydia")
    return out


def run_topk(predictions: pd.DataFrame, tables: Path, tag: str) -> pd.DataFrame:
    rows = []
    for endpoint in ENDPOINTS:
        y = predictions[endpoint].astype("int8").to_numpy()
        prob = predictions[f"pred_ssl_plus_phenotype_{endpoint}"].to_numpy(dtype=float)
        for row in top_k_rows(y, prob, [0.005, 0.01, 0.02, 0.05]):
            row.update({"endpoint": endpoint, "endpoint_label": ENDPOINT_LABEL[endpoint], "model": "SSL + phenotype"})
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(tables / tagged("cns_topk_utility", tag), index=False)
    return out


def run_decision(predictions: pd.DataFrame, tables: Path, tag: str) -> pd.DataFrame:
    threshold_map = {
        "outcome_maternal_morbidity_core": [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05],
        "outcome_severe_neonatal_no_nicu": [0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20],
    }
    rows = []
    for endpoint in ENDPOINTS:
        y = predictions[endpoint].astype("int8").to_numpy()
        prob = predictions[f"pred_ssl_plus_phenotype_{endpoint}"].to_numpy(dtype=float)
        for row in decision_curve_rows(y, prob, threshold_map[endpoint]):
            row.update({"endpoint": endpoint, "endpoint_label": ENDPOINT_LABEL[endpoint], "model": "SSL + phenotype"})
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(tables / tagged("cns_decision_curve", tag), index=False)
    return out


def run_subgroups(data: pd.DataFrame, tables: Path, tag: str) -> pd.DataFrame:
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
    for endpoint in ENDPOINTS:
        y_all = data[endpoint].astype("int8").to_numpy()
        prob_all = data[f"pred_ssl_plus_phenotype_{endpoint}"].to_numpy(dtype=float)
        overall_auprc = metric_snapshot(y_all, prob_all)["auprc"]
        for subgroup_var in subgroup_vars:
            for subgroup, group in data.groupby(subgroup_var, dropna=False):
                y = group[endpoint].astype("int8").to_numpy()
                prob = group[f"pred_ssl_plus_phenotype_{endpoint}"].to_numpy(dtype=float)
                top1 = top_k_rows(y, prob, [0.01])[0] if len(group) >= 100 else {}
                rows.append(
                    {
                        "endpoint": endpoint,
                        "endpoint_label": ENDPOINT_LABEL[endpoint],
                        "model": "SSL + phenotype",
                        "subgroup_variable": subgroup_var,
                        "subgroup": str(subgroup),
                        "n": int(len(group)),
                        "events": int(np.sum(y)),
                        "prevalence": float(np.mean(y)) if len(y) else np.nan,
                        **metric_snapshot(y, prob),
                        "top1_event_rate": top1.get("event_rate", np.nan),
                        "top1_enrichment_over_prevalence": top1.get("enrichment_over_prevalence", np.nan),
                        "overall_prevalence": float(np.mean(y_all)),
                        "overall_model_auprc": overall_auprc,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(tables / tagged("cns_subgroup_metrics", tag), index=False)
    return out


def phenotype_profiles(data: pd.DataFrame, tables: Path, tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        scale = float(np.nanstd(values, ddof=1)) if kind == "numeric" else float(np.sqrt(max(overall * (1 - overall), 1e-9)))
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
    profile = pd.DataFrame(rows)
    profile.to_csv(tables / tagged("cns_phenotype_standardized_profiles", tag), index=False)

    birth_rows = []
    for phenotype, group in data.groupby("phenotype", sort=True):
        birth_rows.append(
            {
                "phenotype": int(phenotype),
                "n": int(len(group)),
                "proportion": float(len(group) / len(data)),
                "mean_gestational_age_weeks": float(pd.to_numeric(group["outcome_gest_weeks_combined"], errors="coerce").mean()),
                "mean_birthweight_g": float(pd.to_numeric(group["outcome_birthweight_g"], errors="coerce").mean()),
                "preterm_lt37_rate": float(pd.to_numeric(group["outcome_preterm_lt37"], errors="coerce").mean()),
                "low_birthweight_lt2500g_rate": float(pd.to_numeric(group["outcome_low_birthweight_lt2500g"], errors="coerce").mean()),
                "very_low_birthweight_lt1500g_rate": float(pd.to_numeric(group["outcome_very_low_birthweight_lt1500g"], errors="coerce").mean()),
                "low_apgar5_lt7_rate": float(pd.to_numeric(group["outcome_low_apgar5_lt7"], errors="coerce").mean()),
                "maternal_core_morbidity_rate": float(group["outcome_maternal_morbidity_core"].mean()),
                "severe_neonatal_no_nicu_rate": float(group["outcome_severe_neonatal_no_nicu"].mean()),
            }
        )
    birth = pd.DataFrame(birth_rows)
    birth.to_csv(tables / tagged("phenotype_birth_status_profile", tag), index=False)
    return profile, birth


def main() -> None:
    args = parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    print("load predictions", flush=True)
    predictions = pd.read_parquet(args.predictions)
    print("load full 2024 raw profile columns", flush=True)
    raw = pd.read_parquet(args.cohort, columns=RAW_COLUMNS)
    raw["record_id"] = raw["record_id"].astype("int64")
    data = predictions.merge(raw, on="record_id", how="left")
    data = add_subgroups(data)

    topk = run_topk(predictions, args.tables, args.output_tag)
    decision = run_decision(predictions, args.tables, args.output_tag)
    subgroup = run_subgroups(data, args.tables, args.output_tag)
    profile, birth = phenotype_profiles(data, args.tables, args.output_tag)
    metadata = {
        "predictions": str(args.predictions),
        "cohort": str(args.cohort),
        "rows": int(len(data)),
        "topk_rows": int(len(topk)),
        "decision_curve_rows": int(len(decision)),
        "subgroup_rows": int(len(subgroup)),
        "profile_rows": int(len(profile)),
        "birth_profile_rows": int(len(birth)),
        "model": "SSL + phenotype",
    }
    metadata_path = args.tables / f"full2024_ssl_utility_profiles_{args.output_tag}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lines = [
        "# Full 2024 SSL Utility and Phenotype Profile Report",
        "",
        "This report recomputes clinical utility, subgroup robustness, and phenotype profile summaries on the full 3,638,436-record 2024 temporal test year for the SSL + phenotype model.",
        "",
        f"- rows: {len(data):,}",
        f"- top-k table: `{args.tables / tagged('cns_topk_utility', args.output_tag)}`",
        f"- decision-curve table: `{args.tables / tagged('cns_decision_curve', args.output_tag)}`",
        f"- subgroup table: `{args.tables / tagged('cns_subgroup_metrics', args.output_tag)}`",
        f"- phenotype profile table: `{args.tables / tagged('cns_phenotype_standardized_profiles', args.output_tag)}`",
        f"- birth-status profile table: `{args.tables / tagged('phenotype_birth_status_profile', args.output_tag)}`",
        f"- metadata: `{metadata_path}`",
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
