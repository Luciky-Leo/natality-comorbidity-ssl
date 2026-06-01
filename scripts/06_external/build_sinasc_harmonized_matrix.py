#!/usr/bin/env python
"""Build harmonized SINASC matrices for independent registry stress-testing."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINASC_DIR = PROJECT_ROOT / "data" / "external" / "opendatasus_sinasc"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "sinasc"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

INPUT_COLUMNS = [
    "IDADEMAE",
    "RACACORMAE",
    "RACACOR",
    "ESTCIVMAE",
    "ESCMAE",
    "ESCMAE2010",
    "ESCMAEAGR1",
    "CODMUNRES",
    "CODPAISRES",
    "QTDFILVIVO",
    "QTDGESTANT",
    "PARIDADE",
    "QTDPARTNOR",
    "QTDPARTCES",
    "MESPRENAT",
    "CONSULTAS",
    "CONSPRENAT",
    "KOTELCHUCK",
    "GRAVIDEZ",
    "SEXO",
]

OUTCOME_SOURCE_COLUMNS = [
    "SEMAGESTAC",
    "GESTACAO",
    "PESO",
    "APGAR5",
    "IDANOMAL",
    "CODANOMAL",
]

USE_COLUMNS = sorted(set(INPUT_COLUMNS + OUTCOME_SOURCE_COLUMNS))

ZIP_BY_YEAR = {
    2023: SINASC_DIR / "SINASC_2023_csv.zip",
    2024: SINASC_DIR / "SINASC_2024_csv.zip",
}


def parse_num(series: pd.Series, missing_values: set[str] | None = None) -> pd.Series:
    missing_values = missing_values or set()
    text = series.astype("string").str.strip()
    text = text.mask(text.isin({"", "NA", "nan", "None", *missing_values}))
    return pd.to_numeric(text, errors="coerce")


def clean_cat(series: pd.Series, missing_values: set[str] | None = None) -> pd.Series:
    missing_values = missing_values or set()
    text = series.astype("string").str.strip()
    text = text.mask(text.isin({"", "NA", "nan", "None", *missing_values}))
    return text.astype("string")


def yes_no_from_numeric_gt0(series: pd.Series) -> pd.Series:
    values = parse_num(series)
    return values.gt(0).where(values.notna(), pd.NA).astype("Int8")


def derive_preterm_from_group(group: pd.Series, threshold: str) -> pd.Series:
    # GESTACAO: 1 <22, 2 22-27, 3 28-31, 4 32-36, 5 37-41, 6 >=42, 9 ignored.
    g = clean_cat(group, {"9"})
    if threshold == "lt37":
        out = g.isin(["1", "2", "3", "4"]).where(g.notna(), pd.NA)
    elif threshold == "lt32":
        out = g.isin(["1", "2", "3"]).where(g.notna(), pd.NA)
    else:
        raise ValueError(threshold)
    return out.astype("boolean")


def derive_frame(chunk: pd.DataFrame, year: int, record_offset: int) -> pd.DataFrame:
    out = pd.DataFrame(index=chunk.index)
    out["source_registry"] = "SINASC"
    out["source_year"] = np.int16(year)
    out["record_id"] = np.arange(record_offset, record_offset + len(chunk), dtype=np.int64)

    out["input_IDADEMAE"] = parse_num(chunk["IDADEMAE"], {"99"}).astype("float32")
    out["input_RACACORMAE"] = clean_cat(chunk["RACACORMAE"], {"9"})
    out["input_RACACOR"] = clean_cat(chunk["RACACOR"], {"9"})
    out["input_ESTCIVMAE"] = clean_cat(chunk["ESTCIVMAE"], {"9"})
    out["input_ESCMAE"] = clean_cat(chunk["ESCMAE"], {"9"})
    out["input_ESCMAE2010"] = clean_cat(chunk["ESCMAE2010"], {"9"})
    out["input_ESCMAEAGR1"] = clean_cat(chunk["ESCMAEAGR1"], {"99"})
    codmun = clean_cat(chunk["CODMUNRES"])
    out["input_UFRES"] = codmun.str.slice(0, 2).astype("string")
    out["input_CODPAISRES"] = clean_cat(chunk["CODPAISRES"])
    for col in ["QTDFILVIVO", "QTDGESTANT", "QTDPARTNOR", "QTDPARTCES", "MESPRENAT", "CONSPRENAT"]:
        out[f"input_{col}"] = parse_num(chunk[col], {"99"}).astype("float32")
    out["input_PARIDADE"] = clean_cat(chunk["PARIDADE"])
    out["input_CONSULTAS"] = clean_cat(chunk["CONSULTAS"], {"9"})
    out["input_KOTELCHUCK"] = clean_cat(chunk["KOTELCHUCK"], {"9"})
    out["input_GRAVIDEZ"] = clean_cat(chunk["GRAVIDEZ"], {"9"})
    out["input_SEXO"] = clean_cat(chunk["SEXO"], {"0"})
    out["input_prior_cesarean"] = yes_no_from_numeric_gt0(chunk["QTDPARTCES"])
    gravidez = clean_cat(chunk["GRAVIDEZ"], {"9"})
    out["input_multiple_gestation"] = gravidez.isin(["2", "3"]).where(gravidez.notna(), pd.NA).astype("Int8")
    partnor = parse_num(chunk["QTDPARTNOR"], {"99"})
    partces = parse_num(chunk["QTDPARTCES"], {"99"})
    out["input_nulliparous"] = ((partnor.fillna(0) + partces.fillna(0)) == 0).where(partnor.notna() | partces.notna(), pd.NA).astype("Int8")

    input_cols = [col for col in out.columns if col.startswith("input_")]
    for col in input_cols:
        out[f"missing_{col}"] = out[col].isna().astype("bool")

    gest_weeks = parse_num(chunk["SEMAGESTAC"], {"99"}).astype("float32")
    gest_weeks = gest_weeks.mask((gest_weeks < 15) | (gest_weeks > 45))
    gest_group = clean_cat(chunk["GESTACAO"], {"9"})
    birthweight = parse_num(chunk["PESO"], {"9999"}).astype("float32")
    birthweight = birthweight.mask((birthweight < 200) | (birthweight > 7000))
    apgar5 = parse_num(chunk["APGAR5"], {"99"}).astype("float32")
    apgar5 = apgar5.mask((apgar5 < 0) | (apgar5 > 10))
    anomaly = clean_cat(chunk["IDANOMAL"], {"9"})

    preterm_direct = gest_weeks.lt(37).where(gest_weeks.notna(), pd.NA).astype("boolean")
    preterm_group = derive_preterm_from_group(gest_group, "lt37")
    very_preterm_direct = gest_weeks.lt(32).where(gest_weeks.notna(), pd.NA).astype("boolean")
    very_preterm_group = derive_preterm_from_group(gest_group, "lt32")

    out["outcome_gest_weeks"] = gest_weeks
    out["outcome_birthweight_g"] = birthweight
    out["outcome_apgar5"] = apgar5
    out["outcome_preterm_lt37"] = preterm_direct.fillna(preterm_group).astype("Int8")
    out["outcome_very_preterm_lt32"] = very_preterm_direct.fillna(very_preterm_group).astype("Int8")
    out["outcome_low_birthweight_lt2500g"] = birthweight.lt(2500).where(birthweight.notna(), pd.NA).astype("Int8")
    out["outcome_very_low_birthweight_lt1500g"] = birthweight.lt(1500).where(birthweight.notna(), pd.NA).astype("Int8")
    out["outcome_low_apgar5_lt7"] = apgar5.lt(7).where(apgar5.notna(), pd.NA).astype("Int8")
    out["outcome_congenital_anomaly"] = anomaly.eq("1").where(anomaly.notna(), pd.NA).astype("Int8")

    severe_components = [
        "outcome_very_preterm_lt32",
        "outcome_very_low_birthweight_lt1500g",
        "outcome_low_apgar5_lt7",
        "outcome_congenital_anomaly",
    ]
    broad_components = [
        "outcome_preterm_lt37",
        "outcome_low_birthweight_lt2500g",
        "outcome_low_apgar5_lt7",
        "outcome_congenital_anomaly",
    ]
    out["outcome_sinasc_severe_birth_status"] = component_any(out, severe_components)
    out["outcome_sinasc_broad_birth_status"] = component_any(out, broad_components)
    out["context_PARTO"] = clean_cat(chunk.get("PARTO", pd.Series(index=chunk.index, dtype="string")), {"9"})
    out["context_TPROBSON"] = clean_cat(chunk.get("TPROBSON", pd.Series(index=chunk.index, dtype="string")), {"99"})
    out["context_CODANOMAL"] = clean_cat(chunk["CODANOMAL"])
    return out


def component_any(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns]
    any_positive = values.eq(1).any(axis=1)
    all_missing = values.isna().all(axis=1)
    return any_positive.mask(all_missing, pd.NA).astype("Int8")


def schema_for(frame: pd.DataFrame) -> pa.Schema:
    fields = []
    for col in frame.columns:
        if col == "record_id":
            fields.append(pa.field(col, pa.int64()))
        elif col == "source_year":
            fields.append(pa.field(col, pa.int16()))
        elif col.startswith("missing_"):
            fields.append(pa.field(col, pa.bool_()))
        elif col.startswith("outcome_") and col not in {"outcome_gest_weeks", "outcome_birthweight_g", "outcome_apgar5"}:
            fields.append(pa.field(col, pa.int8()))
        elif col.startswith("input_") and pd.api.types.is_numeric_dtype(frame[col]):
            fields.append(pa.field(col, pa.float32()))
        elif col in {"outcome_gest_weeks", "outcome_birthweight_g", "outcome_apgar5"}:
            fields.append(pa.field(col, pa.float32()))
        else:
            fields.append(pa.field(col, pa.string()))
    return pa.schema(fields)


def build_year(year: int, zip_path: Path, chunk_size: int = 250_000) -> dict[str, object]:
    output_path = OUTPUT_DIR / f"sinasc_harmonized_{year}.parquet"
    if output_path.exists():
        output_path.unlink()
    writer: pq.ParquetWriter | None = None
    record_offset = 0
    missing_counts: dict[str, int] = {}
    outcome_counts: dict[str, dict[str, int]] = {}
    n_rows = 0
    with zipfile.ZipFile(zip_path) as archive:
        entry = archive.namelist()[0]
    reader = pd.read_csv(
        zip_path,
        sep=";",
        usecols=lambda col: col in set(USE_COLUMNS + ["PARTO", "TPROBSON"]),
        dtype="string",
        chunksize=chunk_size,
        encoding="utf-8",
        low_memory=False,
    )
    for chunk in reader:
        derived = derive_frame(chunk, year, record_offset)
        if writer is None:
            schema = schema_for(derived)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(output_path, schema=schema, compression="zstd")
        table = pa.Table.from_pandas(derived, schema=writer.schema, preserve_index=False)
        writer.write_table(table)
        n_rows += len(derived)
        record_offset += len(derived)
        for col in derived.columns:
            if col.startswith("input_") or col.startswith("outcome_"):
                missing_counts[col] = missing_counts.get(col, 0) + int(derived[col].isna().sum())
            if col.startswith("outcome_") and col not in {"outcome_gest_weeks", "outcome_birthweight_g", "outcome_apgar5"}:
                known = derived[col].dropna()
                counts = outcome_counts.setdefault(col, {"known": 0, "positive": 0})
                counts["known"] += int(len(known))
                counts["positive"] += int((known.astype("int8") == 1).sum())
        print(f"{year}: {n_rows:,}", flush=True)
    if writer is not None:
        writer.close()
    return {
        "year": year,
        "zip_path": str(zip_path),
        "zip_entry": entry,
        "output_path": str(output_path),
        "n_rows": n_rows,
        "missing_counts": missing_counts,
        "outcome_counts": outcome_counts,
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = [build_year(year, zip_path) for year, zip_path in ZIP_BY_YEAR.items()]

    manifest_rows = []
    missing_rows = []
    outcome_rows = []
    for item in metadata:
        year = int(item["year"])
        split = "development" if year == 2023 else "test"
        manifest_rows.append({"year": year, "split": split, "path": item["output_path"], "n_rows": item["n_rows"]})
        for col, n_missing in item["missing_counts"].items():
            missing_rows.append(
                {
                    "year": year,
                    "column": col,
                    "n_rows": item["n_rows"],
                    "n_missing": n_missing,
                    "missing_pct": n_missing / item["n_rows"] * 100,
                }
            )
        for col, counts in item["outcome_counts"].items():
            known = counts["known"]
            positive = counts["positive"]
            outcome_rows.append(
                {
                    "year": year,
                    "outcome": col,
                    "known_n": known,
                    "positive_n": positive,
                    "positive_pct_known": positive / known * 100 if known else np.nan,
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = OUTPUT_DIR / "sinasc_2023_2024_split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    missing_path = TABLE_DIR / "sinasc_harmonized_missingness.csv"
    outcome_path = TABLE_DIR / "sinasc_harmonized_outcome_prevalence.csv"
    metadata_path = TABLE_DIR / "sinasc_harmonized_metadata.json"
    pd.DataFrame(missing_rows).to_csv(missing_path, index=False)
    pd.DataFrame(outcome_rows).to_csv(outcome_path, index=False)
    metadata_path.write_text(json.dumps({"years": metadata, "manifest": str(manifest_path)}, indent=2), encoding="utf-8")

    print(manifest_path)
    print(missing_path)
    print(outcome_path)
    print(metadata_path)


if __name__ == "__main__":
    main()

