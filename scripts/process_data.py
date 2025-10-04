#!/usr/bin/env python3
"""
Process CSV datasets in DATASETS/CSV_DATA and export compact JSON stats for the dashboard.

Outputs one JSON per category under export/:
  - heat.json (MODIS LST)
  - green.json (NDVI)
  - housing.json (NDBI)
  - light.json (VIIRS NTL)
  - air.json (NO2)
  - population.json (WorldPop 2020 baseline)

Schema per file:
{
    "year": {
        "area": {
            "mean": float,
            "median": float,
            "min": float,
            "max": float,
            "p10": float,
            "p90": float,
            "std": float,
            "units": str
        }
    }
}

Run:
  python scripts/process_data.py
"""

import os
import re
import json
import math
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Read CSVs from DATASETS/CSV_DATA per latest files
CSV_DIR = ROOT / "DATASETS" / "CSV_DATA"
EXPORT_DIR = ROOT / "export"

# Default units per category to improve dashboard readability
CATEGORY_UNITS = {
    "heat": "°C",
    "green": "% of area covered by vegetation (NDVI normalized 0–1)",
    "housing": "km² built-up area (or % of land built-up)",
    "light": "nW·cm⁻²·sr⁻¹ (brightness index 0–100)",
    "air": "µmol/m²",
    "population": "People/km² (population density, 2020 baseline)"
}


def ensure_export_dir() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def infer_category(filename: str) -> str:
    name = filename.lower()
    # new file names use simple prefixes: heat_*.csv, green_space_*.csv, housing_*.csv, hosuing_*.csv, light_*.csv, population_2020.csv
    if name.startswith("heat_") or "modis_lst" in name:
        return "heat"
    if name.startswith("green_space_") or "ndvi" in name:
        return "green"
    if name.startswith("housing_") or name.startswith("hosuing_") or "ndbi" in name:
        return "housing"
    if name.startswith("light_") or "viirs_ntl" in name or "viirs" in name or "ntl" in name:
        return "light"
    if "no2" in name or name.startswith("air_quality"):
        return "air"
    if name.startswith("population_") or "worldpop" in name:
        return "population"
    return "unknown"


def infer_year(filename: str, category: str) -> str:
    m = re.search(r"(2018|2020|2022|2024)", filename)
    if m:
        return m.group(1)
    # Population baseline
    if category == "population":
        return "2020"
    return ""


def find_area_column(df: pd.DataFrame) -> str:
    candidates = [
        "area", "Area", "AREA",
        "area_name", "Area_Name", "AREA_NAME",
        "neighborhood", "Neighborhood",
        "name", "Name", "NAME",
        "district", "District",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: choose first object column with limited unique values
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) and df[c].nunique(dropna=True) > 1 and df[c].nunique(dropna=True) < 100:
            return c
    raise ValueError("Could not identify area column in CSV")


def normalize_area_key(raw: Any) -> str:
    if raw is None:
        return ''
    s = str(raw).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]", "", s)
    return s


def metric_candidates(columns: List[str]) -> Dict[str, List[str]]:
    lc = {c.lower(): c for c in columns}
    def find_one(keys: List[str]) -> List[str]:
        return [lc[k] for k in lc if any(tag in k for tag in keys)]

    return {
        # Heat (MODIS LST)
        "mean": find_one(["lst_day_1km_mean", "dnb_brdf_corrected_ntl_mean", "ndvi_mean", "ndbi_mean", "ntl_mean", "mean", "avg", "average"]),
        "median": find_one(["median", "p50"]),
        "min": find_one(["lst_day_1km_min", "dnb_brdf_corrected_ntl_min", "population_min", "min"]),
        "max": find_one(["lst_day_1km_max", "dnb_brdf_corrected_ntl_max", "population_max", "max"]),
        "p10": find_one(["p10", "p_10", "percentile_10", "quantile_0.1"]),
        "p90": find_one(["p90", "p_90", "percentile_90", "quantile_0.9"]),
        "std": find_one(["std", "stdev", "stddev", "ndvi_stddev"]),
        # Generic value fallback
        "value": find_one(["lst", "ndvi", "ndbi", "viirs", "ntl", "value", "val", "measurement", "z"]),
        "units": find_one(["unit", "units"]),
    }


def safe_float(x: Any) -> float:
    try:
        if pd.isna(x):
            return math.nan
        return float(x)
    except Exception:
        return math.nan


def summarize_group(df: pd.DataFrame, value_col: str) -> Dict[str, Any]:
    vals = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if vals.empty:
        return {"samples": []}
    # convert to native python floats for JSON serialization
    samples = [float(x) for x in vals.tolist()]
    return {
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "p10": float(vals.quantile(0.10)),
        "p90": float(vals.quantile(0.90)),
        "std": float(vals.std(ddof=1)) if vals.shape[0] > 1 else 0.0,
        "samples": samples,
    }


def extract_units(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            # take the most common non-null string
            series = df[c].dropna().astype(str)
            if not series.empty:
                return series.mode().iloc[0]
    return ""


def normalize_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a row dict (from pandas) into JSON-serializable Python scalars.

    - pd.NA / NaN -> None
    - numpy / pandas scalar -> native python via .item() when available
    - otherwise keep as-is
    """
    out: Dict[str, Any] = {}
    for k, v in row.items():
        try:
            if pd.isna(v):
                out[k] = None
            else:
                # numpy and pandas scalar types support item()
                if hasattr(v, 'item'):
                    try:
                        out[k] = v.item()
                    except Exception:
                        out[k] = v
                else:
                    out[k] = v
        except Exception:
            # fallback: stringify anything that can't be serialized
            out[k] = str(v)
    return out


def process_csv(path: Path) -> Dict[str, Any]:
    category = infer_category(path.name)
    year = infer_year(path.name, category)
    if category == "unknown" or not year:
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        # try UTF-16 or different encodings if needed
        try:
            df = pd.read_csv(path, encoding="utf-16")
        except Exception:
            df = pd.read_csv(path, encoding="latin-1")

    area_col = find_area_column(df)
    cand = metric_candidates(list(df.columns))

    result: Dict[str, Any] = {}
    result.setdefault(year, {})

    # Special-case: population CSVs may use different population_* column names
    if category == "population":
        # prefer an explicit population_mean column if present
        pop_mean_col = None
        for c in df.columns:
            if "population_mean" in c.lower():
                pop_mean_col = c
                break

        # fallback to population_max_* and population_min_* (prefer local per-area values first)
        pop_max_col = None
        pop_min_col = None
        # prefer local (per-area) columns if available
        for c in df.columns:
            lc = c.lower()
            if "population_max_local" in lc:
                pop_max_col = c
            if "population_min_local" in lc:
                pop_min_col = c
        # if local not found, prefer global
        if not pop_max_col or not pop_min_col:
            for c in df.columns:
                lc = c.lower()
                if not pop_max_col and "population_max_global" in lc:
                    pop_max_col = c
                if not pop_min_col and "population_min_global" in lc:
                    pop_min_col = c
        # final fallback to any columns containing population_max/min
        if not pop_max_col or not pop_min_col:
            for c in df.columns:
                lc = c.lower()
                if not pop_max_col and "population_max" in lc:
                    pop_max_col = c
                if not pop_min_col and "population_min" in lc:
                    pop_min_col = c

        if pop_mean_col or (pop_max_col and pop_min_col):
            grouped = df.groupby(area_col, dropna=True)
            for area, g in grouped:
                area_key = normalize_area_key(area)
                if not area_key:
                    continue
                if pop_mean_col and pop_mean_col in g.columns:
                    stats = summarize_group(g, pop_mean_col)
                    # include original rows for reference
                    stats["records"] = [normalize_record(r) for r in g.to_dict(orient='records')]
                else:
                    # compute per-row avg of max/min then summarize
                    max_series = pd.to_numeric(g[pop_max_col], errors="coerce")
                    min_series = pd.to_numeric(g[pop_min_col], errors="coerce")
                    avg_vals = pd.concat([max_series, min_series], axis=1).mean(axis=1).dropna()
                    if avg_vals.empty:
                        stats = {"records": []}
                    else:
                        stats = {
                            "mean": float(avg_vals.mean()),
                            "median": float(avg_vals.median()),
                            "min": float(avg_vals.min()),
                            "max": float(avg_vals.max()),
                            "p10": float(avg_vals.quantile(0.10)),
                            "p90": float(avg_vals.quantile(0.90)),
                            "std": float(avg_vals.std(ddof=1)) if avg_vals.shape[0] > 1 else 0.0,
                        }
                    # attach original records
                    stats["records"] = [normalize_record(r) for r in g.to_dict(orient='records')]
                # ensure units present, prefer extracted but fallback to category default
                stats["units"] = stats.get("units", "") or CATEGORY_UNITS.get(category, "")
                result[year][area_key] = stats
            return {"category": category, "data": result}

    # Direct metrics available
    metrics: Dict[str, str] = {}
    for key in ["mean", "median", "min", "max", "p10", "p90", "std"]:
        if cand[key]:
            metrics[key] = cand[key][0]

    units = extract_units(df, cand.get("units", []))

    # Dataset-specific unit detection/overrides
    # - Housing: NDBI index (unitless) when NDBI columns are present
    # - Green: NDVI index (unitless) when NDVI columns are present
    # - Heat: prefer explicit LST unit (°C) when LST columns are present
    if category == "housing":
        if any("ndbi" in c.lower() for c in df.columns):
            units = "NDBI index (unitless; typical range -1..1)"
        else:
            units = units or CATEGORY_UNITS.get(category, "")
    elif category == "green":
        if any("ndvi" in c.lower() for c in df.columns):
            units = "NDVI index (unitless; typical range -1..1)"
        else:
            units = units or CATEGORY_UNITS.get(category, "")
    elif category == "heat":
        # detect LST-like columns and set °C
        if any("lst" in c.lower() for c in df.columns):
            units = "°C"
        else:
            units = units or CATEGORY_UNITS.get(category, "")

    if metrics and (metrics.get("mean") or metrics.get("min") or metrics.get("max")):
        # There may be one row per area (pre-computed stats) OR many rows per area (time-series).
        # If multiple rows per area exist (or a date column is present) aggregate per-area using
        # the preferred numeric column (prefer mean, else average of min/max, else a value column).
        rows_per_area = df.groupby(area_col).size()
        multiple_rows = rows_per_area.max() > 1 if not rows_per_area.empty else False

        # choose preferred numeric column to summarize
        preferred_col = None
        # For most datasets prefer the 'mean' column. For VIIRS NTL (light) prefer median when available
        if category == 'light' and metrics.get("median"):
            preferred_col = metrics.get("median")
        elif metrics.get("mean"):
            preferred_col = metrics.get("mean")
        elif metrics.get("value"):
            preferred_col = metrics.get("value")
        elif metrics.get("max") and metrics.get("min"):
            # we'll compute rowwise average of min/max
            preferred_col = None

        if multiple_rows or "date" in df.columns:
            # aggregate time-series per area
            grouped = df.groupby(area_col, dropna=True)
            for area, g in grouped:
                area_key = normalize_area_key(area)
                if not area_key:
                    continue

                if preferred_col and preferred_col in g.columns:
                    stats = summarize_group(g, preferred_col)
                    stats["records"] = [normalize_record(r) for r in g.to_dict(orient='records')]
                elif metrics.get("max") and metrics.get("min") and metrics.get("max") in g.columns and metrics.get("min") in g.columns:
                    # compute row-wise average then summarize
                    avg_series = pd.to_numeric(g[metrics.get("max")], errors="coerce")
                    min_series = pd.to_numeric(g[metrics.get("min")], errors="coerce")
                    avg_vals = pd.concat([avg_series, min_series], axis=1).mean(axis=1).dropna()
                    if avg_vals.empty:
                        stats = {"records": []}
                    else:
                        stats = {
                            "mean": float(avg_vals.mean()),
                            "median": float(avg_vals.median()),
                            "min": float(avg_vals.min()),
                            "max": float(avg_vals.max()),
                            "p10": float(avg_vals.quantile(0.10)),
                            "p90": float(avg_vals.quantile(0.90)),
                            "std": float(avg_vals.std(ddof=1)) if avg_vals.shape[0] > 1 else 0.0,
                        }
                    # original records
                    stats["records"] = [normalize_record(r) for r in g.to_dict(orient='records')]
                else:
                    # fallback: try first candidate from value list
                    vals = None
                    if cand.get("value"):
                        for vc in cand.get("value"):
                            if vc in g.columns:
                                vals = pd.to_numeric(g[vc], errors="coerce").dropna()
                                if not vals.empty:
                                    break
                    if vals is None or vals.empty:
                        stats = {"records": []}
                    else:
                        stats = {
                            "mean": float(vals.mean()),
                            "median": float(vals.median()),
                            "min": float(vals.min()),
                            "max": float(vals.max()),
                            "p10": float(vals.quantile(0.10)),
                            "p90": float(vals.quantile(0.90)),
                            "std": float(vals.std(ddof=1)) if vals.shape[0] > 1 else 0.0,
                        }
                    stats["records"] = [normalize_record(r) for r in g.to_dict(orient='records')]

                stats["units"] = units or CATEGORY_UNITS.get(category, "")
                # Make downstream consumers (which expect a 'mean' key) use the median for light datasets
                if category == 'light' and 'median' in stats and stats.get('median') is not None:
                    stats['mean'] = stats.get('median')
                result[year][area_key] = stats
        else:
            # single row per area: keep previous per-row behavior but compute mean fallback when needed
            for _, row in df.iterrows():
                area = normalize_area_key(row[area_col])
                if not area:
                    continue
                mean_val = safe_float(row.get(metrics.get("mean", "")))
                if (isinstance(mean_val, float) and math.isnan(mean_val)) or mean_val is None:
                    max_v = safe_float(row.get(metrics.get("max", "")))
                    min_v = safe_float(row.get(metrics.get("min", "")))
                    if not (isinstance(max_v, float) and math.isnan(max_v)) and not (isinstance(min_v, float) and math.isnan(min_v)):
                        mean_val = (max_v + min_v) / 2.0
                area_stats = {
                    "mean": (mean_val if (mean_val is not None and not (isinstance(mean_val, float) and math.isnan(mean_val))) else None),
                    "median": (safe_float(row.get(metrics.get("median", ""))) if pd.notna(row.get(metrics.get("median", ""))) else None),
                    "min": (safe_float(row.get(metrics.get("min", ""))) if pd.notna(row.get(metrics.get("min", ""))) else None),
                    "max": (safe_float(row.get(metrics.get("max", ""))) if pd.notna(row.get(metrics.get("max", ""))) else None),
                    "p10": (safe_float(row.get(metrics.get("p10", ""))) if pd.notna(row.get(metrics.get("p10", ""))) else None),
                    "p90": (safe_float(row.get(metrics.get("p90", ""))) if pd.notna(row.get(metrics.get("p90", ""))) else None),
                    "std": (safe_float(row.get(metrics.get("std", ""))) if pd.notna(row.get(metrics.get("std", ""))) else None),
                    "units": units or CATEGORY_UNITS.get(category, ""),
                }
                # remove keys with None or NaN values but keep 'n' and 'units'
                cleaned = {k: v for k, v in area_stats.items() if k == "units" or (v is not None and not (isinstance(v, float) and math.isnan(v)))}
                # For light single-row exports prefer to expose median as the primary 'mean' value if available
                if category == 'light' and 'median' in cleaned and cleaned.get('median') is not None:
                    cleaned['mean'] = cleaned.get('median')
                # attach original row
                cleaned["records"] = [normalize_record(row.to_dict())]
                result[year][area] = cleaned
    else:
        # Need to compute stats from value column(s)
        value_cols = cand["value"]
        if not value_cols:
            # No usable numeric column
            return {}
        value_col = value_cols[0]
        grouped = df.groupby(area_col, dropna=True)
        for area, g in grouped:
            area_key = str(area).strip().lower().replace(" ", "_")
            stats = summarize_group(g, value_col)
            stats["units"] = units
            # attach original records
            stats["records"] = [normalize_record(r) for r in g.to_dict(orient='records')]
            result[year][area_key] = stats

    return {"category": category, "data": result}


def merge_into(accum: Dict[str, Dict[str, Dict[str, Any]]], piece: Dict[str, Any]) -> None:
    category = piece["category"]
    data = piece["data"]
    if category not in accum:
        accum[category] = {}
    for year, areas in data.items():
        accum[category].setdefault(year, {})
        accum[category][year].update(areas)


def main() -> None:
    ensure_export_dir()
    accum: Dict[str, Dict[str, Dict[str, Any]]] = {}
    csv_paths = sorted(CSV_DIR.glob("*.csv"))
    if not csv_paths:
        print("No CSV files found in", CSV_DIR)
    for p in csv_paths:
        piece = process_csv(p)
        if piece:
            merge_into(accum, piece)

    # Write out JSON per category
    mapping = {
        "heat": "heat.json",
        "green": "green.json",
        "housing": "housing.json",
        "light": "light.json",
        "air": "air.json",
        "population": "population.json",
    }
    for category, filename in mapping.items():
        out_path = EXPORT_DIR / filename
        payload = accum.get(category, {})
        # Build meta: list of areas, years, and city_means per year (array of means across areas)
        meta: Dict[str, Any] = {}
        years = sorted(payload.keys())
        all_areas = set()
        city_means: Dict[str, List[float]] = {}
        for y in years:
            areas = payload.get(y, {})
            for a, stats in areas.items():
                all_areas.add(a)
            # collect means for this year
            means = []
            for a, stats in areas.items():
                m = stats.get("mean")
                if isinstance(m, (int, float)) and not (isinstance(m, float) and math.isnan(m)):
                    means.append(float(m))
            city_means[y] = means

        meta["years"] = years
        meta["areas"] = sorted(list(all_areas))
        meta["city_means"] = city_means

        # attach meta under a reserved key
        out_payload = {"__meta__": meta}
        out_payload.update(payload)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=False, separators=(",", ":"))
        print("Wrote", out_path)


if __name__ == "__main__":
    main()



