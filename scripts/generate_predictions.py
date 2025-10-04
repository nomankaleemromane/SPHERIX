#!/usr/bin/env python3
"""
Generate simple linear extrapolation predictions for datasets under DATASETS/CSV_DATA.
Scans CSV files, aggregates per-area yearly means, fits a linear model (year -> mean) when possible,
and predicts values for target years [2026,2028,2030].

Outputs JSON to export/predictions.json and export/predictions_<dataset>.json
"""
import os
import re
import json
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'DATASETS', 'CSV_DATA')
EXPORT_DIR = os.path.join(ROOT, 'export')
os.makedirs(EXPORT_DIR, exist_ok=True)

TARGET_YEARS = [2026, 2028, 2030]

def detect_dataset_key(filename):
    n = filename.lower()
    if 'air' in n or 'no2' in n:
        return 'air'
    if 'light' in n or 'viirs' in n or 'ntl' in n:
        return 'light'
    if 'housing' in n or 'ndbi' in n:
        return 'housing'
    if 'population' in n or 'worldpop' in n:
        return 'population'
    if 'ndvi' in n or 'green' in n:
        return 'green'
    if 'modis' in n or 'lst' in n or 'heat' in n:
        return 'heat'
    return None

def extract_year_from_filename(filename):
    m = re.search(r'(20\d{2})', filename)
    if m:
        return int(m.group(1))
    return None

def pick_value_column(df, dataset_key):
    cols = [c.lower() for c in df.columns]
    lookup = []
    if dataset_key == 'air':
        lookup = ['no2_column_number_density_mean', 'no2_mean', 'no2_column_mean', 'mean_no2']
    elif dataset_key == 'light':
        # VIIRS NTL exports often provide median values; prefer 'median' or explicit median columns
        lookup = ['dnb_brdf_corrected_ntl_median', 'dnb_brdf_corrected_ntl_mean', 'dnb_median', 'dnb_mean', 'ntlf_median', 'ntlf_mean', 'mean_ntl']
    elif dataset_key == 'housing':
        lookup = ['ndbi_mean', 'ndbi_mean_']
    elif dataset_key == 'population':
        lookup = ['population_mean', 'population_max_local', 'population_max_global', 'population_max', 'population']
    elif dataset_key == 'green':
        lookup = ['ndvi_mean', 'ndvi']
    elif dataset_key == 'heat':
        lookup = ['lst_mean', 'modis_lst_mean', 'mean_lst']

    # first try prioritized lookup names
    for name in lookup:
        for c in df.columns:
            if c.lower() == name:
                return c
    # fallback: pick the first column that contains 'mean'
    for c in df.columns:
        if 'mean' in c.lower():
            return c
    # fallback: any numeric column besides coords/geometry
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # filter out obvious coordinate columns
    numeric_cols = [c for c in numeric_cols if not re.search(r'coord|lat|lon|latitude|longitude|x|y', c, re.I)]
    if numeric_cols:
        return numeric_cols[0]
    return None

def load_and_aggregate():
    # data[dataset][area][year] = mean
    data = defaultdict(lambda: defaultdict(dict))
    files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith('.csv')])
    for fn in files:
        path = os.path.join(DATA_DIR, fn)
        dataset = detect_dataset_key(fn)
        if not dataset:
            # skip unknown files
            continue
        # (previously population was skipped; now included and handled specially later)
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            print('Failed to read', path, e)
            continue

        # find year
        year = extract_year_from_filename(fn)
        if 'year' in df.columns:
            # try to use the first non-null year value
            try:
                ys = df['year'].dropna().unique()
                if len(ys) and str(ys[0]).strip():
                    year = int(ys[0])
            except Exception:
                pass
        if not year:
            # skip if no year found
            print('No year for', fn, 'skipping')
            continue

        # area column
        area_col = None
        for cand in ['area_name', 'area', 'name', 'area_name ']:
            if cand in df.columns:
                area_col = cand
                break
        if area_col is None:
            # try case-insensitive
            for c in df.columns:
                if c.lower() == 'area_name' or c.lower() == 'area' or c.lower().endswith('area_name'):
                    area_col = c
                    break
        if area_col is None:
            print('No area column in', fn, 'skipping')
            continue

        val_col = pick_value_column(df, dataset)
        if not val_col:
            print('No value column detected for', fn, 'skipping')
            continue

        # coerce to numeric
        df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
        # For light data (VIIRS NTL) prefer median aggregation if column suggests median or file uses median
        if dataset == 'light' and ('median' in val_col.lower() or 'median' in [c.lower() for c in df.columns]):
            grouped = df.groupby(area_col)[val_col].median()
        else:
            grouped = df.groupby(area_col)[val_col].mean()
        for area, v in grouped.items():
            if pd.isna(v):
                continue
            a = str(area).strip()
            data[dataset][a][year] = float(v)

    return data

def compute_r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else None


def fit_and_predict(series):
    # series: dict year->value
    years = sorted(list(series.keys()))
    x = np.array(years, dtype=float)
    y = np.array([series[y] for y in years], dtype=float)
    n = len(x)
    if n < 2:
        return None

    candidates = []
    # linear always available
    coef1 = np.polyfit(x, y, 1)
    slope1, intercept1 = float(coef1[0]), float(coef1[1])
    yhat1 = slope1 * x + intercept1
    r2_1 = compute_r2(y, yhat1)
    preds1 = {ty: float(slope1 * ty + intercept1) for ty in TARGET_YEARS}
    candidates.append({'model': 'linear', 'r2': r2_1, 'coef': [slope1, intercept1], 'preds': preds1, 'yhat': yhat1})

    # quadratic if we have at least 3 points
    if n >= 3:
        coef2 = np.polyfit(x, y, 2)
        a, b, c = float(coef2[0]), float(coef2[1]), float(coef2[2])
        yhat2 = a * x**2 + b * x + c
        r2_2 = compute_r2(y, yhat2)
        preds2 = {ty: float(a * ty**2 + b * ty + c) for ty in TARGET_YEARS}
        candidates.append({'model': 'quadratic', 'r2': r2_2, 'coef': [a, b, c], 'preds': preds2, 'yhat': yhat2})

    # choose best by R2 (prefer higher, but require at least some improvement)
    best = None
    best_r2 = -999
    for cnd in candidates:
        r2 = cnd['r2'] if cnd['r2'] is not None else -999
        if r2 > best_r2:
            best_r2 = r2
            best = cnd

    selected = best
    preds = selected['preds']
    model_type = selected['model']
    coeffs = selected['coef']
    r2 = selected['r2']

    # trend estimate: use slope for linear, derivative for quadratic at last year
    if model_type == 'linear':
        slope = coeffs[0]
    else:
        # derivative: 2*a*year + b, use most recent year
        a, b = coeffs[0], coeffs[1]
        slope = 2 * a * float(x[-1]) + b

    last_year = int(x[-1])
    last_value = float(y[-1])
    trend = 'increasing' if slope > 0 else ('decreasing' if slope < 0 else 'stable')

    # include a warning if R2 is low or few points
    warning = None
    if r2 is None or (r2 is not None and r2 < 0.3):
        warning = 'Low R² — fit may be unreliable.'
    if n < 3:
        warning = (warning + ' ') if warning else ''
        warning = (warning or '') + 'Insufficient historical points for complex models.'

    return {
        'predictions': preds,
        'model_type': model_type,
        'coefficients': coeffs,
        'r2': r2,
        'num_points': int(n),
        'last_year': last_year,
        'last_value': last_value,
        'trend': trend,
        'warning': warning
    }

def generate():
    data = load_and_aggregate()
    out = {}
    per_dataset = {}
    for dataset, areas in data.items():
        per_dataset[dataset] = {}
        # try to read units from existing export/<dataset>.json so predictions include units
        export_units_map = {}
        try:
            export_path = os.path.join(EXPORT_DIR, f'{dataset}.json')
            if os.path.exists(export_path):
                with open(export_path, 'r', encoding='utf-8') as ef:
                    exp = json.load(ef)
                # exp has year keys (e.g., "2018") mapping to per-area objects
                for yr, yrdict in exp.items():
                    if yr == '__meta__' or not isinstance(yrdict, dict):
                        continue
                    for area_key, area_obj in yrdict.items():
                        try:
                            if isinstance(area_obj, dict) and 'units' in area_obj and area_obj.get('units'):
                                export_units_map[area_key.lower()] = area_obj.get('units')
                        except Exception:
                            continue
        except Exception:
            export_units_map = {}
        for area, series in areas.items():
            # Special handling for population: if only baseline year present (2020),
            # generate simple compound-growth predictions rather than fitting a curve.
            info = None
            if dataset == 'population':
                years_present = sorted(series.keys())
                # if only 2020 exists, apply default CAGR
                if len(years_present) == 1 and int(years_present[0]) == 2020:
                    baseline = float(series[2020])
                    # default annual growth rate (CAGR). This is configurable; choose 1.5% as conservative default
                    cagr = 0.015
                    preds = {}
                    for ty in TARGET_YEARS:
                        years_ahead = ty - 2020
                        preds[ty] = float(baseline * ((1 + cagr) ** years_ahead))
                    info = {
                        'predictions': preds,
                        'model_type': 'cagr',
                        'cagr': cagr,
                        'num_points': 1,
                        'last_year': 2020,
                        'last_value': baseline,
                        'trend': 'increasing' if cagr > 0 else ('decreasing' if cagr < 0 else 'stable'),
                        'warning': 'Population predicted using a fixed CAGR from 2020 baseline.'
                    }
                else:
                    info = fit_and_predict(series)
            else:
                info = fit_and_predict(series)

            # determine units for this area by looking up in export_units_map (case-insensitive)
            unit_val = None
            try:
                if area and isinstance(area, str):
                    unit_val = export_units_map.get(area.lower())
            except Exception:
                unit_val = None

            per_dataset[dataset][area] = {
                'history': {str(k): v for k, v in sorted(series.items())},
                'prediction': info,
                'units': unit_val
            }
        out[dataset] = per_dataset[dataset]

    # write combined JSON
    out_path = os.path.join(EXPORT_DIR, 'predictions.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('Wrote', out_path)

    # write per-dataset files
    for ds, body in per_dataset.items():
        p = os.path.join(EXPORT_DIR, f'predictions_{ds}.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({ds: body}, f, indent=2)
        print('Wrote', p)

if __name__ == '__main__':
    generate()
