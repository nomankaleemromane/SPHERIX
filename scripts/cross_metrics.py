#!/usr/bin/env python3
"""
cross_metrics.py

Reads per-AOI exported JSON files from the `export/` folder and computes pairwise cross-metrics.
Outputs a single JSON `export/cross_metrics.json` with per-year, per-area metric objects that include
status text, recommendation, and severity (green/yellow/red).

Usage: python cross_metrics.py

This script is conservative: if a dataset/year/area is missing it will skip that metric gracefully.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # go one level up (project root)
EXPORT_DIR = ROOT / 'export'
OUT_FILE = EXPORT_DIR / 'cross_metrics.json'


DATASETS = {
    'heat': 'heat.json',
    'green': 'green.json',
    'housing': 'housing.json',
    'light': 'light.json',
    'air': 'air.json',
    'population': 'population.json'
}

# Pairs to compute (left x right)
PAIRS = [
    ('housing','population'),
    ('housing','light'),
    ('population','light'),
    ('housing','green'),
    ('housing','air'),
    ('population','green'),
    ('housing','heat'),
    ('population','heat')
]

SEMANTICS = {
    'air': 'higher_worse',
    'heat': 'higher_worse',
    'housing': 'higher_worse',
    'green': 'higher_better',
    'light': 'neutral',
    'population': 'neutral'
}

def load_dataset(fname):
    p = EXPORT_DIR / fname
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf8'))
    except Exception:
        return None

def percentile_of(value, arr):
    """Compute fraction of values strictly below value (0..1)."""
    if value is None or arr is None or not arr:
        return None
    nums = [float(x) for x in arr if isinstance(x, (int, float))]
    if not nums:
        return None
    nums = sorted(nums)
    below = sum(1 for x in nums if x < value)
    return below / len(nums)

def interpret_semantic(cat, p):
    # p expected in 0..1
    if p is None:
        return None
    sem = SEMANTICS.get(cat, 'neutral')
    if sem == 'higher_worse':
        if p <= 0.25: return ('Good', 0)
        if p <= 0.6: return ('Moderate', 1)
        return ('Poor', 2)
    if sem == 'higher_better':
        if p <= 0.25: return ('Low', 2)
        if p <= 0.6: return ('Moderate', 1)
        return ('High', 0)
    # neutral
    if p <= 0.25: return ('Low', 0)
    if p <= 0.6: return ('Medium', 1)
    return ('High', 2)

def severity_label(n):
    return {0: 'green', 1: 'yellow', 2: 'red'}.get(n, 'yellow')


def ordinal(n):
    """Return an ordinal string for an integer, e.g. 1 -> '1st', 2 -> '2nd', 11 -> '11th'."""
    try:
        n = int(n)
    except Exception:
        return str(n)
    if 10 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def format_value_with_units(value, units: str) -> str:
    """Return a short human-readable string for a numeric value with units.

    Choose formatting precision based on the unit type to keep output compact and informative.
    """
    if value is None:
        return "(no data)"
    try:
        v = float(value)
    except Exception:
        return str(value)

    u = (units or "").strip()
    low = u.lower()
    # Index-like units (NDVI / NDBI) -> more precision
    if 'ndbi' in low or 'ndvi' in low or 'index' in low:
        return f"{v:.4f} {u}" if u else f"{v:.4f}"
    # Temperature
    if '°c' in low or 'celsius' in low or '°' in low:
        return f"{v:.2f} {u}" if u else f"{v:.2f}"
    # Population-like -> integer
    if 'people' in low or 'person' in low or 'population' in low:
        try:
            return f"{int(round(v))} {u}" if u else f"{int(round(v))}"
        except Exception:
            pass
    # Default: two decimals
    return f"{v:.2f} {u}" if u else f"{v:.2f}"

def recommend_for_pair(a_cat, a_label, a_sev, b_cat, b_label, b_sev):
    # Simple heuristic recommendations tailored for common pairings
    maxsev = max(a_sev, b_sev)
    if maxsev == 0:
        return 'Situation looks acceptable; maintain current management and monitor trends.'
    # Specific heuristics
    if a_cat == 'housing' and b_cat == 'green' or (a_cat == 'green' and b_cat == 'housing'):
        # If housing high and green low -> plant trees / parks
        if (a_cat=='housing' and a_sev==2 and b_sev>=1) or (b_cat=='housing' and b_sev==2 and a_sev>=1):
            return 'High built area and low vegetation — prioritise greening (street trees, pocket parks, urban canopy).'
    # Housing/Heat: recommend cooling measures only when heat or housing/population severity is elevated
    if (a_cat in ('housing','population') and b_cat == 'heat') or (b_cat in ('housing','population') and a_cat == 'heat'):
        # identify heat severity and population/housing severity
        heat_sev = a_sev if a_cat == 'heat' else b_sev if b_cat == 'heat' else 0
        pop_house_sev = max(a_sev if a_cat in ('housing','population') else 0, b_sev if b_cat in ('housing','population') else 0)
        if heat_sev >= 2 or (heat_sev >= 1 and pop_house_sev >= 1):
            return 'High exposure to heat in populated/built areas — implement cooling measures (shading, cool roofs, tree planting).'
        if heat_sev == 1:
            return 'Moderate heat exposure — consider targeted cooling interventions and monitor trends.'
        return 'Heat levels are low; continue monitoring and consider passive cooling in developments.'

    # Housing/Air or Population/Air: tailor message based on air severity
    if (a_cat in ('population','housing') and b_cat == 'air') or (b_cat in ('population','housing') and a_cat == 'air'):
        air_sev = a_sev if a_cat == 'air' else b_sev if b_cat == 'air' else 0
        pop_house_sev = max(a_sev if a_cat in ('housing','population') else 0, b_sev if b_cat in ('housing','population') else 0)
        if air_sev >= 2:
            if pop_house_sev >= 1:
                return 'High population/housing with poor air quality — investigate pollution sources and protect residents (filters, traffic reduction).'
            return 'Poor air quality detected — investigate sources and consider mitigation (emissions control, monitoring).'
        if air_sev == 1:
            return 'Moderate air quality — monitor closely and investigate local sources if trends worsen.'
        return 'Air quality is good; continue routine monitoring and focus on long-term planning measures.'

    # Lighting in populated/built areas: recommend only when brightness is elevated
    if (a_cat in ('population','housing') and b_cat == 'light') or (b_cat in ('population','housing') and a_cat == 'light'):
        light_sev = a_sev if a_cat == 'light' else b_sev if b_cat == 'light' else 0
        pop_house_sev = max(a_sev if a_cat in ('housing','population') else 0, b_sev if b_cat in ('housing','population') else 0)
        if light_sev >= 2 and pop_house_sev >= 1:
            return 'High brightness in populated/built areas — review lighting design, consider light pollution reduction and safety-focused lighting.'
        if light_sev == 1 and pop_house_sev >= 1:
            return 'Moderate lighting levels — assess lighting efficiency and community impact.'
        return 'Lighting levels are within expected ranges; prioritise safety and energy efficiency.'
    # Generic escalation advice based on severity
    if maxsev == 2:
        return 'High concern detected — immediate investigation and targeted mitigation recommended.'
    return 'Monitor the situation and consider targeted interventions where appropriate.'

def main():
    # Load all datasets
    datasets = {}
    metas = {}
    for k, fname in DATASETS.items():
        j = load_dataset(fname)
        if j is None:
            print(f'Warning: dataset {k} ({fname}) not available or empty')
            continue
        datasets[k] = j
        metas[k] = j.get('__meta__', {})

    # Determine available years per dataset
    years_by_ds = {}
    for k, j in datasets.items():
        years = [y for y in j.keys() if not y.startswith('__')]
        years_by_ds[k] = set(years)

    # Prepare output structure
    out = {}

    for a, b in PAIRS:
        if a not in datasets or b not in datasets:
            # skip pairs where either dataset is missing
            continue
        years = years_by_ds[a].intersection(years_by_ds[b])
        if not years:
            continue
        for year in sorted(years):
            out.setdefault(year, {})
            # city arrays for percentile calculation: prefer __meta__.city_means if available
            city_a = metas.get(a, {}).get('city_means', {}).get(year) or []
            city_b = metas.get(b, {}).get('city_means', {}).get(year) or []
            # fallback: derive from per-area means in dataset
            if not city_a:
                byyear = datasets[a].get(year, {})
                city_a = [v.get('mean') for v in byyear.values() if isinstance(v.get('mean'), (int, float))]
            if not city_b:
                byyear = datasets[b].get(year, {})
                city_b = [v.get('mean') for v in byyear.values() if isinstance(v.get('mean'), (int, float))]

            # areas to iterate over: union of areas present in either dataset for that year
            areas = set(list(datasets[a].get(year, {}).keys()) + list(datasets[b].get(year, {}).keys()))
            for area in sorted(areas):
                a_rec = datasets[a].get(year, {}).get(area)
                b_rec = datasets[b].get(year, {}).get(area)
                # require at least one of the two means to be numeric
                if not a_rec or not b_rec:
                    # if either missing, skip
                    continue
                a_val = a_rec.get('mean')
                b_val = b_rec.get('mean')
                if a_val is None or b_val is None:
                    # skip if missing numeric means
                    continue

                p_a = percentile_of(a_val, city_a)
                p_b = percentile_of(b_val, city_b)
                interp_a = interpret_semantic(a, p_a)
                interp_b = interpret_semantic(b, p_b)
                if interp_a is None or interp_b is None:
                    continue
                label_a, sev_a = interp_a
                label_b, sev_b = interp_b
                # combined severity: max of components
                comb = max(sev_a, sev_b)
                severity = severity_label(comb)

                # Friendly status & recommendation
                pct_a = None if p_a is None else int(round(p_a * 100))
                pct_b = None if p_b is None else int(round(p_b * 100))
                # human-friendly formatted values including units
                units_a = a_rec.get('units', '') if isinstance(a_rec, dict) else ''
                units_b = b_rec.get('units', '') if isinstance(b_rec, dict) else ''
                value_a_text = format_value_with_units(a_val, units_a)
                value_b_text = format_value_with_units(b_val, units_b)
                # Compose status including percentile and formatted value with units
                pa_str = (f"{ordinal(pct_a)} percentile" if pct_a is not None else "no percentile")
                pb_str = (f"{ordinal(pct_b)} percentile" if pct_b is not None else "no percentile")
                status = f"{a.capitalize()}: {label_a} ({pa_str}, {value_a_text}); {b.capitalize()}: {label_b} ({pb_str}, {value_b_text})"
                recommendation = recommend_for_pair(a, label_a, sev_a, b, label_b, sev_b)

                # write entry
                area_out = out[year].setdefault(area, [])
                metric_id = f"{a}_{b}"
                area_out.append({
                    'id': metric_id,
                    'title': f"{a.capitalize()} × {b.capitalize()}",
                    'dataset_a': a,
                    'dataset_b': b,
                    'value_a': a_val,
                    'value_b': b_val,
                    'units_a': units_a,
                    'units_b': units_b,
                    'value_a_text': value_a_text,
                    'value_b_text': value_b_text,
                    'percentile_a': p_a,
                    'percentile_b': p_b,
                    'status': status,
                    'recommendation': recommendation,
                    'severity': severity
                })

    # Ensure export directory exists
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2), encoding='utf8')
    print(f'Wrote cross metrics to {OUT_FILE}')

if __name__ == '__main__':
    main()
