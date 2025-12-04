#!/usr/bin/env python3
"""Build a canonical premodel CSV by combining mapped CSV and cluster CSV.

Assumes both files have same row order (they were produced from the same source).
Writes artifacts/premodel_canonical.csv
"""
from pathlib import Path
import pandas as pd

ART = Path('artifacts')
mapped = ART / 'premodel_mapped.csv'
clusters = ART / 'premodel_clusters.csv'
out = ART / 'premodel_canonical.csv'

if not mapped.exists():
    raise SystemExit(f'missing {mapped}')
if not clusters.exists():
    raise SystemExit(f'missing {clusters}')

dm = pd.read_csv(mapped)
dc = pd.read_csv(clusters)

if len(dm) != len(dc):
    print('Row counts differ: mapped', len(dm), 'clusters', len(dc))

# take all columns from mapped and add weather_tire_cluster from clusters
dc_col = 'weather_tire_cluster'
if dc_col not in dc.columns:
    raise SystemExit(f'{dc_col} not present in {clusters}')

dm[dc_col] = dc[dc_col].values
dm.to_csv(out, index=False)
print('Wrote', out)
