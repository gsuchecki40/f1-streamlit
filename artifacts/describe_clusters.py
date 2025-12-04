#!/usr/bin/env python3
"""Generate human-friendly summaries for weather/tire clusters.

Reads artifacts/premodel_clusters.csv and artifacts/cluster_summary.csv and
produces a short textual description per cluster and a CSV summary.
"""
import pandas as pd
from pathlib import Path
import numpy as np

art = Path('artifacts')
in_csv = art / 'premodel_clusters.csv'
summary_csv = art / 'cluster_summary.csv'
out_txt = art / 'cluster_readable_summary.txt'
out_csv = art / 'cluster_readable_summary.csv'

if not in_csv.exists():
    raise SystemExit('Missing ' + str(in_csv))

df = pd.read_csv(in_csv)
if 'weather_tire_cluster' not in df.columns:
    raise SystemExit('premodel_clusters.csv missing weather_tire_cluster')

# aggregate numeric features to describe clusters
num_cols = []
for c in ['AirTemp_C','TrackTemp_C','Humidity_%','Pressure_hPa','WindSpeed_mps','SOFT','MEDIUM','HARD','INTERMEDIATE','WET']:
    if c in df.columns:
        num_cols.append(c)

groups = df.groupby('weather_tire_cluster')

rows = []
lines = []
for cluster, g in groups:
    n = len(g)
    stats = g[num_cols].median().to_dict() if num_cols else {}
    # basic textual rules
    temp = None
    if 'AirTemp_C' in stats:
        at = stats['AirTemp_C']
        if at >= 30:
            temp = 'hot'
        elif at >= 20:
            temp = 'warm'
        elif at >= 10:
            temp = 'mild'
        else:
            temp = 'cool'
    hum = None
    if 'Humidity_%' in stats:
        h = stats['Humidity_%']
        if h >= 80:
            hum = 'very humid'
        elif h >= 60:
            hum = 'humid'
        elif h >= 30:
            hum = 'moderate humidity'
        else:
            hum = 'dry'

    # tires: pick dominant tyre by median usage
    tyre = None
    tyre_counts = {k: stats[k] for k in ['SOFT','MEDIUM','HARD','INTERMEDIATE','WET'] if k in stats}
    if tyre_counts:
        tyre = max(tyre_counts.items(), key=lambda x: x[1])[0]

    desc = f'Cluster {cluster}: {n} rows'
    parts = []
    if temp:
        parts.append(temp)
    if hum:
        parts.append(hum)
    if tyre:
        parts.append(f'dominant tyre: {tyre}')
    if parts:
        desc += ' — ' + ', '.join(parts)

    lines.append(desc)
    row = {'cluster': cluster, 'n': n}
    for k, v in stats.items():
        row[k] = v
    rows.append(row)

with open(out_txt, 'w') as f:
    f.write('\n'.join(lines) + '\n')

pd.DataFrame(rows).sort_values('cluster').to_csv(out_csv, index=False)

print('Wrote', out_txt, 'and', out_csv)
