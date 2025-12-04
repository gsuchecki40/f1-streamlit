#!/usr/bin/env python3
"""Cluster weather and tire usage into categorical strategies.

Outputs:
 - artifacts/premodel_clusters.csv (original rows + cluster labels)
 - artifacts/cluster_summary.csv (per-cluster feature means)

Uses KMeans on scaled features. Wind direction is converted to sin/cos.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
try:
    from tqdm import tqdm
except Exception:
    # fallback: identity function
    def tqdm(x, **_kw):
        return x

P = Path('premodeldatav1.csv')
out_csv = Path('artifacts/premodel_clusters.csv')
summary_csv = Path('artifacts/cluster_summary.csv')
Path('artifacts').mkdir(exist_ok=True)

df = pd.read_csv(P)

# select weather and tire columns
cols_weather = ['AirTemp_C','TrackTemp_C','Humidity_%','Pressure_hPa','WindSpeed_mps','WindDirection_deg']
cols_tires = ['SOFT','MEDIUM','HARD','INTERMEDIATE','WET']

available = [c for c in cols_weather+cols_tires if c in df.columns]
if not available:
    raise RuntimeError('No weather/tire columns found in CSV')

work = df[available].copy()

# convert wind direction to sin/cos if present
if 'WindDirection_deg' in work.columns:
    wd = work['WindDirection_deg'].fillna(0).astype(float)
    # convert degrees to radians
    rad = np.deg2rad(wd)
    work['WindDir_sin'] = np.sin(rad)
    work['WindDir_cos'] = np.cos(rad)
    work = work.drop(columns=['WindDirection_deg'])

# fillna for numeric
for c in work.columns:
    work[c] = pd.to_numeric(work[c], errors='coerce')
    work[c] = work[c].fillna(work[c].median())

# scale
scaler = StandardScaler()
X = scaler.fit_transform(work)

# cluster range configurable
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
labels = kmeans.fit_predict(X)

# attach labels back to df and save a new CSV
out = df.copy()
out['weather_tire_cluster'] = labels
out.to_csv(out_csv, index=False)

# cluster summary
centers = scaler.inverse_transform(kmeans.cluster_centers_)
summary = pd.DataFrame(centers, columns=work.columns)
summary['cluster'] = range(n_clusters)
summary = summary[['cluster'] + [c for c in work.columns]]
summary.to_csv(summary_csv, index=False)

print('Saved clusters to', out_csv)
print('Saved cluster summary to', summary_csv)
