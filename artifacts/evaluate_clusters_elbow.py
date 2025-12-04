#!/usr/bin/env python3
"""Evaluate KMeans for k in range and save metrics + elbow plot."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

art = Path('artifacts')
in_csv = art / 'premodel_clusters.csv'
out_metrics = art / 'cluster_elbow_metrics.csv'
out_plot = art / 'cluster_elbow.png'

if not in_csv.exists():
    raise SystemExit('Missing ' + str(in_csv))

df = pd.read_csv(in_csv)

# pick same features as clustering script
num_cols = []
for c in ['AirTemp_C','TrackTemp_C','Humidity_%','Pressure_hPa','WindSpeed_mps','WindDirection_deg','SOFT','MEDIUM','HARD','INTERMEDIATE','WET']:
    if c in df.columns:
        num_cols.append(c)

if not num_cols:
    raise SystemExit('No numeric features found for clustering evaluation')

work = df[num_cols].copy()
if 'WindDirection_deg' in work.columns:
    wd = work['WindDirection_deg'].fillna(0).astype(float)
    rad = np.deg2rad(wd)
    work['WindDir_sin'] = np.sin(rad)
    work['WindDir_cos'] = np.cos(rad)
    work = work.drop(columns=['WindDirection_deg'])

for c in work.columns:
    work[c] = pd.to_numeric(work[c], errors='coerce')
    work[c] = work[c].fillna(work[c].median())

X = StandardScaler().fit_transform(work)

ks = list(range(2, 11))
rows = []
inertias = []
silhs = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X)
    inertia = km.inertia_
    inertias.append(inertia)
    try:
        sil = silhouette_score(X, labels)
    except Exception:
        sil = np.nan
    silhs.append(sil)
    rows.append({'k': k, 'inertia': inertia, 'silhouette': sil})

pd.DataFrame(rows).to_csv(out_metrics, index=False)

plt.figure(figsize=(6,4))
plt.plot(ks, inertias, '-o', label='inertia')
plt.xlabel('k')
plt.ylabel('inertia')
plt.twinx()
plt.plot(ks, silhs, '-s', color='C1', label='silhouette')
plt.ylabel('silhouette')
plt.title('KMeans: inertia & silhouette')
plt.grid(True)
plt.savefig(out_plot, bbox_inches='tight', dpi=150)

print('Saved metrics to', out_metrics)
print('Saved elbow plot to', out_plot)
