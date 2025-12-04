#!/usr/bin/env python3
"""Compute PCA (2D) of weather/tire features and plot clusters.

Outputs:
 - artifacts/cluster_pca_2d.png (static matplotlib PNG)
 - artifacts/cluster_pca_2d.html (interactive Plotly HTML, if plotly is installed)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

art = Path('artifacts')
in_csv = art / 'premodel_clusters.csv'
out_png = art / 'cluster_pca_2d.png'
out_html = art / 'cluster_pca_2d.html'

if not in_csv.exists():
    raise SystemExit('Missing ' + str(in_csv))

df = pd.read_csv(in_csv)
if 'weather_tire_cluster' not in df.columns:
    raise SystemExit('premodel_clusters.csv missing weather_tire_cluster')

# choose features consistent with previous scripts
num_cols = []
for c in ['AirTemp_C','TrackTemp_C','Humidity_%','Pressure_hPa','WindSpeed_mps','WindDirection_deg','SOFT','MEDIUM','HARD','INTERMEDIATE','WET']:
    if c in df.columns:
        num_cols.append(c)

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

pca = PCA(n_components=2, random_state=42)
Xp = pca.fit_transform(X)

clusters = df['weather_tire_cluster'].astype(int).values

plt.figure(figsize=(7,6))
scatter = plt.scatter(Xp[:,0], Xp[:,1], c=clusters, cmap='tab10', s=12, alpha=0.8)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA (2D) of weather/tire features colored by cluster')
plt.colorbar(scatter, label='cluster')
plt.grid(alpha=0.2)
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print('Saved PCA plot to', out_png)

# per-cluster centroids in PCA space
centroids = []
for k in sorted(np.unique(clusters)):
    mask = clusters == k
    cx = Xp[mask].mean(axis=0)
    centroids.append({'cluster': int(k), 'pc1': float(cx[0]), 'pc2': float(cx[1]), 'n': int(mask.sum())})

centroids_df = pd.DataFrame(centroids).sort_values('cluster')
print('Cluster PCA centroids:')
print(centroids_df.to_string(index=False))

# try to write an interactive plotly html if available
try:
    import plotly.express as px
    fig = px.scatter(x=Xp[:,0], y=Xp[:,1], color=clusters.astype(str), labels={'x':'PC1','y':'PC2','color':'cluster'})
    fig.update_traces(marker={'size':6, 'opacity':0.8})
    fig.write_html(out_html)
    print('Wrote interactive HTML to', out_html)
except Exception:
    print('Plotly not available; skipped interactive HTML')
