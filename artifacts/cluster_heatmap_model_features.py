"""Create a cluster heatmap using only the columns used in the minimal regression model.

Reads: artifacts/premodel_clusters.csv
Writes: artifacts/cluster_heatmap_model_features.png
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parents[1]
ART = BASE / 'artifacts'

# features used in the regression model
MODEL_FEATURES = [
    'GridPosition',
    'AvgQualiTime',
    'SOFT','MEDIUM','HARD','INTERMEDIATE','WET',
    'races_prior_this_season',
    'Rain'
]

def main():
    csv = ART / 'premodel_clusters.csv'
    if not csv.exists():
        print('premodel_clusters.csv not found in artifacts/. Cannot build heatmap.')
        return
    cdf = pd.read_csv(csv)

    # verify cluster label exists
    if 'weather_tire_cluster' not in cdf.columns:
        print('weather_tire_cluster column missing in premodel_clusters.csv')
        return

    # select model features present in the cluster df
    present = [f for f in MODEL_FEATURES if f in cdf.columns]
    if not present:
        print('None of the model features found in clusters file:', MODEL_FEATURES)
        return

    # coerce numeric where possible; tyre flags may be floats already
    df_num = cdf[present].copy()
    for col in df_num.columns:
        df_num[col] = pd.to_numeric(df_num[col], errors='coerce').fillna(0)

    grouped = df_num.groupby(cdf['weather_tire_cluster']).mean()
    if grouped.empty:
        print('No data after grouping — check premodel_clusters.csv')
        return

    plt.figure(figsize=(max(6, grouped.shape[1]*1.2), max(4, grouped.shape[0]*0.6)))
    sns.set(font_scale=1.0)
    ax = sns.heatmap(grouped, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'label':'mean'})
    ax.set_xlabel('feature')
    ax.set_ylabel('weather_tire_cluster')
    plt.title('Cluster means — model features')
    plt.tight_layout()
    out = ART / 'cluster_heatmap_model_features.png'
    plt.savefig(out)
    print('Wrote', out)

if __name__ == '__main__':
    main()
