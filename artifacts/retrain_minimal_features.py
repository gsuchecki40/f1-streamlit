"""Retrain XGBoost using a minimal set of explanatory variables and compute SHAP.

Features used (explicit):
- GridPosition
- AvgQualiTime
- weather_tire_cluster (from premodel_clusters.csv)
- SOFT, MEDIUM, HARD, INTERMEDIATE, WET
- races_prior_this_season
- Rain (binary, produced by preprocess_premodel.py)

Outputs saved to artifacts/: xgb_minimal.joblib, columns_minimal.json, shap_minimal_summary.csv and plots
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
ART = BASE / 'artifacts'
ART.mkdir(exist_ok=True)

CSV = BASE / 'premodeldatav1.csv'

FEATURES = [
    'GridPosition',
    'AvgQualiTime',
    'weather_tire_cluster',
    'SOFT','MEDIUM','HARD','INTERMEDIATE','WET',
    'races_prior_this_season',
    'Rain'
]

TARGET = 'DeviationFromAvg_s'

def load_and_select():
    df = pd.read_csv(CSV)
    # Ensure Rain is binary (preprocess script should have done this, but double-check)
    if 'Rain' in df.columns and df['Rain'].dtype != np.int64 and df['Rain'].dtype != np.int32:
        df['Rain'] = df['Rain'].apply(lambda v: 1 if str(v).strip().lower().find('rain') != -1 else 0).astype(int)
    # merge clusters if separate file exists
    clusters = ART / 'premodel_clusters.csv'
    if clusters.exists():
        cdf = pd.read_csv(clusters)
        # join on BroadcastName/DriverNumber/Season/Round if available — try simple join on index-like columns
        # prefer a left-join on a subset if the merge column exists
        if 'weather_tire_cluster' not in df.columns and 'weather_tire_cluster' in cdf.columns:
            # try to align by row order if lengths match
            if len(cdf) == len(df):
                df['weather_tire_cluster'] = cdf['weather_tire_cluster'].values
            else:
                # fallback: attempt to join on ['DriverNumber','Season','Round'] if present
                join_cols = [c for c in ['DriverNumber','Season','Round'] if c in df.columns and c in cdf.columns]
                if join_cols:
                    df = df.merge(cdf[['weather_tire_cluster'] + join_cols], on=join_cols, how='left')
    # select features
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required features in CSV: {missing}")
    X = df[FEATURES].copy()
    y = df[TARGET].copy() if TARGET in df.columns else None
    return X, y, df
    return X, y, df

def train_and_save(X, y):
    # Ensure target exists
    if y is None:
        raise RuntimeError(f"Target column '{TARGET}' not found in CSV; cannot train")

    # drop rows with missing target values
    mask = ~y.isna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # simple random split (temporal split could be added if Season available)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
    dval = xgb.DMatrix(X_val.values, label=y_val.values)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'auto',
        'seed': 42,
        'verbosity': 1,
    }
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round=200, evals=evallist, early_stopping_rounds=20)

    # save model using joblib wrapper for convenience
    joblib.dump(bst, ART / 'xgb_minimal.joblib')
    with open(ART / 'columns_minimal.json', 'w') as f:
        json.dump(FEATURES, f)
    return bst, X_val

def compute_shap(bst, X_val):
    try:
        expl = shap.TreeExplainer(bst)
        shap_vals = expl.shap_values(X_val)
    except Exception:
        expl = shap.KernelExplainer(lambda x: bst.predict(x), shap.sample(X_val, 100))
        shap_vals = expl.shap_values(X_val)

    # summary
    mean_abs = np.abs(shap_vals).mean(axis=0)
    summary = pd.DataFrame({'feature': X_val.columns, 'mean_abs_shap': mean_abs})
    summary = summary.sort_values('mean_abs_shap', ascending=False)
    summary.to_csv(ART / 'shap_minimal_summary.csv', index=False)

    # plots
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_vals, X_val, show=False)
    plt.tight_layout()
    plt.savefig(ART / 'shap_minimal_beeswarm.png')

    # Rain dependence plot
    try:
        shap.dependence_plot('Rain', shap_vals, X_val, show=False)
        plt.tight_layout()
        plt.savefig(ART / 'shap_minimal_rain_dependence.png')
    except Exception:
        pass

def main():
    X, y, df = load_and_select()
    bst, X_val = train_and_save(X, y)
    compute_shap(bst, X_val)
    print('Retrain complete. Artifacts written to artifacts/')

if __name__ == '__main__':
    main()
