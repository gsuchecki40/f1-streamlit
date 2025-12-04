"""Retrain using xgboost.cv to pick boosting rounds (5-fold) then train final model.

Params:
- learning_rate=0.05
- max_depth=4
- 5-fold CV

Outputs to artifacts/: xgb_minimal_cv.joblib, columns_minimal.json (reused), shap_minimal_cv_summary.csv and plots
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

BASE = Path(__file__).resolve().parent.parent
ART = BASE / 'artifacts'
CSV = BASE / 'premodeldatav1.csv'

FEATURES = [
    'GridPosition','AvgQualiTime','weather_tire_cluster',
    'SOFT','MEDIUM','HARD','INTERMEDIATE','WET',
    'races_prior_this_season','Rain'
]
TARGET = 'DeviationFromAvg_s'

def load():
    df = pd.read_csv(CSV)
    # ensure Rain binary
    if 'Rain' in df.columns and not np.issubdtype(df['Rain'].dtype, np.integer):
        df['Rain'] = df['Rain'].apply(lambda v: 1 if 'rain' in str(v).lower() else 0).astype(int)

    # if weather_tire_cluster not in main DF, try to pull from artifacts/premodel_clusters.csv
    if 'weather_tire_cluster' not in df.columns:
        clusters_path = ART / 'premodel_clusters.csv'
        if clusters_path.exists():
            cdf = pd.read_csv(clusters_path)
            if 'weather_tire_cluster' in cdf.columns:
                # align by row order if lengths match
                if len(cdf) == len(df):
                    df['weather_tire_cluster'] = cdf['weather_tire_cluster'].values
                else:
                    join_cols = [c for c in ['DriverNumber','Season','Round'] if c in df.columns and c in cdf.columns]
                    if join_cols:
                        df = df.merge(cdf[['weather_tire_cluster'] + join_cols], on=join_cols, how='left')

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise RuntimeError(f"Missing features: {missing}")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    mask = ~y.isna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    return X, y

def cv_train(X, y):
    params = {
        'objective':'reg:squarederror',
        'eta':0.05,
        'max_depth':4,
        'seed':42,
        'eval_metric':'rmse'
    }
    dtrain = xgb.DMatrix(X.values, label=y.values)
    cvres = xgb.cv(params, dtrain, num_boost_round=1000, nfold=5,
                   early_stopping_rounds=30, metrics='rmse', seed=42, as_pandas=True, verbose_eval=20)
    best_boost_rounds = len(cvres)
    print(f"CV completed. Best rounds: {best_boost_rounds}")
    return params, best_boost_rounds

def train_final(params, rounds, X, y):
    dtrain = xgb.DMatrix(X.values, label=y.values)
    bst = xgb.train(params, dtrain, num_boost_round=rounds)
    joblib.dump(bst, ART / 'xgb_minimal_cv.joblib')
    return bst

def compute_shap(bst, X):
    try:
        expl = shap.TreeExplainer(bst)
        shap_vals = expl.shap_values(X)
    except Exception:
        expl = shap.KernelExplainer(lambda x: bst.predict(x), shap.sample(X, 100))
        shap_vals = expl.shap_values(X)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    summary = pd.DataFrame({'feature': X.columns, 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False)
    summary.to_csv(ART / 'shap_minimal_cv_summary.csv', index=False)
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_vals, X, show=False)
    plt.tight_layout()
    plt.savefig(ART / 'shap_minimal_cv_beeswarm.png')
    try:
        shap.dependence_plot('Rain', shap_vals, X, show=False)
        plt.tight_layout()
        plt.savefig(ART / 'shap_minimal_cv_rain_dependence.png')
    except Exception:
        pass

def main():
    X,y = load()
    params, rounds = cv_train(X,y)
    bst = train_final(params, rounds, X, y)
    compute_shap(bst, X)
    print('CV retrain complete; artifacts saved in artifacts/')

if __name__ == '__main__':
    main()
