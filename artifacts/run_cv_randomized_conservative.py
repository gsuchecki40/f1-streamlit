"""Randomized conservative CV tuning (Option A, wider than existing script).

This script:
- Loads the minimal-feature dataset used previously.
- Runs randomized trials: for each trial pick params from conservative ranges and run xgboost.cv (5-fold) to find best rounds.
- Records CV rmse and best rounds for each trial.
- Selects the best trial (lowest mean CV rmse), trains a final model on the full training set with chosen params and rounds.
- Saves final model, validation predictions, test predictions, SHAP summary and plots, and a trials CSV.
"""
import json
import random
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

random.seed(42)
BASE = Path(__file__).resolve().parent.parent
ART = BASE / 'artifacts'
CSV = BASE / 'premodeldatav1.csv'
ART.mkdir(exist_ok=True)

# Features consistent with retrain scripts
FEATURES = [
    'GridPosition','AvgQualiTime','weather_tire_cluster',
    'SOFT','MEDIUM','HARD','INTERMEDIATE','WET',
    'races_prior_this_season','Rain','PointsProp'
]
TARGET = 'DeviationFromAvg_s'

def load_data():
    df = pd.read_csv(CSV)
    # ensure Rain binary
    if 'Rain' in df.columns and not np.issubdtype(df['Rain'].dtype, np.integer):
        df['Rain'] = df['Rain'].apply(lambda v: 1 if 'rain' in str(v).lower() else 0).astype(int)

    # ensure PointsProp exists
    if 'PointsProp' not in df.columns:
        raise RuntimeError('PointsProp missing; run add_pointsprop.py first')

    # merge clusters if missing
    if 'weather_tire_cluster' not in df.columns:
        cpath = ART / 'premodel_clusters.csv'
        if cpath.exists():
            cdf = pd.read_csv(cpath)
            if len(cdf) == len(df) and 'weather_tire_cluster' in cdf.columns:
                df['weather_tire_cluster'] = cdf['weather_tire_cluster'].values

    # filter by seasons used previously (train:2023, val:2024, test:2025)
    train = df[df.get('Season') == 2023].copy()
    val = df[df.get('Season') == 2024].copy()
    test = df[df.get('Season') == 2025].copy()

    # basic impute from train medians
    for col in FEATURES:
        if col in train.columns:
            med = train[col].median()
            train[col] = train[col].fillna(med)
            val[col] = val[col].fillna(med)
            test[col] = test[col].fillna(med)

    # drop rows with missing target
    train = train[~train[TARGET].isna()]
    val = val[~val[TARGET].isna()]
    test = test[~test[TARGET].isna()]

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_val = val[FEATURES]
    y_val = val[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]
    return X_train, y_train, X_val, y_val, X_test, y_test

def run_randomized_cv(n_trials=25):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    dtrain_full = xgb.DMatrix(X_train.values, label=y_train.values)

    trials = []
    best = {'cv_rmse': 1e9}

    # conservative search space
    space = {
        'eta': [0.01, 0.03, 0.05, 0.08],
        'max_depth': [3,4,5],
        'subsample': [0.6,0.7,0.8,0.9],
        'colsample_bytree': [0.6,0.7,0.8,0.9],
        'reg_lambda': [0.5,1.0,2.0,5.0],
        'reg_alpha': [0.0,0.1,0.5,1.0],
        'gamma': [0.0,0.1,0.5,1.0]
    }

    for i in range(n_trials):
        params = {
            'objective':'reg:squarederror',
            'eval_metric':'rmse',
            'eta': random.choice(space['eta']),
            'max_depth': random.choice(space['max_depth']),
            'subsample': random.choice(space['subsample']),
            'colsample_bytree': random.choice(space['colsample_bytree']),
            'reg_lambda': random.choice(space['reg_lambda']),
            'reg_alpha': random.choice(space['reg_alpha']),
            'gamma': random.choice(space['gamma']),
            'seed': 42
        }
        # run xgboost.cv for this param set
        print(f"Trial {i+1}/{n_trials}: {params}")
        try:
            cvres = xgb.cv(params, dtrain_full, num_boost_round=2000, nfold=5,
                           early_stopping_rounds=50, metrics='rmse', seed=42, as_pandas=True, verbose_eval=False)
            best_rounds = len(cvres)
            cv_rmse = float(cvres['test-rmse-mean'].iloc[-1])
            print(f"  cv_rmse={cv_rmse:.4f}, rounds={best_rounds}")
        except Exception as e:
            print('  cv failed for params:', e)
            cv_rmse = np.nan
            best_rounds = None

        trials.append({'trial': i+1, **params, 'cv_rmse': cv_rmse, 'best_rounds': best_rounds})

        # keep best based on cv_rmse
        if not np.isnan(cv_rmse) and cv_rmse < best['cv_rmse']:
            best = {'cv_rmse': cv_rmse, 'params': params, 'best_rounds': best_rounds}

    trials_df = pd.DataFrame(trials)
    trials_df.to_csv(ART / 'randomized_cv_trials.csv', index=False)
    print('Trials saved to', ART / 'randomized_cv_trials.csv')

    if 'params' not in best:
        raise RuntimeError('No successful CV trials')

    # train final on full training data with chosen params and rounds
    print('Best trial:', best)
    bst = xgb.train(best['params'], xgb.DMatrix(X_train.values, label=y_train.values), num_boost_round=best['best_rounds'])
    joblib.dump(bst, ART / 'xgb_minimal_cv_randomized.joblib')

    # predictions
    dval = xgb.DMatrix(X_val.values)
    dtest = xgb.DMatrix(X_test.values)
    pd.DataFrame({'y_true': y_val.reset_index(drop=True), 'y_pred': bst.predict(dval)}).to_csv(ART / 'val_predictions_xgb_minimal_cv_randomized.csv', index=False)
    pd.DataFrame({'y_true': y_test.reset_index(drop=True), 'y_pred': bst.predict(dtest)}).to_csv(ART / 'test_predictions_xgb_minimal_cv_randomized.csv', index=False)

    # SHAP
    try:
        expl = shap.TreeExplainer(bst)
        shap_vals = expl.shap_values(X_train)
    except Exception:
        # fallback to KernelExplainer on a sample
        expl = shap.KernelExplainer(lambda x: bst.predict(xgb.DMatrix(x)), shap.sample(X_train, 100))
        shap_vals = expl.shap_values(X_train)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    summary = pd.DataFrame({'feature': FEATURES, 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False)
    summary.to_csv(ART / 'shap_minimal_cv_randomized_summary.csv', index=False)

    plt.figure(figsize=(8,6))
    try:
        shap.summary_plot(shap_vals, X_train, show=False)
        plt.tight_layout()
        plt.savefig(ART / 'shap_minimal_cv_randomized_beeswarm.png')
    except Exception as e:
        print('Could not render shap beeswarm:', e)

    print('Randomized CV training complete. Artifacts saved to', ART)

def main():
    run_randomized_cv(n_trials=30)

if __name__ == '__main__':
    main()
