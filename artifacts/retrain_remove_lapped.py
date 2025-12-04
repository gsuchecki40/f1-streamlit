"""
Train a quick model excluding lapped drivers and compare metrics to baseline.
Saves artifacts: xgb_remove_lapped.joblib, val/test preds and metrics, shap summary.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
ART = BASE / 'artifacts'
CSV = BASE / 'premodeldatav1.csv'
ART.mkdir(exist_ok=True)

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
    # merge clusters if missing
    if 'weather_tire_cluster' not in df.columns:
        cpath = ART / 'premodel_clusters.csv'
        if cpath.exists():
            cdf = pd.read_csv(cpath)
            if len(cdf) == len(df) and 'weather_tire_cluster' in cdf.columns:
                df['weather_tire_cluster'] = cdf['weather_tire_cluster'].values
    # filter season splits
    train = df[df.get('Season') == 2023].copy()
    val = df[df.get('Season') == 2024].copy()
    test = df[df.get('Season') == 2025].copy()
    # drop lapped drivers: check categorical or columns
    if 'Status' in train.columns:
        train = train[train['Status'] != 'Lapped']
        val = val[val['Status'] != 'Lapped']
        test = test[test['Status'] != 'Lapped']
    else:
        # one-hot encoded patterns
        for col in ['Status__Lapped','Status__Lapped.0']:
            if col in train.columns:
                train = train[train[col] != 1]
                val = val[val[col] != 1]
                test = test[test[col] != 1]
                break
    # basic impute from train medians
    for col in FEATURES:
        if col in train.columns:
            med = train[col].median()
            train[col] = train[col].fillna(med)
            val[col] = val[col].fillna(med)
            test[col] = test[col].fillna(med)
    # drop missing target
    train = train[~train[TARGET].isna()]
    val = val[~val[TARGET].isna()]
    test = test[~test[TARGET].isna()]
    return train[FEATURES], train[TARGET], val[FEATURES], val[TARGET], test[FEATURES], test[TARGET]


def get_best_params():
    trials = ART / 'randomized_cv_trials.csv'
    if trials.exists():
        df = pd.read_csv(trials)
        df = df.dropna(subset=['cv_rmse'])
        if len(df):
            best = df.loc[df['cv_rmse'].idxmin()].to_dict()
            params = {
                'objective':'reg:squarederror',
                'eval_metric':'rmse',
                'eta': float(best.get('eta', 0.01)),
                'max_depth': int(best.get('max_depth', 3)),
                'subsample': float(best.get('subsample', 0.8)),
                'colsample_bytree': float(best.get('colsample_bytree', 0.7)),
                'reg_lambda': float(best.get('reg_lambda', 1.0)),
                'reg_alpha': float(best.get('reg_alpha', 0.1)),
                'gamma': float(best.get('gamma', 0.0)),
                'seed': 42
            }
            return params
    # fallback conservative
    return {'objective':'reg:squarederror','eval_metric':'rmse','eta':0.01,'max_depth':3,'subsample':0.8,'colsample_bytree':0.7,'reg_lambda':1.0,'reg_alpha':0.1,'gamma':0.0,'seed':42}


def train_and_save():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    params = get_best_params()
    dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
    print('Training on', len(X_train), 'rows after dropping lapped drivers')
    bst = xgb.train(params, dtrain, num_boost_round=200)
    joblib.dump(bst, ART / 'xgb_remove_lapped.joblib')
    # preds
    pd.DataFrame({'y_true': y_val.reset_index(drop=True), 'y_pred': bst.predict(xgb.DMatrix(X_val.values))}).to_csv(ART / 'val_predictions_xgb_remove_lapped.csv', index=False)
    pd.DataFrame({'y_true': y_test.reset_index(drop=True), 'y_pred': bst.predict(xgb.DMatrix(X_test.values))}).to_csv(ART / 'test_predictions_xgb_remove_lapped.csv', index=False)
    # metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import math
    yv = y_val.values
    pv = bst.predict(xgb.DMatrix(X_val.values))
    m = {'rmse': math.sqrt(mean_squared_error(yv,pv)), 'mae': mean_absolute_error(yv,pv), 'r2': r2_score(yv,pv), 'n': len(yv)}
    pd.DataFrame([m]).to_csv(ART / 'metrics_remove_lapped_val.csv', index=False)
    yt = y_test.values
    pt = bst.predict(xgb.DMatrix(X_test.values))
    mt = {'rmse': math.sqrt(mean_squared_error(yt,pt)), 'mae': mean_absolute_error(yt,pt), 'r2': r2_score(yt,pt), 'n': len(yt)}
    pd.DataFrame([mt]).to_csv(ART / 'metrics_remove_lapped_test.csv', index=False)
    # SHAP
    try:
        expl = shap.TreeExplainer(bst)
        shap_vals = expl.shap_values(X_train)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        pd.DataFrame({'feature': FEATURES, 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False).to_csv(ART / 'shap_remove_lapped_summary.csv', index=False)
    except Exception as e:
        print('SHAP failed:', e)
    print('Saved artifacts for remove_lapped model')

if __name__ == '__main__':
    train_and_save()
