"""Run diagnostics, train 5-fold OOF ensemble with conservative params, calibrate and save artifacts.

Outputs:
- artifacts/oof_preds.csv
- artifacts/val_avg_preds.csv
- artifacts/test_avg_preds.csv
- artifacts/ensemble_fold_models/joblib per fold
- artifacts/metrics_ensemble.csv
- artifacts/residuals_hist_ensemble.png
- artifacts/pred_vs_actual_ensemble.png
- artifacts/confusion_residuals_ensemble.png
- artifacts/shap_ensemble_summary.csv
"""
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import xgboost as xgb
import shap

BASE = Path(__file__).resolve().parent.parent
ART = BASE / 'artifacts'
CSV = BASE / 'premodeldatav1.csv'
ART.mkdir(exist_ok=True)

# Use FEATURES consistent with previous runs
FEATURES = [
    'GridPosition','AvgQualiTime','weather_tire_cluster',
    'SOFT','MEDIUM','HARD','INTERMEDIATE','WET',
    'races_prior_this_season','Rain','PointsProp'
]
TARGET = 'DeviationFromAvg_s'

# Best params from randomized CV (found earlier)
BEST_PARAMS = {'objective':'reg:squarederror','eval_metric':'rmse','eta':0.01,'max_depth':3,'subsample':0.8,'colsample_bytree':0.7,'reg_lambda':1.0,'reg_alpha':0.1,'gamma':0.0,'seed':42}
BEST_ROUNDS = 305

def load_splits():
    df = pd.read_csv(CSV)
    if 'Rain' in df.columns and not np.issubdtype(df['Rain'].dtype, np.integer):
        df['Rain'] = df['Rain'].apply(lambda v: 1 if 'rain' in str(v).lower() else 0).astype(int)
    # merge clusters if missing
    if 'weather_tire_cluster' not in df.columns:
        cpath = ART / 'premodel_clusters.csv'
        if cpath.exists():
            cdf = pd.read_csv(cpath)
            if len(cdf) == len(df) and 'weather_tire_cluster' in cdf.columns:
                df['weather_tire_cluster'] = cdf['weather_tire_cluster'].values

    train = df[df.get('Season') == 2023].copy()
    val = df[df.get('Season') == 2024].copy()
    test = df[df.get('Season') == 2025].copy()

    for col in FEATURES:
        if col in train.columns:
            med = train[col].median()
            train[col] = train[col].fillna(med)
            val[col] = val[col].fillna(med)
            test[col] = test[col].fillna(med)

    train = train[~train[TARGET].isna()]
    val = val[~val[TARGET].isna()]
    test = test[~test[TARGET].isna()]

    X_train = train[FEATURES].reset_index(drop=True)
    y_train = train[TARGET].reset_index(drop=True)
    X_val = val[FEATURES].reset_index(drop=True)
    y_val = val[TARGET].reset_index(drop=True)
    X_test = test[FEATURES].reset_index(drop=True)
    y_test = test[TARGET].reset_index(drop=True)
    return X_train,y_train,X_val,y_val,X_test,y_test

def compute_metrics(y, yp):
    mse = mean_squared_error(y, yp)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y, yp)
    r2 = r2_score(y, yp)
    return rmse, mae, r2

def diagnostics(y_true, y_pred, out_prefix):
    rmse, mae, r2 = compute_metrics(y_true, y_pred)
    dfm = pd.DataFrame([{'rmse':rmse,'mae':mae,'r2':r2,'n':len(y_true)}])
    dfm.to_csv(ART / f'metrics_{out_prefix}.csv', index=False)

    plt.figure(figsize=(6,4))
    sns.histplot((y_true-y_pred), bins=40, kde=True)
    plt.title('Residual histogram')
    plt.xlabel('y_true - y_pred')
    plt.tight_layout()
    plt.savefig(ART / f'residuals_hist_{out_prefix}.png')

    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    m = min(min(y_true), min(y_pred))
    M = max(max(y_true), max(y_pred))
    plt.plot([m,M],[m,M],'r--',linewidth=1)
    plt.xlabel('y_true')
    plt.ylabel('y_pred')
    plt.title('Predicted vs Actual')
    plt.tight_layout()
    plt.savefig(ART / f'pred_vs_actual_{out_prefix}.png')

    # confusion by tertiles
    q = pd.qcut(y_true, 3, labels=['low','med','high'])
    qp = pd.qcut(y_pred, 3, labels=['low','med','high'])
    conf = pd.crosstab(q, qp)
    conf.to_csv(ART / f'confusion_residuals_{out_prefix}.csv')
    plt.figure(figsize=(6,4))
    sns.heatmap(conf, annot=True, fmt='d', cmap='Blues')
    plt.title('True tertile vs Predicted tertile')
    plt.tight_layout()
    plt.savefig(ART / f'confusion_residuals_{out_prefix}.png')

def train_oof_ensemble(X,y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    fold_models = []
    shap_vals_accum = []

    fold = 0
    for tr_idx, val_idx in kf.split(X):
        fold += 1
        X_tr, X_va = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[val_idx]
        dtr = xgb.DMatrix(X_tr.values, label=y_tr.values)
        dva = xgb.DMatrix(X_va.values)
        bst = xgb.train(BEST_PARAMS, dtr, num_boost_round=BEST_ROUNDS)
        preds = bst.predict(dva)
        oof[val_idx] = preds
        # save model
        mpath = ART / 'ensemble_fold_models'
        mpath.mkdir(exist_ok=True)
        joblib.dump(bst, mpath / f'fold_{fold}.joblib')
        fold_models.append(mpath / f'fold_{fold}.joblib')
        # shap on fold training data sample (train side)
        try:
            expl = shap.TreeExplainer(bst)
            sv = expl.shap_values(X_tr)
            shap_vals_accum.append(sv)
        except Exception:
            pass

    return oof, fold_models, shap_vals_accum

def aggregate_fold_preds(fold_models, X):
    preds = []
    for p in fold_models:
        bst = joblib.load(p)
        d = xgb.DMatrix(X.values)
        preds.append(bst.predict(d))
    return np.mean(preds, axis=0)

def calibrate_linear(y_val, y_pred_val, y_test, y_pred_test):
    lr = LinearRegression()
    lr.fit(y_pred_val.reshape(-1,1), y_val)
    y_test_cal = lr.predict(y_pred_test.reshape(-1,1))
    return lr, y_test_cal

def main():
    X_train,y_train,X_val,y_val,X_test,y_test = load_splits()

    # Diagnostics for randomized-CV model (existing predictions)
    try:
        val_df = pd.read_csv(ART / 'val_predictions_xgb_minimal_cv_randomized.csv')
        test_df = pd.read_csv(ART / 'test_predictions_xgb_minimal_cv_randomized.csv')
        diagnostics(val_df['y_true'].values, val_df['y_pred'].values, 'randomized_cv')
        diagnostics(test_df['y_true'].values, test_df['y_pred'].values, 'randomized_cv_test')
    except Exception:
        print('Could not load existing randomized CV preds; skipping diagnostics for that model')

    # Train OOF ensemble on training set
    print('Training OOF ensemble...')
    oof, fold_models, shap_acc = train_oof_ensemble(X_train, y_train, n_splits=5)
    pd.DataFrame({'y_true': y_train, 'y_oof': oof}).to_csv(ART / 'oof_preds.csv', index=False)
    rmse_oof, mae_oof, r2_oof = compute_metrics(y_train.values, oof)
    pd.DataFrame([{'rmse':rmse_oof,'mae':mae_oof,'r2':r2_oof,'n':len(y_train)}]).to_csv(ART / 'metrics_oof.csv', index=False)

    # Average fold preds for val/test
    val_avg = aggregate_fold_preds(fold_models, X_val)
    test_avg = aggregate_fold_preds(fold_models, X_test)
    pd.DataFrame({'y_true': y_val, 'y_pred': val_avg}).to_csv(ART / 'val_avg_preds_ensemble.csv', index=False)
    pd.DataFrame({'y_true': y_test, 'y_pred': test_avg}).to_csv(ART / 'test_avg_preds_ensemble.csv', index=False)

    diagnostics(y_val.values, val_avg, 'ensemble_val')
    diagnostics(y_test.values, test_avg, 'ensemble_test')

    # Calibration
    lr, test_cal = calibrate_linear(y_val.values, val_avg, y_test.values, test_avg)
    pd.DataFrame({'y_true': y_test, 'y_pred': test_avg, 'y_pred_cal': test_cal}).to_csv(ART / 'test_avg_preds_ensemble_calibrated.csv', index=False)
    rmse_cal, mae_cal, r2_cal = compute_metrics(y_test.values, test_cal)
    pd.DataFrame([{'rmse':rmse_cal,'mae':mae_cal,'r2':r2_cal,'n':len(y_test)}]).to_csv(ART / 'metrics_ensemble_calibrated.csv', index=False)

    # aggregate shap across folds
    if shap_acc:
        shap_concat = np.concatenate(shap_acc, axis=0)
        mean_abs = np.abs(shap_concat).mean(axis=0)
        summary = pd.DataFrame({'feature': FEATURES, 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False)
        summary.to_csv(ART / 'shap_ensemble_summary.csv', index=False)

    print('All artifacts written to', ART)

if __name__ == '__main__':
    main()
