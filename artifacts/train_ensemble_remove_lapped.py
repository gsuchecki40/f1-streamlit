"""
Train a 5-fold OOF XGBoost ensemble after removing lapped drivers.
Saves fold models, OOF preds, averaged val/test preds, linear calibration, SHAP summary, and metrics.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import math

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


def load_and_filter():
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

    # Split seasons
    train = df[df.get('Season') == 2023].copy()
    val = df[df.get('Season') == 2024].copy()
    test = df[df.get('Season') == 2025].copy()

    # remove lapped drivers
    if 'Status' in train.columns:
        train = train[train['Status'] != 'Lapped']
        val = val[val['Status'] != 'Lapped']
        test = test[test['Status'] != 'Lapped']
    else:
        if 'Status__Lapped' in train.columns:
            train = train[train['Status__Lapped'] != 1]
            val = val[val['Status__Lapped'] != 1]
            test = test[test['Status__Lapped'] != 1]

    # impute medians
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

    X_train = train[FEATURES].reset_index(drop=True)
    y_train = train[TARGET].reset_index(drop=True)
    X_val = val[FEATURES].reset_index(drop=True)
    y_val = val[TARGET].reset_index(drop=True)
    X_test = test[FEATURES].reset_index(drop=True)
    y_test = test[TARGET].reset_index(drop=True)
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_best_trial_params_and_rounds():
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
            rounds = int(best.get('best_rounds') or 100)
            return params, rounds
    # fallback
    return {'objective':'reg:squarederror','eval_metric':'rmse','eta':0.01,'max_depth':3,'subsample':0.8,'colsample_bytree':0.7,'reg_lambda':1.0,'reg_alpha':0.1,'gamma':0.0,'seed':42}, 100


def train_oof_ensemble(n_splits=5):
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_filter()
    params, rounds = get_best_trial_params_and_rounds()
    print('Ensemble training: rows', len(X_train), 'folds', n_splits, 'rounds', rounds)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    fold_models_dir = ART / 'ensemble_fold_models_remove_lapped'
    fold_models_dir.mkdir(exist_ok=True)

    val_fold_preds = []
    test_fold_preds = []

    for i, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        Xtr, ytr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        Xva, yva = X_train.iloc[va_idx], y_train.iloc[va_idx]
        dtr = xgb.DMatrix(Xtr.values, label=ytr.values)
        dva = xgb.DMatrix(Xva.values, label=yva.values)
        dval = xgb.DMatrix(X_val.values)
        dtest = xgb.DMatrix(X_test.values)

        bst = xgb.train(params, dtr, num_boost_round=rounds)
        # save booster
        joblib.dump(bst, fold_models_dir / f'fold_{i+1}.joblib')

        # preds
        oof_preds[va_idx] = bst.predict(dva)
        val_fold_preds.append(bst.predict(dval))
        test_fold_preds.append(bst.predict(dtest))

    # save OOF
    pd.DataFrame({'y_true': y_train, 'y_pred': oof_preds}).to_csv(ART / 'oof_preds_remove_lapped.csv', index=False)

    # average val/test preds
    val_avg = np.vstack(val_fold_preds).mean(axis=0)
    test_avg = np.vstack(test_fold_preds).mean(axis=0)
    pd.DataFrame({'y_true': y_val, 'y_pred': val_avg}).to_csv(ART / 'val_avg_preds_remove_lapped.csv', index=False)
    pd.DataFrame({'y_true': y_test, 'y_pred': test_avg}).to_csv(ART / 'test_avg_preds_remove_lapped.csv', index=False)

    # fit linear calibration on val
    lr = LinearRegression()
    lr.fit(val_avg.reshape(-1,1), y_val.values)
    joblib.dump(lr, ART / 'linear_calibration_remove_lapped.joblib')
    test_calibrated = lr.predict(test_avg.reshape(-1,1))
    pd.DataFrame({'y_true': y_test, 'y_pred': test_calibrated}).to_csv(ART / 'test_avg_preds_remove_lapped_calibrated.csv', index=False)

    # metrics
    def metrics(y_true, y_pred):
        return {
            'rmse': math.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'n': len(y_true)
        }

    import math
    m_oof = metrics(y_train.values, oof_preds)
    m_val = metrics(y_val.values, val_avg)
    m_test = metrics(y_test.values, test_avg)
    m_test_cal = metrics(y_test.values, test_calibrated)

    pd.DataFrame([m_oof]).to_csv(ART / 'metrics_remove_lapped_oof.csv', index=False)
    pd.DataFrame([m_val]).to_csv(ART / 'metrics_remove_lapped_val_ensemble.csv', index=False)
    pd.DataFrame([m_test]).to_csv(ART / 'metrics_remove_lapped_test_ensemble.csv', index=False)
    pd.DataFrame([m_test_cal]).to_csv(ART / 'metrics_remove_lapped_test_ensemble_calibrated.csv', index=False)

    # SHAP: compute mean abs shap across folds on training set
    try:
        shap_vals_accum = None
        for f in (fold_models_dir).iterdir():
            if f.suffix in ('.joblib', '.pkl'):
                bst = joblib.load(f)
                expl = shap.TreeExplainer(bst)
                sv = expl.shap_values(X_train)
                if shap_vals_accum is None:
                    shap_vals_accum = np.abs(sv)
                else:
                    shap_vals_accum += np.abs(sv)
        mean_abs = shap_vals_accum.mean(axis=0) / float(n_splits)
        pd.DataFrame({'feature': FEATURES, 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False).to_csv(ART / 'shap_remove_lapped_ensemble_summary.csv', index=False)
    except Exception as e:
        print('SHAP ensemble failed:', e)

    print('Ensemble (remove_lapped) complete. Metrics saved to artifacts/')


if __name__ == '__main__':
    train_oof_ensemble()
