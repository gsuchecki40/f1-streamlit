#!/usr/bin/env python3
"""External randomized tuning script for XGBoost.
Saves best estimator (joblib) and validation predictions to artifacts/.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging

log = logging.getLogger('tune')
logging.basicConfig(level=logging.INFO)

try:
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint
except Exception as e:
    log.error('xgboost/scipy not available in this environment: %s', e)
    raise

P = Path('premodeldatav1.csv')
df = pd.read_csv(P)
features = ['GridPosition','AirTemp_C','TrackTemp_C','Humidity_%','Pressure_hPa','WindSpeed_mps','WindDirection_deg','SOFT','MEDIUM','HARD','INTERMEDIATE','WET','races_prior_this_season']
target = 'DeviationFromAvg_s'

df = df[~df[target].isna()].copy()
train = df[df.get('Season') == 2023].copy()
val = df[df.get('Season') == 2024].copy()

X_train = train[features].copy()
X_val = val[features].copy()
y_train = train[target]
y_val = val[target]

for col in features:
    med = X_train[col].median()
    X_train[col] = X_train[col].fillna(med)
    X_val[col] = X_val[col].fillna(med)

param_dist = {
    'max_depth': randint(3,10),
    'learning_rate': np.linspace(0.01,0.2,20),
    'n_estimators': randint(50,500),
    'subsample': np.linspace(0.5,1.0,6),
    'colsample_bytree': np.linspace(0.5,1.0,6),
    'reg_alpha': np.linspace(0,1.0,11),
    'reg_lambda': np.linspace(0,1.0,11)
}

model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0)
rs = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, scoring='neg_root_mean_squared_error', cv=3, n_jobs=1, verbose=2, random_state=42)
log.info('Starting randomized search...')
rs.fit(X_train, y_train)
log.info('Best params: %s', rs.best_params_)
best = rs.best_estimator_
art = Path('artifacts')
art.mkdir(exist_ok=True)
joblib.dump(best, art / 'xgb_best_external.joblib')
pd.DataFrame({'y_true': y_val.reset_index(drop=True), 'y_pred': best.predict(X_val)}).to_csv(art / 'val_predictions_xgb_tuned_external.csv', index=False)
log.info('Saved best model and validation predictions to artifacts/')
