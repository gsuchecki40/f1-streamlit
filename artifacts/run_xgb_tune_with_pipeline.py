#!/usr/bin/env python3
"""Train XGBoost using the saved preprocessing pipeline and produce SHAP values.

Inputs: artifacts/preprocessing_pipeline.joblib, artifacts/premodel_canonical.csv
Outputs: artifacts/xgb_best_with_pipeline.joblib, artifacts/val_predictions_with_pipeline.csv,
         artifacts/shap_val_with_pipeline.csv
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

try:
    import xgboost as xgb
    import shap
except Exception as e:
    raise RuntimeError('xgboost/shap required: %s' % e)

ART = Path('artifacts')
pipe_path = ART / 'preprocessing_pipeline.joblib'
csv_path = ART / 'premodel_canonical.csv'

if not pipe_path.exists():
    raise SystemExit('Missing pipeline: ' + str(pipe_path))
if not csv_path.exists():
    raise SystemExit('Missing canonical CSV: ' + str(csv_path))

pipe = joblib.load(pipe_path)
df = pd.read_csv(csv_path)
target = 'DeviationFromAvg_s'
df = df[~df[target].isna()].copy()

# seasonal split
train = df[df['Season'] == 2023].copy()
val = df[df['Season'] == 2024].copy()
test = df[df['Season'] == 2025].copy()

X_train = pipe.transform(train)
X_val = pipe.transform(val)
X_test = pipe.transform(test)

# get feature names
try:
    feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()
    feature_names = list(feature_names)
except Exception:
    feature_names = [f'f_{i}' for i in range(X_train.shape[1])]

X_train = pd.DataFrame(X_train, columns=feature_names)
X_val = pd.DataFrame(X_val, columns=feature_names)
X_test = pd.DataFrame(X_test, columns=feature_names)

y_train = train[target]
y_val = val[target]

param_dist = {
    'max_depth': randint(3,8),
    'learning_rate': np.linspace(0.01,0.2,20),
    'n_estimators': randint(50,400),
    'subsample': np.linspace(0.6,1.0,5),
    'colsample_bytree': np.linspace(0.6,1.0,5),
    'reg_alpha': np.linspace(0,1.0,11),
    'reg_lambda': np.linspace(0,1.0,11)
}

model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0)
rs = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, scoring='neg_root_mean_squared_error', cv=3, n_jobs=1, verbose=2, random_state=42)
print('Starting randomized search...')
rs.fit(X_train, y_train)
print('Best params:', rs.best_params_)
best = rs.best_estimator_

joblib.dump(best, ART / 'xgb_best_with_pipeline.joblib')
pd.DataFrame({'y_true': y_val.reset_index(drop=True), 'y_pred': best.predict(X_val)}).to_csv(ART / 'val_predictions_with_pipeline.csv', index=False)

# SHAP TreeExplainer on best model
print('Computing SHAP values on validation set (may be slow)...')
explainer = shap.TreeExplainer(best)
shap_vals = explainer.shap_values(X_val)
shap_df = pd.DataFrame(shap_vals, columns=feature_names)
shap_df.to_csv(ART / 'shap_val_with_pipeline.csv', index=False)

print('Saved model, predictions and SHAP values to artifacts/')
