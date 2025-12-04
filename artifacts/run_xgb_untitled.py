import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_squared_error

p = Path('premodeldatav1.csv')
df = pd.read_csv(p)
features = ['GridPosition', 'AirTemp_C', 'TrackTemp_C', 'Humidity_%', 'Pressure_hPa', 'WindSpeed_mps', 'WindDirection_deg', 'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET', 'races_prior_this_season']
target = 'DeviationFromAvg_s'
df = df[~df[target].isna()].copy()
train = df[df.get('Season') == 2023].copy()
val = df[df.get('Season') == 2024].copy()
test = df[df.get('Season') == 2025].copy()
X_train = train[features].copy()
X_val = val[features].copy()
X_test = test[features].copy()
y_train = train[target]
y_val = val[target]
y_test = test[target]
for col in features:
  med = X_train[col].median()
  X_train[col] = X_train[col].fillna(med)
  X_val[col] = X_val[col].fillna(med)
  X_test[col] = X_test[col].fillna(med)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {'objective':'reg:squarederror','eval_metric':'rmse','learning_rate':0.05,'max_depth':6,'seed':42}
bst = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dtrain,'train'),(dval,'val')], early_stopping_rounds=50, verbose_eval=50)
Path('artifacts').mkdir(exist_ok=True)
bst.save_model('artifacts/xgb_model_untitled_external.json')
pd.DataFrame({'y_true': y_val.reset_index(drop=True), 'y_pred': bst.predict(dval)}).to_csv('artifacts/val_predictions_xgb_untitled_external.csv', index=False)
pd.DataFrame({'y_true': y_test.reset_index(drop=True), 'y_pred': bst.predict(dtest)}).to_csv('artifacts/test_predictions_xgb_untitled_external.csv', index=False)
# attempt to save SHAP-like contributions (XGBoost pred_contribs)
try:
  shap_val = bst.predict(dval, pred_contribs=True)
  shap_test = bst.predict(dtest, pred_contribs=True)
  cols = features + ['bias']
  import numpy as _np
  pd.DataFrame(shap_val, columns=cols).to_csv('artifacts/shap_val_xgb_untitled_external.csv', index=False)
  pd.DataFrame(shap_test, columns=cols).to_csv('artifacts/shap_test_xgb_untitled_external.csv', index=False)
  pd.Series(_np.abs(shap_val).mean(axis=0), index=cols).sort_values(ascending=False).to_csv('artifacts/shap_summary_val_xgb_untitled_external.csv')
  print('Saved SHAP-like pred_contribs to artifacts/')
except Exception as _e:
  print('Could not compute pred_contribs in external script:', _e)

print('External xgboost run complete')
