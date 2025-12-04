
# XGBoost — direct run (requires xgboost in this kernel)
import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import mean_squared_error
try:
  import xgboost as xgb
  print('xgboost version in-kernel:', xgb.__version__)
except Exception as e:
  print('xgboost not available in-kernel:', e)

p = Path('premodeldatav1.csv')
df = pd.read_csv(p)

features = ['GridPosition','AirTemp_C','TrackTemp_C','Humidity_%','Pressure_hPa','WindSpeed_mps','WindDirection_deg','SOFT','MEDIUM','HARD','INTERMEDIATE','WET','races_prior_this_season']
target = 'DeviationFromAvg_s'

df = df[~df[target].isna()].copy()
train = df[df.get('Season') == 2023].copy()
val = df[df.get('Season') == 2024].copy()
test = df[df.get('Season') == 2025].copy()

if len(train)==0 or len(val)==0 or len(test)==0:
  raise RuntimeError('Seasonal split incomplete; ensure seasons 2023/2024/2025 are present')

X_train = train[features].copy()
X_val = val[features].copy()
X_test = test[features].copy()
y_train = train[target]
y_val = val[target]
y_test = test[target]

# impute medians
for col in features:
  med = X_train[col].median()
  X_train[col] = X_train[col].fillna(med)
  X_val[col] = X_val[col].fillna(med)
  X_test[col] = X_test[col].fillna(med)

if 'xgb' in globals() and hasattr(xgb, 'DMatrix'):
  dtrain = xgb.DMatrix(X_train, label=y_train)
  dval = xgb.DMatrix(X_val, label=y_val)
  params = {'objective':'reg:squarederror','eval_metric':'rmse','learning_rate':0.05,'max_depth':6,'seed':42}
  bst = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dtrain,'train'),(dval,'val')], early_stopping_rounds=50, verbose_eval=50)
  pred_val = bst.predict(xgb.DMatrix(X_val))
  pred_test = bst.predict(xgb.DMatrix(X_test))
  Path('artifacts').mkdir(exist_ok=True)
  bst.save_model('artifacts/xgb_model_untitled_direct.json')
  joblib.dump({'features':features}, 'artifacts/xgb_feature_list_untitled_direct.joblib')
  pd.DataFrame({'y_true': y_val.reset_index(drop=True), 'y_pred': pred_val}).to_csv('artifacts/val_predictions_xgb_untitled_direct.csv', index=False)
  pd.DataFrame({'y_true': y_test.reset_index(drop=True), 'y_pred': pred_test}).to_csv('artifacts/test_predictions_xgb_untitled_direct.csv', index=False)
  print('Direct xgboost training complete — artifacts saved to artifacts/')
else:
  print('xgboost not available in this kernel; use the external helper below to run with conda python')




# External-run helper: write a script to artifacts/run_xgb_untitled.py and run it with a detected conda python
import os
import subprocess
from pathlib import Path

conda_python = None
if os.environ.get('CONDA_PREFIX'):
  maybe = os.path.join(os.environ['CONDA_PREFIX'], 'bin', 'python')
  if os.path.exists(maybe):
    conda_python = maybe
for candidate in ['/opt/anaconda3/bin/python', '/usr/local/anaconda3/bin/python', '/opt/homebrew/bin/python', '/usr/bin/python3']:
  if conda_python is None and os.path.exists(candidate):
    conda_python = candidate

script_path = Path('artifacts/run_xgb_untitled.py')
script_path.parent.mkdir(exist_ok=True)

script = '''import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_squared_error

p = Path('premodeldatav1.csv')
df = pd.read_csv(p)
features = %s
target = '%s'
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
'''

script = script % (repr(['GridPosition','AirTemp_C','TrackTemp_C','Humidity_%','Pressure_hPa','WindSpeed_mps','WindDirection_deg','SOFT','MEDIUM','HARD','INTERMEDIATE','WET','races_prior_this_season']), 'DeviationFromAvg_s')
script_path.write_text(script)

if conda_python and os.path.exists(conda_python):
  print('Running external script with', conda_python)
  rc = subprocess.call([conda_python, str(script_path)])
  print('external script exit code', rc)
else:
  print('No conda/python detected; run the script manually with your conda python:')
  print('/path/to/conda/python', script_path)




# Results & SHAP loader — compute metrics and summarize SHAP/pred_contribs
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

art = Path('artifacts')
if not art.exists():
  raise RuntimeError('No artifacts/ directory found; run the training first')

# load data
df = pd.read_csv('premodeldatav1.csv')
df = df[~df['DeviationFromAvg_s'].isna()].copy()
train = df[df.get('Season') == 2023].copy()
val = df[df.get('Season') == 2024].copy()
test = df[df.get('Season') == 2025].copy()
features = ['GridPosition','AirTemp_C','TrackTemp_C','Humidity_%','Pressure_hPa','WindSpeed_mps','WindDirection_deg','SOFT','MEDIUM','HARD','INTERMEDIATE','WET','races_prior_this_season']

for col in features:
  med = train[col].median()
  val[col] = val[col].fillna(med)
  test[col] = test[col].fillna(med)

# try to load in-kernel booster first
rmse_val = rmse_test = None
mae_val = mae_test = None
r2_val = r2_test = None

model_loaded = False
try:
  import xgboost as xgb
  # if a booster object exists in the namespace, use it
  if 'bst' in globals():
    booster = globals()['bst']
    dval = xgb.DMatrix(val[features])
    dtest = xgb.DMatrix(test[features])
    pred_val = booster.predict(dval)
    pred_test = booster.predict(dtest)
    model_loaded = True
  else:
    # try to load saved booster
    for p in ['artifacts/xgb_model_untitled_direct.json','artifacts/xgb_model_untitled_external.json','artifacts/xgb_model.json']:
      if Path(p).exists():
        booster = xgb.Booster()
        booster.load_model(p)
        dval = xgb.DMatrix(val[features])
        dtest = xgb.DMatrix(test[features])
        pred_val = booster.predict(dval)
        pred_test = booster.predict(dtest)
        model_loaded = True
        print('Loaded booster from', p)
        break
except Exception as e:
  print('xgboost not usable in-kernel for metrics loading:', e)

if not model_loaded:
  # try to read prediction files written by external script
  pv = art / 'val_predictions_xgb_untitled_external.csv'
  pt = art / 'test_predictions_xgb_untitled_external.csv'
  if pv.exists() and pt.exists():
    pred_val = pd.read_csv(pv)['y_pred'].values
    pred_test = pd.read_csv(pt)['y_pred'].values
    print('Loaded predictions from artifacts/')
  else:
    raise RuntimeError('No model or prediction artifacts found. Run the training first.')

# compute metrics
def compute_metrics(y_true, y_pred):
  return {
    'rmse': mean_squared_error(y_true, y_pred, squared=False),
    'mae': mean_absolute_error(y_true, y_pred),
    'r2': r2_score(y_true, y_pred)
  }

metrics_val = compute_metrics(val['DeviationFromAvg_s'].values, pred_val)
metrics_test = compute_metrics(test['DeviationFromAvg_s'].values, pred_test)

print('Validation metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}'.format(**metrics_val))
print('Test metrics:       RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}'.format(**metrics_test))

# save metrics to artifacts
pd.Series(metrics_val).to_json(art / 'metrics_val_xgb.json')
pd.Series(metrics_test).to_json(art / 'metrics_test_xgb.json')

# load SHAP-like pred_contribs if present
shap_candidates = [art / 'shap_val_xgb_untitled_external.csv', art / 'shap_test_xgb_untitled_external.csv', art / 'shap_val_xgb.csv']
if (art / 'shap_val_xgb_untitled_external.csv').exists():
  sv = pd.read_csv(art / 'shap_val_xgb_untitled_external.csv')
  st = pd.read_csv(art / 'shap_test_xgb_untitled_external.csv')
  print('Loaded SHAP-like pred_contribs from artifacts; saving brief summaries')
  # mean absolute contribution per feature
  def shap_summary(df_shap):
    return df_shap.abs().mean().sort_values(ascending=False)
  shap_summary(sv).to_csv(art / 'shap_summary_val_xgb.csv')
  shap_summary(st).to_csv(art / 'shap_summary_test_xgb.csv')
  print('Saved shap summaries to artifacts/')
else:
  print('No SHAP/pred_contrib files found in artifacts')

