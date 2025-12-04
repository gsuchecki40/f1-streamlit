import pandas as pd
import xgboost as xgb
from pathlib import Path
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
bst.save_model('artifacts/xgb_model.json')
pd.DataFrame({'y_true': y_val, 'y_pred': bst.predict(dval)}).to_csv('artifacts/val_predictions_xgb_external.csv', index=False)
pd.DataFrame({'y_true': y_test, 'y_pred': bst.predict(dtest)}).to_csv('artifacts/test_predictions_xgb_external.csv', index=False)
print('External xgboost run complete')
