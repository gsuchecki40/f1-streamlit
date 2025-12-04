import random
import json
from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

random.seed(42)
art = Path('artifacts')
Path('artifacts').mkdir(exist_ok=True)

p = Path('premodeldatav1.csv')
df = pd.read_csv(p)
df = df[~df['DeviationFromAvg_s'].isna()].copy()
train = df[df.get('Season') == 2023].copy()
val = df[df.get('Season') == 2024].copy()
test = df[df.get('Season') == 2025].copy()
features = ['GridPosition','AirTemp_C','TrackTemp_C','Humidity_%','Pressure_hPa','WindSpeed_mps','WindDirection_deg','SOFT','MEDIUM','HARD','INTERMEDIATE','WET','races_prior_this_season']

def impute(df_from, df_to):
    for col in features:
        med = df_from[col].median()
        df_to[col] = df_to[col].fillna(med)

impute(train, train)
impute(train, val)
impute(train, test)

X_train = train[features]
y_train = train['DeviationFromAvg_s']
X_val = val[features]
y_val = val['DeviationFromAvg_s']
X_test = test[features]
y_test = test['DeviationFromAvg_s']

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# search space (conservative)
space = {
    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
    'max_depth': [3,4,5,6],
    'subsample': [0.6,0.7,0.8,1.0],
    'colsample_bytree': [0.6,0.7,0.8,1.0]
}

trials = []
best = {'rmse': 1e9}

for i in range(20):
    params = {
        'objective':'reg:squarederror',
        'eval_metric':'rmse',
        'learning_rate': random.choice(space['learning_rate']),
        'max_depth': random.choice(space['max_depth']),
        'subsample': random.choice(space['subsample']),
        'colsample_bytree': random.choice(space['colsample_bytree']),
        'seed': 42
    }
    print(f"Trial {i+1}/20: {params}")
    bst = xgb.train(params, dtrain, num_boost_round=2000, evals=[(dtrain,'train'),(dval,'val')], early_stopping_rounds=50, verbose_eval=False)
    pred_val = bst.predict(dval)
    rmse = mean_squared_error(y_val, pred_val, squared=False)
    trials.append({'trial': i+1, 'params': params, 'rmse_val': float(rmse), 'best_iteration': int(bst.best_iteration) if hasattr(bst, 'best_iteration') else None})
    print('  val RMSE', rmse, 'best_iter', getattr(bst, 'best_iteration', None))
    if rmse < best['rmse']:
        best = {'rmse': rmse, 'params': params, 'model': bst}

# save trials
trials_df = pd.DataFrame([{'trial': t['trial'], **t['params'], 'rmse_val': t['rmse_val'], 'best_iteration': t['best_iteration']} for t in trials])
trials_df.to_csv(art / 'xgb_tune_conservative_trials.csv', index=False)
# save best model and predictions
if 'model' in best:
    best['model'].save_model(art / 'xgb_model_tuned_conservative.json')
    pd.DataFrame({'y_true': y_val.reset_index(drop=True), 'y_pred': best['model'].predict(dval)}).to_csv(art / 'val_predictions_xgb_tuned_conservative.csv', index=False)
    pd.DataFrame({'y_true': y_test.reset_index(drop=True), 'y_pred': best['model'].predict(dtest)}).to_csv(art / 'test_predictions_xgb_tuned_conservative.csv', index=False)
    print('Saved best tuned model with val RMSE', best['rmse'])
    # attempt pred_contribs
    try:
        sv = best['model'].predict(dval, pred_contribs=True)
        st = best['model'].predict(dtest, pred_contribs=True)
        import numpy as _np
        cols = features + ['bias']
        pd.DataFrame(sv, columns=cols).to_csv(art / 'shap_val_xgb_tuned_conservative.csv', index=False)
        pd.DataFrame(st, columns=cols).to_csv(art / 'shap_test_xgb_tuned_conservative.csv', index=False)
        pd.Series(_np.abs(sv).mean(axis=0), index=cols).to_csv(art / 'shap_summary_val_xgb_tuned_conservative.csv')
        print('Saved pred_contribs for tuned model')
    except Exception as e:
        print('Could not compute pred_contribs for tuned model:', e)

# report best trial
with open(art / 'xgb_tune_conservative_best.json', 'w') as f:
    json.dump({'rmse_val': best['rmse'], 'params': best['params']}, f)
print('Tuning complete')
