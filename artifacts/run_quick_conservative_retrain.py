"""Quick conservative retrain: modest regularization, early stopping, and diagnostics.

Outputs:
- artifacts/xgb_conservative_quick.joblib
- artifacts/val_predictions_xgb_conservative_quick.csv
- artifacts/confusion_residuals_quick.png
- artifacts/cluster_heatmap.png
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

BASE = Path(__file__).resolve().parent.parent
ART = BASE / 'artifacts'
ART.mkdir(exist_ok=True)

import sys
from pathlib import Path as _P
# ensure project root is on sys.path so we can import artifacts.* modules
ROOT = _P(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from artifacts.retrain_minimal_features import load_and_select

def train_quick():
    X, y, df_full = load_and_select()
    # drop NA targets
    mask = ~y.isna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
    dval = xgb.DMatrix(X_val.values, label=y_val.values)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'auto',
        'seed': 42,
        'verbosity': 1,
        # conservative regularization
        'eta': 0.05,
        'max_depth': 4,
        'lambda': 2.0,
        'alpha': 0.5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }

    evallist = [(dtrain, 'train'), (dval, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round=1000, evals=evallist, early_stopping_rounds=20)

    joblib.dump(bst, ART / 'xgb_conservative_quick.joblib')

    # predictions
    pred = bst.predict(dval)
    pd.DataFrame({'y_true': y_val.reset_index(drop=True), 'y_pred': pred}).to_csv(ART / 'val_predictions_xgb_conservative_quick.csv', index=False)

    # confusion matrix by binning residuals into 3 classes
    resid = y_val.reset_index(drop=True) - pred
    # define bins: underperform (resid > +t), close (-t..+t), overperform (resid < -t)
    t = np.std(resid)  # threshold = 1 sigma
    labels = ['overperform','close','underperform']
    bins = [-np.inf, -t, t, np.inf]
    y_true_bin = pd.cut(resid, bins=bins, labels=labels)
    # as a quick diagnostic, compare predicted bin via predicted residual assuming y_pred ~ y_true -> use sign of (y_true - y_pred)
    y_pred_resid = y_val.reset_index(drop=True) - pred
    y_pred_bin = pd.cut(y_pred_resid, bins=bins, labels=labels)

    cm = pd.crosstab(y_true_bin, y_pred_bin)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('predicted bin')
    plt.ylabel('true bin')
    plt.title('Confusion matrix: residual bins')
    plt.tight_layout()
    plt.savefig(ART / 'confusion_residuals_quick.png')

    # Cluster heatmap from premodel_clusters.csv if available
    clusters = ART / 'premodel_clusters.csv'
    if clusters.exists():
        cdf = pd.read_csv(clusters)
        # pick numeric columns to cluster/heatmap
        num = cdf.select_dtypes(include=[np.number]).fillna(0)
        if not num.empty:
            # compute cluster means if cluster label exists
            if 'weather_tire_cluster' in cdf.columns:
                means = num.groupby(cdf['weather_tire_cluster']).mean()
                plt.figure(figsize=(8,6))
                sns.heatmap(means, annot=True, fmt='.2f', cmap='coolwarm')
                plt.title('Cluster means heatmap (numerical features)')
                plt.tight_layout()
                plt.savefig(ART / 'cluster_heatmap.png')

    print('Quick conservative retrain complete. Artifacts saved in artifacts/')

if __name__ == '__main__':
    train_quick()
