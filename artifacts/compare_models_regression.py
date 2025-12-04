"""Compute RMSE/MAE/R^2 for each model's validation predictions and produce comparison plots.

Scans `artifacts/` for files named like `val_predictions_*.csv` and expects columns `y_true,y_pred`.
Saves `artifacts/model_comparison_metrics.csv` and plots `artifacts/model_comparison_metrics.png`.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ART = Path(__file__).resolve().parent

def find_val_files():
    files = sorted(ART.glob('val_predictions_*.csv'))
    # also include common names
    extra = [ART / 'val_predictions_with_pipeline.csv', ART / 'val_predictions_xgb_tuned_conservative.csv', ART / 'val_predictions_xgb_external.csv']
    for p in extra:
        if p.exists() and p not in files:
            files.append(p)
    return files

def compute_metrics(files):
    rows = []
    for f in files:
        name = f.stem.replace('val_predictions_','')
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if 'y_true' not in df.columns or 'y_pred' not in df.columns:
            continue
        y_true = df['y_true'].astype(float).values
        y_pred = df['y_pred'].astype(float).values
        # some sklearn versions don't support `squared` kwarg; compute RMSE via sqrt of MSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rows.append({'model': name, 'rmse': rmse, 'mae': mae, 'r2': r2, 'rows': len(df)})
    return pd.DataFrame(rows).set_index('model')

def plot_metrics(df_metrics):
    out = ART / 'model_comparison_metrics.png'
    df = df_metrics.sort_values('rmse')
    fig, ax = plt.subplots(1,3, figsize=(16,5))
    df['rmse'].plot(kind='bar', ax=ax[0], title='RMSE (lower better)')
    df['mae'].plot(kind='bar', ax=ax[1], title='MAE (lower better)')
    df['r2'].plot(kind='bar', ax=ax[2], title='R^2 (higher better)')
    for a in ax:
        a.set_xlabel('model')
        a.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    fig.savefig(out)
    return out

def main():
    files = find_val_files()
    if not files:
        print('No val prediction files found in artifacts/ matching val_predictions_*.csv')
        return
    print('Found prediction files:', [f.name for f in files])
    dfm = compute_metrics(files)
    dfm.to_csv(ART / 'model_comparison_metrics.csv')
    out = plot_metrics(dfm)
    print('Wrote metrics CSV and plot to artifacts/')

if __name__ == '__main__':
    main()
