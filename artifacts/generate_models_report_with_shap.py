#!/usr/bin/env python3
"""Generate SHAP plots (global bar + beeswarm) from shap_val_*.csv and embed into an HTML report.

Outputs:
- artifacts/shap_{model}_global.png
- artifacts/shap_{model}_beeswarm.png
- artifacts/models_report_with_shap.html
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

ART = Path('artifacts')
shap_files = list(ART.glob('shap_val*.csv'))
preds = {p.stem: p for p in ART.glob('val_predictions*.csv')}
models = list(ART.glob('*xgb*.joblib'))

report_entries = []

for s in shap_files:
    name = s.stem
    # create safe short name
    short = name.replace('shap_val_','').replace('.csv','')
    try:
        shap_df = pd.read_csv(s)
    except Exception as e:
        print('Failed to read', s, e)
        continue

    feature_names = list(shap_df.columns)
    shap_vals = shap_df.values

    # global importance
    mean_abs = np.abs(shap_vals).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1]
    topk = min(20, len(idx))
    top_idx = idx[:topk]
    top_feats = [feature_names[i] for i in top_idx]
    top_vals = mean_abs[top_idx]

    global_png = ART / f'shap_{short}_global.png'
    plt.figure(figsize=(6, max(3, topk*0.25)))
    y_pos = np.arange(len(top_feats))
    plt.barh(y_pos, top_vals[::-1], align='center')
    plt.yticks(y_pos, top_feats[::-1])
    plt.xlabel('Mean |SHAP value|')
    plt.title(f'Global SHAP (top {topk}) — {short}')
    plt.tight_layout()
    plt.savefig(global_png, dpi=150)
    plt.close()

    # attempt beeswarm using shap library and artifacts/X_val.csv if available
    beeswarm_png = ART / f'shap_{short}_beeswarm.png'
    try:
        import shap
        X_val = None
        xval_path = ART / 'X_val.csv'
        if xval_path.exists():
            try:
                X_df = pd.read_csv(xval_path)
                # select columns that match shap_df columns
                common = [c for c in feature_names if c in X_df.columns]
                if len(common) == len(feature_names):
                    X_val = X_df[feature_names]
                else:
                    # fall back to using first columns
                    X_val = X_df.iloc[:, :len(feature_names)]
                    X_val.columns = feature_names
            except Exception:
                X_val = None

        # create beeswarm plot
        try:
            shap_values = shap_vals
            plt.figure(figsize=(8,6))
            if X_val is not None:
                shap.plots.beeswarm(shap_values, features=X_val, show=False)
            else:
                # shap beeswarm without feature values
                shap.plots.beeswarm(shap_values, show=False)
            plt.title(f'Beeswarm SHAP — {short}')
            plt.tight_layout()
            plt.savefig(beeswarm_png, dpi=150)
            plt.close()
            beeswarm_ok = True
        except Exception as e:
            print('shap beeswarm failed for', short, e)
            beeswarm_ok = False
    except Exception as e:
        print('shap library not available or failed for', short, e)
        beeswarm_ok = False

    # compute validation RMSE if prediction file exists
    val_rmse = None
    for p in ART.glob('val_predictions*.csv'):
        if short in p.name or 'with_pipeline' in p.name:
            try:
                dfp = pd.read_csv(p)
                dfp = dfp.dropna(subset=['y_true','y_pred'])
                val_rmse = float(np.sqrt(((dfp['y_true']-dfp['y_pred'])**2).mean()))
                break
            except Exception:
                pass

    entry = {
        'short': short,
        'shap_csv': str(s),
        'global_png': str(global_png),
        'beeswarm_png': str(beeswarm_png) if beeswarm_ok else None,
        'val_rmse': val_rmse
    }
    report_entries.append(entry)

# build HTML
html = ['<html><head><meta charset="utf-8"><title>Models report with SHAP</title></head><body>']
html.append('<h1>Models report with SHAP visuals</h1>')
for e in report_entries:
    html.append(f"<h2>{e['short']}</h2>")
    if e['val_rmse'] is not None:
        html.append(f"<p><b>Validation RMSE:</b> {e['val_rmse']:.4f}</p>")
    html.append('<p><b>Global SHAP (top features)</b></p>')
    html.append(f"<img src='{Path(e['global_png']).as_posix()}' style='max-width:900px;'>")
    if e['beeswarm_png']:
        html.append('<p><b>Beeswarm</b></p>')
        html.append(f"<img src='{Path(e['beeswarm_png']).as_posix()}' style='max-width:900px;'>")
    html.append('<hr>')

html.append('</body></html>')
out = ART / 'models_report_with_shap.html'
out.write_text('\n'.join(html))
print('Wrote', out)
