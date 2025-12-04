#!/usr/bin/env python3
"""Generate an HTML report summarizing model runs, validation RMSE, and SHAP summaries.

Scans artifacts/ for files matching known patterns and composes a simple HTML page.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

ART = Path('artifacts')
out_html = ART / 'models_report.html'
out_json = ART / 'models_report_summary.json'

def find_patterns():
    models = list(ART.glob('xgb*.joblib')) + list(ART.glob('*xgb*.joblib'))
    preds = list(ART.glob('val_predictions*.csv'))
    shaps = list(ART.glob('shap_val*.csv'))
    summaries = list(ART.glob('shap_summary*.csv')) + list(ART.glob('shap_summary*.json'))
    return models, preds, shaps, summaries

models, preds, shaps, summaries = find_patterns()

report = {'models': []}

for m in models:
    name = m.stem
    entry = {'name': name, 'path': str(m)}
    # try to get params from joblib if possible
    try:
        import joblib
        obj = joblib.load(m)
        entry['type'] = type(obj).__name__
        # try to extract get_params
        if hasattr(obj, 'get_params'):
            entry['params'] = {k: str(v) for k,v in obj.get_params().items()}
    except Exception:
        entry['load_error'] = True

    # find matching predictions
    pred_file = None
    for p in preds:
        if name in p.name:
            pred_file = p
            break
    if pred_file:
        df = pd.read_csv(pred_file)
        df = df.dropna(subset=['y_true','y_pred'])
        entry['val_rmse'] = float(np.sqrt(((df['y_true']-df['y_pred'])**2).mean()))
        entry['predictions'] = str(pred_file)

    # find shap file
    shap_file = None
    for s in shaps:
        if name in s.name or 'with_pipeline' in s.name:
            shap_file = s
            break
    if shap_file:
        try:
            s_df = pd.read_csv(shap_file)
            # global importance by mean abs
            means = s_df.abs().mean().sort_values(ascending=False).head(10)
            entry['top_shap'] = means.to_dict()
        except Exception:
            entry['shap_error'] = True

    report['models'].append(entry)

# Save JSON summary
with open(out_json, 'w') as f:
    json.dump(report, f, indent=2)

# Build a simple HTML
html_lines = ["<html><head><meta charset='utf-8'><title>Models report</title></head><body>", '<h1>Model runs summary</h1>']
for e in report['models']:
    html_lines.append(f"<h2>{e['name']}</h2>")
    html_lines.append(f"<p>Path: {e['path']}</p>")
    if 'val_rmse' in e:
        html_lines.append(f"<p><b>Validation RMSE:</b> {e['val_rmse']:.4f}</p>")
    if 'params' in e:
        html_lines.append('<details><summary>Params</summary><pre>')
        html_lines.append(json.dumps(e['params'], indent=2))
        html_lines.append('</pre></details>')
    if 'top_shap' in e:
        html_lines.append('<p><b>Top SHAP features (mean abs):</b></p><ul>')
        for k,v in e['top_shap'].items():
            html_lines.append(f"<li>{k}: {v:.4g}</li>")
        html_lines.append('</ul>')
    html_lines.append('<hr>')

html_lines.append('</body></html>')
out_html.write_text('\n'.join(html_lines))
print('Wrote', out_html, 'and', out_json)
