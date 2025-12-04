#!/usr/bin/env python3
"""Create matplotlib beeswarm fallback plots and simple local explanation HTML snippets.

For each shap_val_*.csv, create:
 - artifacts/shap_{short}_beeswarm_fallback.png
 - artifacts/shap_local_{short}_row{i}.html (for i in 0..N-1)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import html

ART = Path('artifacts')
shap_files = list(ART.glob('shap_val*.csv'))
pred_files = {p.stem: p for p in ART.glob('val_predictions*.csv')}

def beeswarm_matplotlib(shap_vals, feature_names, outpath, sample=1000):
    # shap_vals: (n_samples, n_features)
    n, m = shap_vals.shape
    if n > sample:
        idx = np.random.choice(n, sample, replace=False)
        data = shap_vals[idx]
    else:
        data = shap_vals
    abs_mean = np.abs(data).mean(axis=0)
    order = np.argsort(abs_mean)[::-1]
    topk = min(20, m)
    order = order[:topk]
    plt.figure(figsize=(8, max(4, topk*0.3)))
    # create a beeswarm-like scatter per feature
    ys = []
    xs = []
    labels = []
    for i, fi in enumerate(order):
        vals = data[:, fi]
        # jitter x positions around feature index
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.4
        xs.extend(vals.tolist())
        ys.extend((np.ones_like(vals) * i) + jitter.tolist())
        labels.extend([feature_names[fi]] * len(vals))
    plt.scatter(xs, ys, s=6, alpha=0.6)
    plt.yticks(range(len(order)), [feature_names[i] for i in order])
    plt.xlabel('SHAP value')
    plt.title('Beeswarm fallback (top features)')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

for s in shap_files:
    short = s.stem.replace('shap_val_','')
    try:
        s_df = pd.read_csv(s)
    except Exception as e:
        print('skip', s, e)
        continue
    feature_names = list(s_df.columns)
    shap_vals = s_df.values
    out_beeswarm = ART / f'shap_{short}_beeswarm_fallback.png'
    beeswarm_matplotlib(shap_vals, feature_names, out_beeswarm)

    # create local explanation HTML for first 5 rows if predictions exist
    # find matching predictions file
    pred = None
    for p in ART.glob('val_predictions*.csv'):
        if short in p.name or 'with_pipeline' in p.name:
            pred = p
            break
    if pred:
        dfp = pd.read_csv(pred)
        n_local = min(5, len(dfp))
        for i in range(n_local):
            y_true = dfp.loc[i, 'y_true'] if 'y_true' in dfp.columns else None
            y_pred = dfp.loc[i, 'y_pred'] if 'y_pred' in dfp.columns else None
            # get top contributing features for this row
            row_shap = shap_vals[i]
            idx = np.argsort(np.abs(row_shap))[::-1][:10]
            rows = []
            for fi in idx:
                rows.append((feature_names[fi], float(row_shap[fi])))
            html_lines = [f"<h2>Local explanation for {short} row {i}</h2>", f"<p>y_true: {y_true}, y_pred: {y_pred}</p>", '<table border=1><tr><th>feature</th><th>shap</th></tr>']
            for fn, sv in rows:
                html_lines.append(f"<tr><td>{html.escape(str(fn))}</td><td>{sv:.4f}</td></tr>")
            html_lines.append('</table>')
            out_local = ART / f'shap_local_{short}_row{i}.html'
            out_local.write_text('\n'.join(html_lines))

print('Generated beeswarm fallbacks and local HTML snippets')
