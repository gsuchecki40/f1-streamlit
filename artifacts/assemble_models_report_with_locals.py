#!/usr/bin/env python3
"""Assemble a final HTML report that embeds SHAP images and links local explanation HTMLs."""
from pathlib import Path
import json

ART = Path('artifacts')
out = ART / 'models_report_with_locals.html'

# find global and beeswarm images
global_imgs = list(ART.glob('shap_*_global.png'))
beeswarm_imgs = {p.stem.replace('_beeswarm_fallback','').replace('_beeswarm',''): p for p in ART.glob('shap_*_beeswarm*.png')}

# find local htmls
local_htmls = {}
for p in ART.glob('shap_local_*.html'):
    # name format: shap_local_{short}_row{i}.html
    parts = p.stem.split('_')
    # e.g. ['shap','local','with','pipeline','row0'] or ['shap','local','xgb','...','row0']
    if 'row' in parts[-1]:
        row = parts[-1]
        short = '_'.join(parts[2:-1])
    else:
        short = '_'.join(parts[2:])
        row = 'row0'
    local_htmls.setdefault(short, []).append(p)

html = ['<html><head><meta charset="utf-8"><title>Models report with locals</title>',
        '<style>',
        'body{font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; margin:24px; color:#222}',
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}',
        '.card{border:1px solid #e6e6e6;border-radius:8px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,0.04)}',
        '.title{display:flex;justify-content:space-between;align-items:center}',
        '.badge{background:#0366d6;color:white;padding:6px 10px;border-radius:999px;font-size:0.9rem}',
        'img{max-width:100%;height:auto;border-radius:6px}',
        '.meta{color:#666;font-size:0.9rem;margin-bottom:8px}',
        '.caption{font-size:0.85rem;color:#444;margin-top:6px}',
        '</style></head><body>']
html.append('<h1 style="margin-top:0">Models report: SHAP visuals and local explanations</h1>')
html.append('<div class="grid">')

for g in sorted(global_imgs):
    short = g.stem.replace('shap_','').replace('_global','')
    html.append('<div class="card">')
    # header
    html.append('<div class="title">')
    html.append(f"<strong>{short}</strong>")
    # show RMSE if available
    # try to find matching val_predictions file
    rmse_text = ''
    for p in ART.glob('val_predictions*.csv'):
        if short in p.name or 'with_pipeline' in p.name:
            try:
                import pandas as _pd
                dfp = _pd.read_csv(p)
                dfp = dfp.dropna(subset=['y_true','y_pred'])
                import numpy as _np
                rmse = float(_np.sqrt(((dfp['y_true']-dfp['y_pred'])**2).mean()))
                rmse_text = f"<span class='badge'>RMSE {rmse:.2f}s</span>"
            except Exception:
                pass
            break
    if rmse_text:
        html.append(rmse_text)
    html.append('</div>')

    html.append(f"<div class=\"meta\">Path: {g.as_posix()}</div>")
    # global image
    html.append(f"<img src='{g.as_posix()}' alt='global shap'>")
    html.append(f"<div class=\"caption\">Global mean |SHAP| (top features)</div>")

    bw = beeswarm_imgs.get(short)
    if bw:
        html.append('<div style="margin-top:10px">')
        html.append(f"<img src='{bw.as_posix()}' alt='beeswarm'>")
        html.append(f"<div class=\"caption\">Beeswarm (fallback)</div>")
        html.append('</div>')

    locals_for = local_htmls.get(short)
    if locals_for:
        html.append('<div style="margin-top:10px"><details><summary>Local explanations</summary><ul>')
        for p in sorted(locals_for):
            html.append(f"<li><a href='{p.as_posix()}' target='_blank'>{p.name}</a></li>")
        html.append('</ul></details></div>')

    html.append('</div>')

html.append('</div>')
html.append('</body></html>')
out.write_text('\n'.join(html))
print('Wrote', out)
