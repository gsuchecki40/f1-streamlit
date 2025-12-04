import shap
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

art = Path('artifacts')

# load data (use artifacts/ CSVs)
X_val = pd.read_csv(art / 'X_val.csv')
X_test = pd.read_csv(art / 'X_test.csv')

# load booster
booster = xgb.Booster()
if (art / 'xgb_model_untitled_external.json').exists():
    booster.load_model(str(art / 'xgb_model_untitled_external.json'))
elif (art / 'xgb_model.json').exists():
    booster.load_model(str(art / 'xgb_model.json'))
else:
    raise FileNotFoundError('No xgb model JSON found in artifacts/')

# Create TreeExplainer
explainer = shap.Explainer(booster)
shap_vals_val = explainer(X_val)
shap_vals_test = explainer(X_test)

# Global importance: mean abs
shap_abs_mean = pd.Series(abs(shap_vals_val.values).mean(axis=0), index=X_val.columns).sort_values(ascending=False)
plt.figure(figsize=(8,6))
shap_abs_mean.head(20).plot.bar()
plt.tight_layout()
plt.savefig(art / 'shap_global_bar.png')
plt.close()

# SHAP summary plot (beeswarm)
plt.figure(figsize=(8,6))
shap.plots.beeswarm(shap_vals_val, show=False)
plt.tight_layout()
plt.savefig(art / 'shap_summary_beeswarm.png')
plt.close()

# Local explanations: save force plots for 3 sample rows from test
sample_idx = list(range(min(3, len(X_test))))
for i in sample_idx:
    f = shap.plots.force(shap_vals_test[i], matplotlib=False, show=False)
    html = shap.plots._html.get_iframe_html(f)
    (art / f'shap_force_test_row_{i}.html').write_text(html)

# Build a small HTML report
html_lines = [
    '<html><head><meta charset="utf-8"><title>SHAP report</title></head><body>',
    '<h1>SHAP report</h1>',
    '<h2>Global feature importance (mean |SHAP|)</h2>',
    '<img src="shap_global_bar.png" style="max-width:900px">',
    '<h2>Beeswarm summary</h2>',
    '<img src="shap_summary_beeswarm.png" style="max-width:900px">',
    '<h2>Local force plots (test rows)</h2>',
]
for i in sample_idx:
    html_lines.append(f'<a href="shap_force_test_row_{i}.html">Force plot test row {i}</a><br>')
html_lines.append('</body></html>')
(art / 'shap_report.html').write_text('\n'.join(html_lines))
print('SHAP report generated in artifacts/')
