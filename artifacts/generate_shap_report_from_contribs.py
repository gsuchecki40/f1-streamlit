import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

art = Path('artifacts')
sv = art / 'shap_val_xgb_untitled_external.csv'
st = art / 'shap_test_xgb_untitled_external.csv'
if not sv.exists() or not st.exists():
    raise FileNotFoundError('Expected pred_contrib CSVs not found in artifacts/')

s_val = pd.read_csv(sv)
s_test = pd.read_csv(st)

# mean absolute contributions
mean_abs_val = s_val.abs().mean().sort_values(ascending=False)
mean_abs_test = s_test.abs().mean().sort_values(ascending=False)

# save bar plot for global importance (validation)
plt.figure(figsize=(10,6))
mean_abs_val.plot.bar(figsize=(10,6))
plt.ylabel('mean |contribution|')
plt.tight_layout()
plt.savefig(art / 'shap_global_bar_from_contribs.png')
plt.close()

# create simple HTML report
html_lines = [
    '<html><head><meta charset="utf-8"><title>SHAP (pred_contrib) report</title></head><body>',
    '<h1>SHAP-like report (XGBoost pred_contribs)</h1>',
    '<h2>Global importance (validation — mean |contribution|)</h2>',
    '<img src="shap_global_bar_from_contribs.png" style="max-width:900px">',
    '<h2>Top features (validation)</h2>',
    '<ol>'
]
for feat, val in mean_abs_val.head(10).items():
    html_lines.append(f'<li>{feat}: {val:.4f}</li>')
html_lines.append('</ol>')

# local explanations for first 3 test rows
html_lines.append('<h2>Local contributions (first 3 test rows)</h2>')
for i in range(min(3, len(s_test))):
    row = s_test.iloc[i:i+1].T
    row.columns = ['contribution']
    row_html = row.to_html()
    fname = f'shap_local_test_row_{i}.html'
    (art / fname).write_text(row_html)
    html_lines.append(f'<h3>Test row {i}</h3><a href="{fname}">Open contribution table</a>')

html_lines.append('</body></html>')
(art / 'shap_report_from_contribs.html').write_text('\n'.join(html_lines))
print('Generated SHAP-like report from pred_contrib CSVs in artifacts/')
