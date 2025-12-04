# SHAP full visualizations for tuned model
# Uses Agg backend to reduce GUI-related crashes; samples X_val for beeswarm to limit memory.
import matplotlib
matplotlib.use('Agg')
import shap
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

art = Path('artifacts')
# load data
X_val = pd.read_csv(art / 'X_val.csv')
X_test = pd.read_csv(art / 'X_test.csv')

# load tuned model
model_path = art / 'xgb_model_tuned_conservative.json'
if not model_path.exists():
    raise FileNotFoundError(f'Tuned model not found at {model_path}')
booster = xgb.Booster()
booster.load_model(str(model_path))

# build explainer and compute shap values on a sample to reduce memory
print('Building TreeExplainer...')
explainer = shap.Explainer(booster)

# sample validation set (max 500 rows)
sample_n = min(500, len(X_val))
X_val_sample = X_val.sample(sample_n, random_state=42)
print(f'Computing SHAP values on {len(X_val_sample)} validation rows...')
shap_vals_sample = explainer(X_val_sample)

# summary bar plot (mean abs)
mean_abs = pd.Series(abs(shap_vals_sample.values).mean(axis=0), index=X_val_sample.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
mean_abs.head(20).plot.bar()
plt.ylabel('mean |SHAP value|')
plt.tight_layout()
plt.savefig(art / 'shap_beeswarm_summary_bar_tuned.png')
plt.close()

# beeswarm plot
print('Rendering beeswarm (PNG)...')
plt.figure(figsize=(10,6))
shap.plots.beeswarm(shap_vals_sample, max_display=20, show=False)
plt.tight_layout()
plt.savefig(art / 'shap_beeswarm_tuned.png')
plt.close()

# force plots for first 3 test rows
n_local = min(3, len(X_test))
shap_vals_test = explainer(X_test.iloc[:n_local])
for i in range(n_local):
    try:
        print('Rendering force plot for test row', i)
        f = shap.plots.force(shap_vals_test[i], matplotlib=False, show=False)
        html = shap.plots._html.get_iframe_html(f)
        (art / f'shap_force_tuned_row_{i}.html').write_text(html)
    except Exception as e:
        print('Failed to render force plot for row', i, 'error:', e)

print('SHAP full visuals completed')
