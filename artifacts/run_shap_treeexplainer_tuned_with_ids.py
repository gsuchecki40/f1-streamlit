#!/usr/bin/env python3
"""
Run SHAP TreeExplainer on the tuned model without dropping identity columns.
Saves:
 - artifacts/shap_tuned_beeswarm_with_ids.png
 - artifacts/shap_tuned_force_with_ids.html
and logs to artifacts/shap_tuned_run_with_ids.log
"""
import os, sys, traceback
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import xgboost as xgb
    import shap

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(BASE_DIR, 'shap_tuned_run_with_ids.log')
    def log(msg):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    xval_path = os.path.join(BASE_DIR, 'X_val.csv')
    model_path = os.path.join(BASE_DIR, 'xgb_model_tuned_conservative.json')

    log('Loading X_val from {}'.format(xval_path))
    try:
        X_val = pd.read_csv(xval_path, index_col=0)
    except Exception:
        X_val = pd.read_csv(xval_path)

    log('X_val shape: {}'.format(X_val.shape))

    # sample
    sample_n = min(100, max(1, int(len(X_val) * 0.1)))
    if len(X_val) > sample_n:
        X_sample = X_val.sample(n=sample_n, random_state=0)
    else:
        X_sample = X_val.copy()

    log('Loading model {}'.format(model_path))
    bst = xgb.Booster()
    bst.load_model(model_path)
    log('Model loaded; booster.num_features()={}'.format(bst.num_features()))

    # Pad/truncate to match model expected features
    expected = int(bst.num_features())
    X_used = X_sample
    if X_sample.shape[1] != expected:
        log('Mismatch: X_sample has {}, model expects {}. Padding/truncating'.format(X_sample.shape[1], expected))
        if expected > X_sample.shape[1]:
            n_pad = expected - X_sample.shape[1]
            pad_cols = {f'pad_col_{i}': 0.0 for i in range(n_pad)}
            pad_df = pd.DataFrame(pad_cols, index=X_sample.index)
            X_used = pd.concat([X_sample.reset_index(drop=True), pad_df.reset_index(drop=True)], axis=1)
        else:
            X_used = X_sample.iloc[:, :expected]
    log('X_used shape: {}'.format(X_used.shape))

    log('Building TreeExplainer...')
    explainer = shap.TreeExplainer(bst)
    shap_exp = explainer(X_used)
    log('Computed SHAP values shape: {}'.format(getattr(shap_exp, 'values', None).shape))

    beeswarm_path = os.path.join(BASE_DIR, 'shap_tuned_beeswarm_with_ids.png')
    plt.figure(figsize=(8,6))
    shap.plots.beeswarm(shap_exp, max_display=30)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()
    log('Saved beeswarm to {}'.format(beeswarm_path))

    force_path = os.path.join(BASE_DIR, 'shap_tuned_force_with_ids.html')
    try:
        i = 0
        fv = shap_exp.values[i]
        bv_all = getattr(shap_exp, 'base_values', None)
        bv = bv_all[i] if (hasattr(bv_all, '__len__') and len(bv_all) > i) else bv_all
        X_row = X_used.iloc[i] if hasattr(X_used, 'iloc') else pd.Series(X_used[i])
        f = shap.force_plot(bv, fv, X_row, matplotlib=False)
        shap.save_html(force_path, f)
        log('Saved force to {}'.format(force_path))
    except Exception as e:
        log('Failed to save force plot: {}'.format(e))
        traceback.print_exc()

    log('Done')
except Exception as exc:
    tb = traceback.format_exc()
    print(tb)
    sys.exit(2)
