#!/usr/bin/env python3
"""
Run SHAP TreeExplainer on the tuned XGBoost model using the pipeline-transformed validation set (artifacts/X_val.csv).
Saves:
 - artifacts/shap_tuned_beeswarm.png
 - artifacts/shap_tuned_force.html
and logs to artifacts/shap_tuned_run.log

This script is defensive: uses a small random sample and writes tracebacks on failure.
"""
import os
import sys
import traceback

try:
    # Set Agg backend early
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import json
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import shap

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    log_path = os.path.join(BASE_DIR, 'shap_tuned_run.log')

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    log('Starting SHAP TreeExplainer run...')

    # Paths
    xval_path = os.path.join(BASE_DIR, 'X_val.csv')
    possible_models = [
        os.path.join(BASE_DIR, 'xgb_model_tuned_conservative.json'),
        os.path.join(BASE_DIR, 'xgb_model_tuned_conservative.model'),
        os.path.join(BASE_DIR, 'xgb_model_tuned_conservative.bin'),
        os.path.join(BASE_DIR, 'xgb_model_tuned_conservative.xgb'),
        os.path.join(BASE_DIR, 'xgb_model_tuned_conservative.pkl'),
    ]

    model_path = None
    for p in possible_models:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        raise FileNotFoundError('Could not find tuned model in artifacts/. Checked: {}'.format(possible_models))

    log('Using model: {}'.format(model_path))

    if not os.path.exists(xval_path):
        raise FileNotFoundError('Missing transformed X_val file at {}'.format(xval_path))

    # Load transformed validation features (already pipeline-transformed)
    log('Loading X_val from {}'.format(xval_path))
    try:
        X_val = pd.read_csv(xval_path, index_col=0)
    except Exception:
        # try without index_col
        X_val = pd.read_csv(xval_path)

    # Remove potential leakage features 'Position' and 'DriverNumber' if present
    drop_cols = []
    for c in ['Position', 'DriverNumber']:
        if c in X_val.columns:
            drop_cols.append(c)
    if drop_cols:
        log(f"Dropping columns from X_val to avoid leakage into explanations: {drop_cols}")
        X_val = X_val.drop(columns=drop_cols)

    log('X_val shape: {}'.format(X_val.shape))

    # Limit sample size to avoid OOMs / crashes
    sample_n = min(100, max(1, int(len(X_val) * 0.1)))
    sample_n = min(sample_n, 200)
    log('Sample size for SHAP: {}'.format(sample_n))
    if len(X_val) > sample_n:
        X_sample = X_val.sample(n=sample_n, random_state=0)
    else:
        X_sample = X_val.copy()

    # Load model
    log('Loading XGBoost model into Booster...')
    bst = xgb.Booster()
    bst.load_model(model_path)
    log('Model loaded.')

    # Determine expected feature count from the booster
    try:
        expected_feat = int(bst.num_features())
    except Exception:
        expected_feat = None
    log('Booster reports num_features={}'.format(expected_feat))

    # If X_sample has different number of columns, pad with zeros to match expected
    X_used = X_sample
    if expected_feat is not None and X_sample.shape[1] != expected_feat:
        log('Feature count mismatch: X_sample has {}, model expects {}. Padding with zeros.'.format(X_sample.shape[1], expected_feat))
        n_pad = expected_feat - X_sample.shape[1]
        if n_pad > 0:
            pad_cols = {f'pad_col_{i}': 0.0 for i in range(n_pad)}
            pad_df = pd.DataFrame(pad_cols, index=X_sample.index)
            # Append pad cols to X_sample
            X_used = pd.concat([X_sample.reset_index(drop=True), pad_df.reset_index(drop=True)], axis=1)
        else:
            # If model expects fewer features, truncate
            X_used = X_sample.iloc[:, :expected_feat]
        log('After padding/truncation, X_used shape: {}'.format(X_used.shape))

    # Build SHAP explainer
    log('Building TreeExplainer (this may take a moment)...')
    explainer = shap.TreeExplainer(bst)
    log('Explainer built. Computing SHAP values...')

    # Compute SHAP values; prefer passing the DataFrame so SHAP keeps feature names
    try:
        shap_exp = explainer(X_used)
        vals_shape = getattr(shap_exp, 'values', None)
        log('Computed SHAP values with DataFrame. shape: {}'.format(None if vals_shape is None else vals_shape.shape))
    except Exception as e:
        log('Explainer(DataFrame) failed: {}'.format(e))
        log('Attempting fallback: call explainer with numpy array values')
        try:
            shap_exp = explainer(X_used.values)
            vals_shape = getattr(shap_exp, 'values', None)
            log('Fallback computed SHAP values (numpy). shape: {}'.format(None if vals_shape is None else vals_shape.shape))
        except Exception:
            log('Fallback explainer call also failed; re-raising')
            raise

    # Beeswarm plot
    beeswarm_path = os.path.join(BASE_DIR, 'shap_tuned_beeswarm.png')
    log('Rendering beeswarm to {}'.format(beeswarm_path))
    try:
        plt.figure(figsize=(8, 6))
        shap.plots.beeswarm(shap_exp, max_display=30)
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=150)
        plt.close()
        log('Saved beeswarm PNG.')
    except Exception as e:
        log('Failed to render beeswarm: {}'.format(e))
        traceback.print_exc()

    # Force plot for a single representative row (first sample)
    try:
        force_path = os.path.join(BASE_DIR, 'shap_tuned_force.html')
        log('Saving force plot to {}'.format(force_path))
        # Choose the first row index
        i = 0
        # Get shap values and base value for the instance
        fv = shap_exp.values[i] if hasattr(shap_exp, 'values') else shap_exp
        bv_all = getattr(shap_exp, 'base_values', None)
        if bv_all is None:
            bv = 0.0
        else:
            try:
                bv = bv_all[i]
            except Exception:
                bv = bv_all

        # Use the same input row representation passed to the explainer
        X_row = None
        if isinstance(X_used, pd.DataFrame):
            X_row = X_used.iloc[i]
        else:
            # if X_used is numpy array
            X_row = pd.Series(X_used[i], index=[f'feature_{j}' for j in range(X_used.shape[1])])

        # Build force plot
        f = shap.force_plot(bv, fv, X_row, matplotlib=False)
        # Save as html
        shap.save_html(force_path, f)
        log('Saved force HTML.')
    except Exception as e:
        log('Failed to produce/save force plot: {}'.format(e))
        traceback.print_exc()

    log('SHAP TreeExplainer run completed successfully (or with recoverable plot errors).')

except Exception as exc:
    tb = traceback.format_exc()
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shap_tuned_run.log'), 'a') as f:
            f.write('\nERROR:\n')
            f.write(tb)
    except Exception:
        pass
    print('ERROR during SHAP run:')
    print(tb)
    sys.exit(2)
