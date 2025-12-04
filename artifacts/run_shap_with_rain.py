"""Run SHAP for the saved pipeline+model and produce CSV + PNG summary highlighting 'Rain'.

Usage:
    python artifacts/run_shap_with_rain.py

This script expects the following files to exist relative to repo root:
- artifacts/xgb_best_with_pipeline.joblib
- artifacts/X_val.csv

It will write:
- artifacts/shap_with_rain_summary.csv
- artifacts/shap_with_rain_summary.png

If TreeExplainer fails it will fallback to KernelExplainer on a small subset.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
pipe_path = ART / 'xgb_best_with_pipeline.joblib'
Xval_path = ART / 'X_val.csv'

if not pipe_path.exists():
    print('model not found:', pipe_path)
    sys.exit(1)
if not Xval_path.exists():
    print('X_val not found:', Xval_path)
    sys.exit(1)

print('Loading pipeline...')
pipeline = joblib.load(pipe_path)
# X_val.csv is expected to be the transformed/encoded feature matrix saved earlier
X_val_df = pd.read_csv(Xval_path)

# Try to extract the underlying model
model = pipeline
try:
    # Pipeline-like
    if hasattr(pipeline, 'named_steps'):
        # common name 'model' or last step
        if 'model' in pipeline.named_steps:
            model = pipeline.named_steps['model']
        else:
            model = pipeline.steps[-1][1]
except Exception:
    model = pipeline

print('Using model type:', type(model))

# If model has a Booster with feature names, align X_val_df to those names (subset/reorder)
expected_feature_names = None
try:
    booster = model.get_booster()
    expected_feature_names = booster.feature_names
    print('Model (booster) expects', len(expected_feature_names), 'features')
except Exception:
    expected_feature_names = None

if expected_feature_names is not None:
    # Determine if X_val_df already contains transformed features (prefixed names like 'num__' or 'cat__')
    transformed_flag = any(c.startswith('num__') or c.startswith('cat__') for c in X_val_df.columns)
    if transformed_flag:
        print('Detected transformed X_val (prefixed columns). Will use as-is for alignment.')
        X_trans_df = X_val_df.copy()
    else:
        # Try to apply the saved preprocessor (if available) to the raw X_val_df
        X_trans_df = None
        try:
            preproc_path = ART / 'preprocessing_pipeline.joblib'
            preprocessor = None
            if preproc_path.exists():
                preprocessor = joblib.load(preproc_path)
            # if the loaded pipeline is an sklearn Pipeline, try to get its preprocessor
            if hasattr(pipeline, 'steps') and len(pipeline.steps) >= 2:
                candidate = pipeline.steps[0][1]
                if hasattr(candidate, 'transform'):
                    preprocessor = candidate

            if preprocessor is not None and hasattr(preprocessor, 'transform'):
                print('Using preprocessor to transform raw X_val:', type(preprocessor))
                X_trans = preprocessor.transform(X_val_df)
                try:
                    trans_names = preprocessor.get_feature_names_out(X_val_df.columns)
                except Exception:
                    try:
                        trans_names = preprocessor.get_feature_names_out()
                    except Exception:
                        trans_names = [f'f_{i}' for i in range(X_trans.shape[1])]
                X_trans_df = pd.DataFrame(X_trans, columns=trans_names, index=X_val_df.index)
            else:
                print('No usable preprocessor found; treating X_val as already transformed')
                X_trans_df = X_val_df.copy()
        except Exception as e:
            print('Error applying preprocessor; falling back to raw X_val:', e)
            X_trans_df = X_val_df.copy()

    # Map model's expected feature names to column names in X_trans_df, then align
    def map_name(n: str) -> str:
        # 'num__X' -> 'X'
        if n.startswith('num__'):
            return n[len('num__'):]
        # 'cat__Col_val' -> 'Col__val'
        if n.startswith('cat__'):
            rest = n[len('cat__'):]
            idx = rest.find('_')
            if idx == -1:
                return rest
            return rest[:idx] + '__' + rest[idx+1:]
        return n

    mapped_expected = [map_name(n) for n in expected_feature_names]
    present = [f for f in mapped_expected if f in X_trans_df.columns]
    missing = [f for f in mapped_expected if f not in X_trans_df.columns]
    extra = [c for c in X_trans_df.columns if c not in mapped_expected]
    print(f'present/expected/missing/extra counts: {len(present)}/{len(mapped_expected)}/{len(missing)}/{len(extra)}')
    if missing:
        print('Filling missing expected mapped features with zeros (mapped names):', missing)
        for m in missing:
            X_trans_df[m] = 0.0
    # reorder columns to match mapped_expected, then produce numpy array for model
    X_for_model_df = X_trans_df[mapped_expected].copy()
    # Instead of dropping columns (which breaks model input shape), set them to zero so the model
    # still receives the same number/order of features but the target and identity info are removed.
    # Target-related columns
    target_names = {'num__DeviationFromAvg_s', 'DeviationFromAvg_s'}
    present_targets = [c for c in X_for_model_df.columns if c in target_names]
    if present_targets:
        print('Zeroing target columns in inputs to avoid leakage:', present_targets)
        for c in present_targets:
            X_for_model_df[c] = 0.0

    # Zero any column containing DeviationFromAvg (all deviation-from-average variables)
    dev_cols = [c for c in X_for_model_df.columns if 'DeviationFromAvg' in c]
    if dev_cols:
        print('Zeroing DeviationFromAvg-related columns:', len(dev_cols))
        for c in dev_cols:
            X_for_model_df[c] = 0.0

    # Zero parsed time / position / driver number and other unwanted simple columns
    simple_zero = [c for c in X_for_model_df.columns if c in ('ParsedTime_s', 'ParsedTime', 'Position', 'DriverNumber', 'num__Position', 'Laps', 'Unnamed: 0', 'Round')]
    if simple_zero:
        print('Zeroing Parsed/Position/DriverNumber columns:', simple_zero)
        for c in simple_zero:
            X_for_model_df[c] = 0.0

    # Zero any column that looks like a time column (case-insensitive match)
    time_cols = [c for c in X_for_model_df.columns if 'time' in c.lower()]
    if time_cols:
        print('Zeroing time-related columns:', len(time_cols))
        for c in time_cols:
            X_for_model_df[c] = 0.0

    # Zero classified position one-hot columns
    classified_cols = [c for c in X_for_model_df.columns if c.startswith('ClassifiedPosition') or c.startswith('ClassifiedPosition__')]
    if classified_cols:
        print('Zeroing ClassifiedPosition one-hot columns:', len(classified_cols))
        for c in classified_cols:
            X_for_model_df[c] = 0.0

    # Identify driver/team columns (prefixed and unprefixed forms) and zero them out
    drop_prefixes = ('cat__Driver_', 'cat__TeamName', 'cat__TeamId')
    drop_unprefixed_prefixes = ('Driver__', 'TeamName__', 'TeamId__', 'Team__')
    identity_cols = [c for c in X_for_model_df.columns if any(c.startswith(p) for p in drop_prefixes + drop_unprefixed_prefixes)]
    if identity_cols:
        print('Zeroing driver/team identity columns in inputs (not necessary for analysis):', len(identity_cols))
        for c in identity_cols:
            X_for_model_df[c] = 0.0

    # Convert to numpy for model; keep feature names for mapping
    X_for_model = X_for_model_df.values
    feature_names = list(X_for_model_df.columns)
else:
    # No booster feature info; use X_val_df as-is
    X_for_model = X_val_df.values
    feature_names = list(X_val_df.columns)

try:
    explainer = shap.TreeExplainer(model)
    print('Using TreeExplainer')
    shap_vals = explainer.shap_values(X_for_model)
except Exception as e:
    print('TreeExplainer failed, attempting KernelExplainer fallback:', e)
    # KernelExplainer is expensive; sample a small background from transformed inputs
    try:
        bg_df = None
        if isinstance(X_for_model, pd.DataFrame):
            bg_df = shap.sample(X_for_model, 50, random_state=42)
            explainer = shap.KernelExplainer(lambda x: model.predict(pd.DataFrame(x, columns=X_for_model.columns)), bg_df)
        else:
            # X_for_model is numpy array; create background as numpy
            import numpy as _np
            bkg = _np.array(X_for_model[:50])
            explainer = shap.KernelExplainer(lambda x: model.predict(x), bkg)
        shap_vals = explainer.shap_values(X_for_model)
    except Exception as e2:
        print('KernelExplainer fallback failed:', e2)
        raise

# Normalize shap_vals shape
sv = shap_vals
if isinstance(sv, list):
    sv = np.array(sv)
if sv.ndim == 3:
    # [n_outputs, n_samples, n_features] -> take first output
    sv = sv[0]

abs_mean = np.abs(sv).mean(axis=0)
# Use the actual columns used for the model (after dropping target/team cols) if available,
# otherwise fall back to the feature_names list.
if 'X_for_model_df' in globals() or 'X_for_model_df' in locals():
    try:
        features = list(X_for_model_df.columns)
    except Exception:
        features = list(feature_names)
else:
    features = list(feature_names)
shap_df = pd.DataFrame({'feature': features, 'mean_abs_shap': abs_mean})
# Filter out the target and identity features from the reported summary
filter_prefixes = ('cat__Driver_', 'Driver__', 'cat__TeamName', 'TeamName__', 'cat__TeamId', 'TeamId__', 'cat__Team', 'Team__')
# Remove target, any DeviationFromAvg features, ParsedTime/Position/DriverNumber, and ClassifiedPosition one-hots
shap_df = shap_df[~shap_df['feature'].isin(target_names) & ~shap_df['feature'].str.startswith(filter_prefixes) & ~shap_df['feature'].str.contains('DeviationFromAvg') & ~shap_df['feature'].isin(['ParsedTime_s','ParsedTime','Position','DriverNumber','Laps','Unnamed: 0','Round']) & ~shap_df['feature'].str.startswith('ClassifiedPosition') & ~shap_df['feature'].str.contains('time', case=False)]
shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
shap_csv = ART / 'shap_with_rain_summary.csv'
shap_df.to_csv(shap_csv, index=False)

# plot, highlight Rain if present
plt.figure(figsize=(10, max(6, len(shap_df) * 0.25)))
colors = ['orange' if 'Rain' in f else 'grey' for f in shap_df['feature']]
plt.barh(shap_df['feature'][::-1], shap_df['mean_abs_shap'][::-1], color=colors[::-1])
plt.xlabel('Mean(|SHAP value|)')
plt.title('Global feature importance (mean absolute SHAP)')
plt.tight_layout()
out = ART / 'shap_with_rain_summary.png'
plt.savefig(out, dpi=150)
print('Wrote', shap_csv)
print('Wrote', out)
# --- Beeswarm and optional dependence plot ---
try:
    top_n = min(30, len(shap_df))
    beeswarm_feats = shap_df['feature'].tolist()[:top_n]
    # map to indices in the full feature list
    if isinstance(feature_names, (list, tuple)):
        idxs = [feature_names.index(f) for f in beeswarm_feats if f in feature_names]
    else:
        idxs = [i for i in range(min(len(beeswarm_feats), sv.shape[1]))]

    if len(idxs) > 0:
        sv_subset = sv[:, idxs]
        X_subset = X_for_model_df[beeswarm_feats]
        plt.figure(figsize=(8, max(6, len(beeswarm_feats) * 0.2)))
        # use summary_plot for compatibility with different shap versions
        shap.summary_plot(sv_subset, X_subset, show=False)
        out_bees = ART / 'shap_beeswarm.png'
        plt.tight_layout()
        plt.savefig(out_bees, dpi=150)
        print('Wrote beeswarm plot to', out_bees)
    else:
        print('No matching features found for beeswarm.')

    # Optional: dependence plot for Rain if present
    rain_candidates = [f for f in shap_df['feature'].tolist() if 'Rain' in f or f.lower().startswith('rain')]
    if rain_candidates:
        rain_feat = rain_candidates[0]
        if rain_feat in feature_names:
            plt.figure(figsize=(6,4))
            shap.dependence_plot(rain_feat, sv, X_for_model_df, show=False)
            out_dep = ART / f'shap_dep_{rain_feat}.png'
            plt.tight_layout()
            plt.savefig(out_dep, dpi=150)
            print('Wrote dependence plot for', rain_feat, 'to', out_dep)
except Exception as e:
    print('Beeswarm/dependence plotting failed:', e)
