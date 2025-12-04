#!/usr/bin/env python3
"""
Apply the persisted preprocessing pipeline to the raw validation rows and run
shap.TreeExplainer against the tuned XGBoost model to produce faithful
beeswarm and per-row force plots.

Run this from project root (recommended under the conda Python used for training):

MALLOC_ARENA_MAX=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python artifacts/run_faithful_shap.py

Outputs placed in `artifacts/`:
 - shap_faithful_beeswarm.png
 - shap_faithful_force.html
 - shap_faithful_run.log
"""
import os
import sys
import traceback
import pandas as pd

try:
    import joblib
    import xgboost as xgb
    import shap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    print('Required packages missing: ', e)
    raise


BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
LOG = os.path.join(BASE, 'shap_faithful_run.log')


def log(msg):
    print(msg)
    with open(LOG, 'a') as f:
        f.write(msg + '\n')


def main():
    try:
        log('Starting faithful SHAP run')

        # load raw CSV and pipeline
        csv_path = os.path.join(ROOT, 'premodeldatav1.csv')
        pipe_path = os.path.join(BASE, 'preprocessing_pipeline.joblib')
        model_path = os.path.join(BASE, 'xgb_model_tuned_conservative.json')

        log(f'Loading CSV {csv_path}')
        df = pd.read_csv(csv_path)

        # detect validation season used in preprocessing (defaults to 2024)
        val_season = 2024
        if 'Season' in df.columns and 2024 not in df['Season'].unique():
            # fallback to the last-but-one season if different
            seasons = sorted(df['Season'].dropna().unique())
            if len(seasons) >= 2:
                val_season = seasons[-2]

        log(f'Filtering validation rows for Season == {val_season}')
        df_val = df[df['Season'] == val_season].copy()
        if df_val.empty:
            log('No validation rows found for chosen season; exiting')
            return

        log(f'Loading pipeline from {pipe_path}')
        pipeline = joblib.load(pipe_path)

        # Apply pipeline.transform to get exact model inputs
        log('Transforming validation rows with pipeline...')
        X_val = pipeline.transform(df_val)

        # derive feature names from pipeline if possible
        feature_names = None
        try:
            feature_names = list(pipeline.named_steps['preprocessor'].get_feature_names_out())
        except Exception:
            try:
                # fallback to pipeline helper if present
                from sklearn.compose import ColumnTransformer
                names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                feature_names = list(names)
            except Exception:
                feature_names = [f'f_{i}' for i in range(X_val.shape[1])]

        log(f'Transformed X_val shape: {X_val.shape}; features: {len(feature_names)}')

        X_val_df = pd.DataFrame(X_val, columns=feature_names, index=df_val.index)

        # optionally drop leakage columns if present
        for leak in ['Position', 'DriverNumber']:
            if leak in X_val_df.columns:
                log(f'Dropping leakage column {leak} from transformed features')
                X_val_df.drop(columns=[leak], inplace=True)

        # load model
        log(f'Loading XGBoost model from {model_path}')
        bst = xgb.Booster()
        bst.load_model(model_path)
        expected = int(bst.num_features())
        log(f'Model expects {expected} features (booster.num_features)')

        # align X_val_df columns to expected by trimming or padding zeros
        X_used = X_val_df.copy()
        if X_used.shape[1] != expected:
            log(f'Feature count mismatch: X has {X_used.shape[1]} cols, model expects {expected}. Padding/truncating')
            if expected > X_used.shape[1]:
                n_pad = expected - X_used.shape[1]
                pad_cols = {f'pad_col_{i}': 0.0 for i in range(n_pad)}
                pad_df = pd.DataFrame(pad_cols, index=X_used.index)
                X_used = pd.concat([X_used.reset_index(drop=True), pad_df.reset_index(drop=True)], axis=1)
            else:
                X_used = X_used.iloc[:, :expected]

        log(f'X_used shape for SHAP: {X_used.shape}')

        # small sample to reduce memory pressure
        sample_n = min(200, max(1, int(len(X_used) * 0.2)))
        X_sample = X_used.sample(n=sample_n, random_state=0) if len(X_used) > sample_n else X_used

        log(f'Running TreeExplainer on sample size {len(X_sample)}')
        explainer = shap.TreeExplainer(bst)
        shap_exp = explainer(X_sample)
        log(f'Computed SHAP values shape: {getattr(shap_exp, "values", None).shape}')

        beeswarm = os.path.join(BASE, 'shap_faithful_beeswarm.png')
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_exp, max_display=30)
        plt.tight_layout()
        plt.savefig(beeswarm, dpi=150)
        plt.close()
        log(f'Saved beeswarm to {beeswarm}')

        # force plot for first sample row
        force = os.path.join(BASE, 'shap_faithful_force.html')
        try:
            i = 0
            fv = shap_exp.values[i]
            bv_all = getattr(shap_exp, 'base_values', None)
            bv = bv_all[i] if (hasattr(bv_all, '__len__') and len(bv_all) > i) else bv_all
            X_row = X_sample.iloc[i]
            f = shap.force_plot(bv, fv, X_row, matplotlib=False)
            shap.save_html(force, f)
            log(f'Saved force plot to {force}')
        except Exception as e:
            log('Could not save force plot: ' + str(e))
            traceback.print_exc()

        log('Faithful SHAP run complete')

    except Exception as exc:
        log('Fatal error in faithful SHAP run: ' + str(exc))
        traceback.print_exc()


if __name__ == '__main__':
    main()
