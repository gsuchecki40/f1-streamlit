#!/usr/bin/env python3
"""
Reconstruct the exact DataFrame used for training the tuned XGBoost model
by using the same feature list and median imputation computed from the 2023
training partition. Then run shap.TreeExplainer on the validation set and
save exact beeswarm + force visuals (no padding/truncation).

Run with the conda Python that has xgboost/shap installed. Use guards:
MALLOC_ARENA_MAX=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python artifacts/run_shap_exact_training_inputs.py
"""
import os
import sys
import traceback
import pandas as pd
import numpy as np

try:
    import xgboost as xgb
    import shap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    print('Missing required packages:', e)
    raise


BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
LOG = os.path.join(BASE, 'shap_exact_run.log')


def log(msg):
    print(msg)
    with open(LOG, 'a') as f:
        f.write(msg + '\n')


def main():
    try:
        log('Starting exact-input SHAP run')

        csv_path = os.path.join(ROOT, 'premodeldatav1.csv')
        model_path = os.path.join(BASE, 'xgb_model_tuned_conservative.json')

        log(f'Loading CSV {csv_path}')
        df = pd.read_csv(csv_path)

        # Use same filtering as tuning: remove rows without target
        df = df[~df['DeviationFromAvg_s'].isna()].copy()

        # define features that tuning used (must match the tune script)
        features = ['GridPosition','AirTemp_C','TrackTemp_C','Humidity_%','Pressure_hPa','WindSpeed_mps','WindDirection_deg','SOFT','MEDIUM','HARD','INTERMEDIATE','WET','races_prior_this_season']

        # split seasons same as tuning
        train = df[df.get('Season') == 2023].copy()
        val = df[df.get('Season') == 2024].copy()
        if val.empty:
            # fallback to second-last season if 2024 missing
            seasons = sorted(df['Season'].dropna().unique())
            if len(seasons) >= 2:
                val = df[df['Season'] == seasons[-2]].copy()

        log(f'Train rows: {len(train)}, Val rows: {len(val)}')

        # compute medians on train and impute
        medians = {}
        for col in features:
            med = train[col].median()
            medians[col] = med
            train[col] = train[col].fillna(med)
            val[col] = val[col].fillna(med)

        X_val = val[features].copy()

        log(f'X_val shape after imputation: {X_val.shape}; columns: {X_val.columns.tolist()}')

        # load tuned model
        log(f'Loading tuned model from {model_path}')
        bst = xgb.Booster()
        bst.load_model(model_path)
        expected = int(bst.num_features())
        log(f'Booster reports it expects {expected} features')

        # check exact alignment
        if X_val.shape[1] != expected:
            log(f'Feature count mismatch: X_val has {X_val.shape[1]} columns, model expects {expected}. Aborting to avoid silent misalignment.')
            raise RuntimeError('Feature count mismatch between reconstructed X_val and model')

        # sample to reduce memory
        sample_n = min(200, max(1, int(len(X_val) * 0.2)))
        X_sample = X_val.sample(n=sample_n, random_state=0) if len(X_val) > sample_n else X_val

        log(f'Running TreeExplainer on sample size {len(X_sample)}')
        explainer = shap.TreeExplainer(bst)
        shap_exp = explainer(X_sample)
        log(f'Computed SHAP values shape: {getattr(shap_exp, "values", None).shape}')

        beeswarm = os.path.join(BASE, 'shap_exact_beeswarm.png')
        plt.figure(figsize=(10,6))
        shap.plots.beeswarm(shap_exp, max_display=30)
        plt.tight_layout()
        plt.savefig(beeswarm, dpi=150)
        plt.close()
        log(f'Saved beeswarm to {beeswarm}')

        force = os.path.join(BASE, 'shap_exact_force.html')
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

        log('Exact-input SHAP run complete')

    except Exception as exc:
        log('Fatal error: ' + str(exc))
        traceback.print_exc()


if __name__ == '__main__':
    main()
