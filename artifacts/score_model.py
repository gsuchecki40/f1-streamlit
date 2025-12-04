"""
Simple scoring wrapper for the Formula1 modeling artifacts.

Usage: run from project root (one level up from `artifacts/`):
  python3 artifacts/score_model.py --input path/to/new_data.csv --output artifacts/scored_preds.csv

Behavior:
- Loads `preprocessing_pipeline.joblib` to transform input features.
- If `artifacts/ensemble_fold_models/` exists, loads each fold model and averages their predictions.
- Otherwise falls back to `artifacts/xgb_minimal_cv_randomized.joblib` or `artifacts/xgb_minimal.joblib`.
- If `artifacts/linear_calibration.joblib` exists it is applied to the averaged predictions.

Produces: CSV with columns [index, prediction].

"""
import argparse
import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import json

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def load_preprocessor():
    p = ARTIFACTS / "preprocessing_pipeline.joblib"
    if not p.exists():
        raise FileNotFoundError(f"Preprocessing pipeline not found at {p}")
    # Some scikit-learn versions use private helper classes when pickling
    # (e.g. `_RemainderColsList` in `sklearn.compose._column_transformer`).
    # When loading a pipeline serialized with a different sklearn release,
    # unpickling can fail with AttributeError. Provide a safe fallback by
    # injecting a compatible placeholder class into the module before
    # calling `joblib.load`.
    try:
        # Ensure the module object that the pickle will reference is
        # present in sys.modules and has the expected attribute. Some
        # sklearn releases move or rename internal helpers which breaks
        # unpickling; adding a minimal stand-in class avoids the
        # AttributeError and lets joblib reconstruct the pipeline.
        import sys
        import types
        mod_name = 'sklearn.compose._column_transformer'
        if mod_name not in sys.modules:
            try:
                import importlib
                sys.modules[mod_name] = importlib.import_module(mod_name)
            except Exception:
                # create a lightweight module object as fallback
                sys.modules[mod_name] = types.ModuleType(mod_name)
        mod = sys.modules[mod_name]
        if not hasattr(mod, '_RemainderColsList'):
            class _RemainderColsList(list):
                pass
            setattr(mod, '_RemainderColsList', _RemainderColsList)
    except Exception:
        # If anything goes wrong here, proceed to joblib.load and allow
        # its original error to surface to the user.
        pass

    return joblib.load(p)


def load_ensemble_models():
    # prefer a remove-lapped ensemble if available
    models_dir = ARTIFACTS / "ensemble_fold_models_remove_lapped"
    if not (models_dir.exists() and any(models_dir.iterdir())):
        models_dir = ARTIFACTS / "ensemble_fold_models"
    if models_dir.exists() and any(models_dir.iterdir()):
        models = []
        for f in sorted(models_dir.iterdir()):
            if f.suffix in ('.joblib', '.pkl'):
                models.append(joblib.load(f))
        if models:
            return models
    # fallback single-model options
    for fname in ("xgb_minimal_cv_randomized.joblib", "xgb_minimal_cv.joblib", "xgb_minimal.joblib"):
        candidate = ARTIFACTS / fname
        if candidate.exists():
            return [joblib.load(candidate)]
    raise FileNotFoundError("No model artifacts found in artifacts/ (searched ensemble folder and fallback models)")


def apply_calibration(preds: np.ndarray) -> np.ndarray:
    calib_path = ARTIFACTS / "linear_calibration.joblib"
    if calib_path.exists():
        lr = joblib.load(calib_path)
        # expect lr to be a scikit-learn regressor with predict
        return lr.predict(preds.reshape(-1, 1))
    return preds


def score(input_csv: Path, output_csv: Path):
    preproc = load_preprocessor()
    models = load_ensemble_models()

    df = pd.read_csv(input_csv)
    # normalize minimal features (Rain tokens -> numeric etc.) as early as possible
    try:
        from artifacts.input_normalize import normalize_minimal_features
        df = normalize_minimal_features(df)
    except Exception:
        # best-effort: if normalization helper fails, continue with original df
        pass
    # Schema validation: ensure input contains at least the minimal columns (or reasonable alternates)
    minimal_cols = ['GridPosition','AvgQualiTime','weather_tire_cluster','SOFT','MEDIUM','HARD','INTERMEDIATE','WET','races_prior_this_season','Rain','PointsProp','Status']
    missing = [c for c in minimal_cols if c not in df.columns]
    if missing:
        # don't abort immediately — we'll attempt to proceed but inform the user
        print(f"Warning: input CSV is missing expected columns: {missing}. The scorer will attempt imputation/fallbacks, but results may be degraded.")
    # drop lapped drivers if present
    initial_n = len(df)
    if 'Status' in df.columns:
        if (df['Status'] == 'Lapped').any():
            df = df[df['Status'] != 'Lapped']
            print(f"Dropped {(initial_n - len(df))} rows with Status == 'Lapped'")
    else:
        # one-hot encoded convention
        for col in ['Status__Lapped', 'Status__Lapped.0']:
            if col in df.columns:
                if (df[col] == 1).any():
                    df = df[df[col] != 1]
                    print(f"Dropped {(initial_n - len(df))} rows with {col} == 1 (lapped)")
                break
    if len(df) == 0:
        print('No rows left to score after dropping lapped drivers; aborting.')
        return
    idx = df.index

    # apply preprocessing pipeline (expect transformer that accepts dataframe)
    try:
        X = preproc.transform(df)
    except Exception as e:
        # common failure: transformer expects raw columns that are missing (we were given raw df or different schema)
        msg = str(e)
        print("Preprocessing transform failed:", msg)
        # Prefer to build the minimal FEATURES matrix directly from the raw DataFrame
        minimal_features = [
            'GridPosition','AvgQualiTime','weather_tire_cluster',
            'SOFT','MEDIUM','HARD','INTERMEDIATE','WET',
            'races_prior_this_season','Rain','PointsProp'
        ]

        provided = df.copy()
        missing = [c for c in minimal_features if c not in provided.columns]
        if not missing:
            print('Building input matrix from minimal FEATURES present in input DataFrame')
            X = provided[minimal_features].values
        else:
            print(f'Minimal features missing from input: {missing}. Attempting to impute from artifacts/X_train.csv medians')
            # try to impute missing minimal features from medians in artifacts/X_train.csv
            xtrain = ARTIFACTS / 'X_train.csv'
            if xtrain.exists():
                ref = pd.read_csv(xtrain)
                # compute medians per column safely (only when present in ref)
                for c in minimal_features:
                    if c not in provided.columns:
                        if c in ref.columns:
                            try:
                                provided[c] = ref[c].median()
                            except Exception:
                                provided[c] = 0
                        else:
                            # sensible defaults when a column isn't present in X_train
                            if c == 'Rain':
                                provided[c] = 0
                            elif c == 'PointsProp':
                                provided[c] = 0
                            elif c == 'weather_tire_cluster':
                                provided[c] = 0
                            else:
                                provided[c] = 0
                X = provided[minimal_features].values
            else:
                # final fallback: use numeric columns from provided DataFrame (wide preprocessed matrix)
                print('Could not find X_train.csv; falling back to numeric columns of provided DataFrame')
                X = provided.select_dtypes(include=[np.number]).values
                if X.shape[1] == 0:
                    raise

    # Build canonical minimal FEATURES matrix from the raw DataFrame (used when models expect the minimal inputs)
    minimal_features = [
        'GridPosition','AvgQualiTime','weather_tire_cluster',
        'SOFT','MEDIUM','HARD','INTERMEDIATE','WET',
        'races_prior_this_season','Rain','PointsProp'
    ]
    provided = df.copy()
    # if any minimal features missing, try to impute from X_train medians
    missing_min = [c for c in minimal_features if c not in provided.columns]
    if missing_min:
        xtrain = ARTIFACTS / 'X_train.csv'
        if xtrain.exists():
            ref = pd.read_csv(xtrain)
            for c in minimal_features:
                if c not in provided.columns:
                    if c in ref.columns:
                        try:
                            provided[c] = ref[c].median()
                        except Exception:
                            provided[c] = 0
                    else:
                        provided[c] = 0
        else:
            # fallback: fill missing minimal features with zeros
            for c in missing_min:
                provided[c] = 0
    # Coerce minimal features to numeric where possible. Map common string encodings for Rain -> 0/1.
    if 'Rain' in provided.columns:
        try:
            # normalize strings then map known tokens
            provided['Rain'] = provided['Rain'].astype(str).str.strip().str.lower()
            provided['Rain'] = provided['Rain'].map({
                'rain': 1,
                'yes': 1,
                'raining': 1,
                'norain': 0,
                'no': 0,
                'no rain': 0,
                'no_rain': 0,
                '0': 0,
                '1': 1
            }).where(lambda s: s.notnull(), other=None)
            # finally coerce to numeric, fill missing as 0
            provided['Rain'] = pd.to_numeric(provided['Rain'], errors='coerce').fillna(0).astype(int)
        except Exception:
            provided['Rain'] = 0
    # Ensure PointsProp and weather_tire_cluster are numeric
    for c in ('PointsProp', 'weather_tire_cluster', 'GridPosition', 'AvgQualiTime', 'races_prior_this_season'):
        if c in provided.columns:
            try:
                provided[c] = pd.to_numeric(provided[c], errors='coerce').fillna(0)
            except Exception:
                provided[c] = 0
    # Ensure tyre cluster one-hot or indicator columns are numeric
    for c in ('SOFT','MEDIUM','HARD','INTERMEDIATE','WET'):
        if c in provided.columns:
            try:
                provided[c] = pd.to_numeric(provided[c], errors='coerce').fillna(0)
            except Exception:
                provided[c] = 0

    # Finally build the minimal numpy matrix and ensure dtype is float
    minimal_X = provided[minimal_features].astype(float).values

    # predict with each model
    preds = []
    for m in models:
        try:
            # xgboost native Booster expects DMatrix
            # determine whether to use the wide transformed X or the minimal features
            use_X = X
            if hasattr(m, 'predict') and isinstance(m, xgb.core.Booster):
                # try to infer model's expected feature count
                expected = None
                try:
                    expected = m.num_features()
                except Exception:
                    try:
                        expected = m.num_features
                    except Exception:
                        expected = None
                # if expected looks like the minimal feature set, use minimal_X
                if expected is not None and expected == minimal_X.shape[1]:
                    use_X = minimal_X
                dmat = xgb.DMatrix(use_X)
                preds.append(m.predict(dmat))
            else:
                preds.append(m.predict(X))
        except Exception as ex:
            # final fallback: try DMatrix -> predict
            try:
                # validate feature count for Booster
                expected = None
                if hasattr(m, 'num_features'):
                    try:
                        expected = m.num_features()
                    except Exception:
                        try:
                            expected = m.num_features
                        except Exception:
                            expected = None
                if expected is not None and X.shape[1] != expected:
                    # try to align using columns from artifacts/X_train.csv
                    xtrain = ARTIFACTS / 'X_train.csv'
                    if xtrain.exists():
                        print(f"Model expects {expected} features but input has {X.shape[1]}; attempting to align using X_train.csv columns")
                        df_ref = pd.read_csv(xtrain)
                        # select numeric columns from df_ref and map by order
                        ref_cols = df_ref.select_dtypes(include=[np.number]).columns.tolist()
                        # Prefer a canonical minimal FEATURES ordering used during training
                        # If available, use columns_minimal.json; otherwise fall back to the known minimal features
                        col_min = ARTIFACTS / 'columns_minimal.json'
                        if col_min.exists():
                            try:
                                with open(col_min, 'r') as fh:
                                    ref_cols = json.load(fh)
                                    # filter to numeric columns that are present in df_ref
                                    ref_cols = [c for c in ref_cols if c in df_ref.columns]
                            except Exception:
                                pass
                        # fallback: use the conservative minimal features used by the randomized CV script
                        if not ref_cols or len(ref_cols) < 5:
                            ref_cols = [
                                'GridPosition','AvgQualiTime','weather_tire_cluster',
                                'SOFT','MEDIUM','HARD','INTERMEDIATE','WET',
                                'races_prior_this_season','Rain','PointsProp'
                            ]
                        # take the intersection with provided df numeric columns
                        provided_numeric = df.select_dtypes(include=[np.number])
                        common = [c for c in ref_cols if c in provided_numeric.columns]
                        if len(common) == expected:
                            X = provided_numeric[common].values
                        else:
                            print('Could not auto-align features; raising')
                            raise
                dmat = xgb.DMatrix(X)
                preds.append(m.predict(dmat))
            except Exception:
                raise
    preds = np.vstack(preds)
    avg = preds.mean(axis=0)

    # Save uncalibrated predictions (keep original index so we can align with truth)
    uncal_out = pd.DataFrame({"prediction": avg}, index=idx)
    uncal_out.reset_index(inplace=True)
    unc_path = ARTIFACTS / 'scored_preds_from_raw_uncalibrated.csv'
    uncal_out.to_csv(unc_path, index=False)
    print(f"Wrote uncalibrated predictions to {unc_path}")

    # calibration
    calibrated = apply_calibration(avg)

    out = pd.DataFrame({"prediction": calibrated}, index=idx)
    out.reset_index(inplace=True)
    out.to_csv(output_csv, index=False)
    print(f"Wrote calibrated predictions to {output_csv}")

    # compute metrics versus DeviationFromAvg_s if present in the original df
    metrics = []
    if 'DeviationFromAvg_s' in df.columns:
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            # Build prediction Series indexed by the original DataFrame index so we can align robustly
            preds_unc_series = pd.Series(avg, index=idx)
            preds_cal_series = pd.Series(calibrated, index=idx)

            # truth series, aligned to same index
            y = df.loc[idx, 'DeviationFromAvg_s'].astype(float)
            # select rows where truth is present
            valid_idx = y.dropna().index
            if len(valid_idx) > 0:
                y_valid = y.loc[valid_idx]
                preds_unc_aligned = preds_unc_series.reindex(valid_idx)
                preds_cal_aligned = preds_cal_series.reindex(valid_idx)

                import math
                mse_unc = mean_squared_error(y_valid, preds_unc_aligned)
                rmse_unc = math.sqrt(mse_unc)
                mae_unc = mean_absolute_error(y_valid, preds_unc_aligned)
                r2_unc = r2_score(y_valid, preds_unc_aligned)

                mse_cal = mean_squared_error(y_valid, preds_cal_aligned)
                rmse_cal = math.sqrt(mse_cal)
                mae_cal = mean_absolute_error(y_valid, preds_cal_aligned)
                r2_cal = r2_score(y_valid, preds_cal_aligned)

                metrics.append({
                    'type': 'uncalibrated',
                    'n': int(len(valid_idx)),
                    'mse': float(mse_unc),
                    'rmse': float(rmse_unc),
                    'mae': float(mae_unc),
                    'r2': float(r2_unc)
                })
                metrics.append({
                    'type': 'calibrated',
                    'n': int(len(valid_idx)),
                    'mse': float(mse_cal),
                    'rmse': float(rmse_cal),
                    'mae': float(mae_cal),
                    'r2': float(r2_cal)
                })
                # write separate CSVs for calibrated and uncalibrated
                pd.DataFrame([metrics[0]]).to_csv(ARTIFACTS / 'metrics_scored_from_raw_uncalibrated.csv', index=False)
                pd.DataFrame([metrics[1]]).to_csv(ARTIFACTS / 'metrics_scored_from_raw_calibrated.csv', index=False)
                print('Wrote metrics to artifacts/metrics_scored_from_raw_{uncalibrated,calibrated}.csv')
        except Exception as e:
            import traceback
            print('Failed computing metrics:', e)
            print(traceback.format_exc())
            # attempt to write placeholder metric files so callers can detect failure
            try:
                nrows = int(len(valid_idx)) if 'valid_idx' in locals() else 0
            except Exception:
                nrows = 0
            placeholder_unc = {'type': 'uncalibrated', 'n': nrows, 'mse': None, 'rmse': None, 'mae': None, 'r2': None, 'error': str(e)}
            placeholder_cal = {'type': 'calibrated', 'n': nrows, 'mse': None, 'rmse': None, 'mae': None, 'r2': None, 'error': str(e)}
            try:
                pd.DataFrame([placeholder_unc]).to_csv(ARTIFACTS / 'metrics_scored_from_raw_uncalibrated.csv', index=False)
                pd.DataFrame([placeholder_cal]).to_csv(ARTIFACTS / 'metrics_scored_from_raw_calibrated.csv', index=False)
            except Exception:
                pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='CSV with raw input rows to score')
    ap.add_argument('--output', default=str(ARTIFACTS / 'scored_preds.csv'))
    args = ap.parse_args()
    score(Path(args.input), Path(args.output))
