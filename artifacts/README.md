Artifacts README

This folder contains model artifacts and diagnostic outputs produced by the training pipeline.

Important files used for scoring:
- `preprocessing_pipeline.joblib` - sklearn-style transformer for converting raw CSV rows into model features.
- `ensemble_fold_models/` - saved fold models (joblib). If present, `score_model.py` will average their predictions.
- `xgb_minimal_cv_randomized.joblib` - fallback single model.
- `linear_calibration.joblib` - optional scikit-learn regressor used to calibrate averaged predictions.

Quick scoring example (from project root):

```bash
python3 artifacts/score_model.py --input path/to/new_rows.csv --output artifacts/scored_preds.csv
```

The input CSV should contain the same raw columns that were used to build `premodel_canonical.csv` (Season, Round, DriverId, GridPosition, etc.). The `preprocessing_pipeline.joblib` expects a pandas DataFrame with those columns.

If you want a self-contained Docker/virtualenv environment, install the pinned requirements in `requirements.txt`.
