"""Small helpers to normalize raw input DataFrames before any numeric imputation.

This centralizes token->numeric mapping for `Rain` and ensures other minimal features
are coerced to numeric types safely.
"""
from typing import List
import pandas as pd
import numpy as np


def normalize_minimal_features(df: pd.DataFrame, minimal_features: List[str] = None) -> pd.DataFrame:
    if minimal_features is None:
        minimal_features = [
            'GridPosition', 'AvgQualiTime', 'weather_tire_cluster',
            'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET',
            'races_prior_this_season', 'Rain', 'PointsProp'
        ]
    df = df.copy()
    # Normalize Rain -> numeric 0/1
    if 'Rain' in df.columns:
        try:
            s = df['Rain'].astype(str).str.strip().str.lower()
            mapping = {
                'rain': 1,
                'yes': 1,
                'raining': 1,
                'norain': 0,
                'no': 0,
                'no rain': 0,
                'no_rain': 0,
                '0': 0,
                '1': 1,
                'false': 0,
                'true': 1,
            }
            # also handle tokens like 'light rain' or 'heavy_rain' by checking substring
            mapped = s.map(mapping)
            # substring heuristics
            mapped = mapped.where(mapped.notnull(), other=s.apply(lambda t: 1 if 'rain' in t else None))
            df['Rain'] = pd.to_numeric(mapped, errors='coerce').fillna(0).astype(int)
        except Exception:
            df['Rain'] = 0

    # Force-numeric for common minimal features, filling NaN with zeros
    for c in ('PointsProp', 'weather_tire_cluster', 'GridPosition', 'AvgQualiTime', 'races_prior_this_season'):
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            except Exception:
                df[c] = 0

    # Tyre flags
    for c in ('SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'):
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            except Exception:
                df[c] = 0

    return df
