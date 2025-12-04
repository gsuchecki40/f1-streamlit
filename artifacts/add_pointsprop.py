"""Compute PointsProp (cumulative points earned up to each race / maximum possible points to date).

Rules/assumptions:
- Maximum points per race = 25 (standard F1 points for a win). If races_prior_this_season is 0, PointsProp=0.
- We compute cumulative points for each driver within the season up to but NOT including the current row's round. If the dataset already includes Points that are the points for that race, we'll sum prior rows.
"""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CSV_PATH = os.path.join(ROOT, 'premodeldatav1.csv')
BACKUP_PATH = os.path.join(ROOT, 'premodeldatav1.csv.bak')

def compute_pointsprop(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure Points and season/round exist
    if 'Points' not in df.columns:
        raise ValueError('Input CSV missing Points column')
    if 'Season' not in df.columns or 'Round' not in df.columns:
        raise ValueError('Input CSV missing Season or Round columns')

    df = df.copy()
    # Convert Points to numeric
    df['Points'] = pd.to_numeric(df['Points'], errors='coerce').fillna(0)

    # Sort so cumulative sums make sense
    df['Round'] = pd.to_numeric(df['Round'], errors='coerce')
    df.sort_values(['Season', 'DriverId', 'Round'], inplace=True)

    # Compute cumulative points up to previous race for each driver-season
    df['cum_points_prior'] = df.groupby(['Season', 'DriverId'])['Points'].cumsum() - df['Points']

    # Maximum possible points up to previous round = races_prior_this_season * 25
    if 'races_prior_this_season' not in df.columns:
        # fallback: compute as (round - 1)
        df['races_prior_this_season'] = df['Round'] - 1

    df['races_prior_this_season'] = pd.to_numeric(df['races_prior_this_season'], errors='coerce').fillna(0)
    df['max_possible_prior'] = df['races_prior_this_season'] * 25.0

    # Avoid divide by zero
    df['PointsProp'] = 0.0
    mask = df['max_possible_prior'] > 0
    df.loc[mask, 'PointsProp'] = df.loc[mask, 'cum_points_prior'] / df.loc[mask, 'max_possible_prior']

    # Clip to [0,1]
    df['PointsProp'] = df['PointsProp'].clip(lower=0.0, upper=1.0)

    # Drop helper cols
    df.drop(columns=['cum_points_prior', 'max_possible_prior'], inplace=True)
    return df

def main():
    print('Reading', CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    # Backup
    if not os.path.exists(BACKUP_PATH):
        print('Writing backup to', BACKUP_PATH)
        df.to_csv(BACKUP_PATH, index=False)
    else:
        print('Backup already exists at', BACKUP_PATH)

    df2 = compute_pointsprop(df)
    if 'PointsProp' in df.columns:
        print('PointsProp column already existed; it will be overwritten')

    print('Writing updated CSV with PointsProp to', CSV_PATH)
    df2.to_csv(CSV_PATH, index=False)
    print('Done')

if __name__ == '__main__':
    main()
