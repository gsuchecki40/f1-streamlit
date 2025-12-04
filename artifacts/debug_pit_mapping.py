#!/usr/bin/env python3
import fastf1
import pandas as pd
import sys

fastf1.Cache.enable_cache('f1_cache')

season = 2023
round_n = 1
print(f'Loading session Season={season} Round={round_n}')
session = fastf1.get_session(season, round_n, 'R')
session.load()
laps = session.laps

print('laps columns:', laps.columns.tolist())
print('laps sample rows:')
print(laps.head(10))

if 'PitTime' in laps.columns:
    print('PitTime present; non-null count:', laps['PitTime'].notna().sum())
    print('Unique Driver values sample:', laps['Driver'].unique()[:20])
    if 'DriverNumber' in laps.columns:
        print('DriverNumber unique sample:', laps['DriverNumber'].unique()[:20])
    print('\nGrouped by Driver mean PitTime:')
    try:
        print(laps[laps['PitTime'].notna()].groupby('Driver')['PitTime'].mean().head(20))
    except Exception as e:
        print('Group by driver failed:', e)

if 'PitInTime' in laps.columns and 'PitOutTime' in laps.columns:
    pit_df = laps[laps['PitInTime'].notna() & laps['PitOutTime'].notna()].copy()
    pit_df['PitInTime_dt'] = pd.to_datetime(pit_df['PitInTime'])
    pit_df['PitOutTime_dt'] = pd.to_datetime(pit_df['PitOutTime'])
    pit_df['StopSeconds'] = (pit_df['PitOutTime_dt'] - pit_df['PitInTime_dt']).dt.total_seconds()
    print('Computed StopSeconds, sample:')
    print(pit_df[['Driver','DriverNumber','StopSeconds']].head(20))

print('Done debug')
