#!/usr/bin/env python3
"""Apply category_mappings.json to a CSV, mapping rare values to 'OTHER'.

Usage:
  python artifacts/apply_category_mappings.py --in premodeldatav1.csv --out artifacts/premodel_mapped.csv
"""
import argparse
import json
from pathlib import Path
import pandas as pd

ART = Path('artifacts')
MAPPING = ART / 'category_mappings.json'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='infile', default='premodeldatav1.csv')
    p.add_argument('--out', dest='outfile', default=str(ART / 'premodel_mapped.csv'))
    args = p.parse_args()

    if not MAPPING.exists():
        print('No mapping file found at', MAPPING)
        return

    with open(MAPPING) as f:
        mappings = json.load(f)

    df = pd.read_csv(args.infile)
    replaced = {}
    for col, rare_vals in mappings.items():
        if col not in df.columns:
            continue
        # fillna with 'missing' so mapping includes missing too
        df[col] = df[col].fillna('missing')
        mask = df[col].isin(rare_vals)
        replaced[col] = int(mask.sum())
        if mask.any():
            df.loc[mask, col] = 'OTHER'

    outp = Path(args.outfile)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    print('Wrote mapped CSV to', outp)
    print('Replacements (col:count):')
    for c, n in replaced.items():
        print(f'  {c}: {n}')

if __name__ == '__main__':
    main()
