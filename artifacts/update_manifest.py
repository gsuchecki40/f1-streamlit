#!/usr/bin/env python3
"""Create or update a richer artifact manifest (artifacts/manifest.json).

Scans common artifact files and records paths, modified timestamps, file sizes,
and SHA256 hashes. Also attempts to include the current git short commit and a
dirty flag (if the repository is available). Designed to be safe in CI or
locally; missing information is omitted silently.
"""
import json
from pathlib import Path
import time
import hashlib
import subprocess

ART = Path(__file__).resolve().parents[1] / 'artifacts'
ROOT = Path(__file__).resolve().parents[2]


def sha256_of(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def git_info():
    info = {}
    try:
        # short commit
        info['commit'] = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT).decode().strip()
        # dirty flag
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).decode().strip()
        info['dirty'] = bool(status)
    except Exception:
        # not a git repo or git not available
        pass
    return info


def scan_artifacts():
    patterns = ['*.joblib', '*.csv', '*.png', '*.json']
    entries = {}
    for pat in patterns:
        for p in sorted(ART.glob(pat)):
            try:
                stat = p.stat()
                entries[p.name] = {
                    'path': str(p),
                    'mtime': stat.st_mtime,
                    'size': stat.st_size,
                    'sha256': sha256_of(p)
                }
            except Exception:
                continue
    manifest = {
        'generated_at': time.time(),
        'items': entries
    }
    # add git info if possible
    g = git_info()
    if g:
        manifest['git'] = g

    out = ART / 'manifest.json'
    out.write_text(json.dumps(manifest, indent=2))
    print('Wrote', out)


if __name__ == '__main__':
    scan_artifacts()
