#!/usr/bin/env bash
set -euo pipefail

# Video-MME
hf download lmms-lab/Video-MME --repo-type dataset

# LongVideoBench
hf download longvideobench/LongVideoBench --repo-type dataset

# MLVU
hf download sy1998/MLVU_dev --repo-type dataset

# MMBench-Video
hf download opencompass/MMBench-Video --repo-type dataset

# Pre-cache datasets for offline evaluation.
# datasets.load_dataset() uses a different cache format than `hf download`.
# Running it once online builds the index so offline mode works later.
python - <<'PY'
import datasets

for name in [
    'lmms-lab/Video-MME',
    'longvideobench/LongVideoBench',
    'sy1998/MLVU_dev',
    'opencompass/MMBench-Video',
]:
    print(f'Pre-caching {name}...')
    try:
        datasets.load_dataset(name, trust_remote_code=True)
        print(f'  Done: {name}')
    except Exception as e:
        print(f'  Warning: {name}: {e}')
PY
