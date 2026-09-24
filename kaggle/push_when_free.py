"""Retry pushing a built kernel until a GPU session slot is free.
    PYTHONUTF8=1 python kaggle/push_when_free.py daxnet-main-b [every_s]"""
import sys, time
from kaggle.api.kaggle_api_extended import KaggleApi
slug = sys.argv[1]; every = int(sys.argv[2]) if len(sys.argv) > 2 else 600
api = KaggleApi(); api.authenticate()
while True:
    r = api.kernels_push(f'kaggle/build/{slug}')
    err = getattr(r, 'error', None)
    print(time.strftime('%H:%M:%S'), slug, 'error:', err, flush=True)
    if not err:
        break
    if 'Maximum' not in str(err):
        sys.exit(1)
    time.sleep(every)
