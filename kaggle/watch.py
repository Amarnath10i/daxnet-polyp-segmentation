"""Poll a kernel until it stops, then download its outputs to kaggle/results/<slug>/.
    python kaggle/watch.py daxnet-smoke [--every 60]"""
import sys, time, os, json  # run with PYTHONUTF8=1 on Windows (kernel logs contain non-cp1252 chars)
from kaggle.api.kaggle_api_extended import KaggleApi
slug = sys.argv[1]; every = int(sys.argv[3]) if len(sys.argv) > 3 else 60
api = KaggleApi(); api.authenticate()
ref = f"amarnathmadaka/{slug}"
last = None
while True:
    try:
        st = api.kernels_status(ref)
        s = str(getattr(st, 'status', st)).split('.')[-1].lower()
    except Exception as e:                  # transient API error, or kernel not pushed yet: keep polling
        s = f'poll-failed {str(e)[:80]}'
    if s != last:
        print(time.strftime('%H:%M:%S'), slug, s, flush=True); last = s
    if not s.startswith('poll-failed') and any(k in s for k in ('complete', 'error', 'cancel')):
        break
    time.sleep(every)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', slug)
os.makedirs(out, exist_ok=True)
# everything except checkpoints (*.pt: up to ~115 MB each), all pages
token, n = None, 0
while True:
    for attempt in range(5):
        try:
            files, token = api.kernels_output(ref, path=out, file_pattern=r'^(?!.*\.pt$).*', page_size=200,
                                              page_token=token)
            break
        except Exception as e:
            print('download retry', attempt, str(e)[:80], flush=True); time.sleep(30)
    else:
        sys.exit('download failed')
    n += len(files)
    if not token:
        break
print('downloaded', n, 'files to', out, flush=True)
