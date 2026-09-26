"""Build a self-contained Kaggle notebook that runs a list of DAXNet jobs on 2 GPUs.

The `daxnet/` package is embedded as a base64 tarball (its sha256 is printed in
the notebook), so every run is tied to an exact code snapshot without a remote.

    python kaggle/build_notebook.py --jobs kaggle/jobs/smoke.json --slug daxnet-smoke
    python -c "from kaggle.api.kaggle_api_extended import KaggleApi as K; a=K(); a.authenticate(); a.kernels_push('kaggle/build/daxnet-smoke')"
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import re
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATASETS = ["debeshjha1/kvasirseg", "balraj98/cvcclinicdb", "longvil/cvc-colondb",
            "kokoroou/polypgen2021", "kholanaseem/pranet-dataset"]


def pack():
    """Deterministic tar.gz of daxnet/*.py: the sha depends only on paths and contents."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for dp, dns, fns in sorted(os.walk(os.path.join(ROOT, "daxnet"))):
            dns[:] = sorted(d for d in dns if d != "__pycache__")
            for fn in sorted(fns):
                if fn.endswith(".py"):
                    p = os.path.join(dp, fn)
                    data = open(p, "rb").read()
                    info = tarfile.TarInfo(os.path.relpath(p, ROOT).replace(os.sep, "/"))
                    info.size, info.mode = len(data), 0o644
                    tar.addfile(info, io.BytesIO(data))
    raw = buf.getvalue()
    return base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()[:12]


RUNNER = r'''
import json, os, queue, subprocess, threading, time, glob
JOBS = json.loads(%r)
N_GPU = max(1, int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1))
DEADLINE = time.time() + %f * 3600          # no job starts after this
HARD_END = time.time() + %f * 3600          # every job finalizes before this (Kaggle kills the session at 12 h)
q = queue.Queue()
for j in JOBS: q.put(j)
os.makedirs('/kaggle/working/runs', exist_ok=True)

def worker(gpu):
    while True:
        try: j = q.get_nowait()
        except queue.Empty: return
        left_h = (HARD_END - time.time()) / 3600
        if time.time() > DEADLINE or j.get('est_h', 0) > left_h:
            print(f'[gpu{gpu}] SKIP (deadline, {left_h:.1f} h left, est {j.get("est_h", "?")} h) {j["name"]}', flush=True); continue
        out = f'/kaggle/working/runs/{j["name"]}'
        if os.path.exists(f'{out}/final.json'):
            continue
        os.makedirs(out, exist_ok=True)
        cmd = (f'python -m daxnet.train {j["args"]} --data-root /kaggle/input --out {out} --workers %d'
               f' --max-hours {max(0.1, left_h - 0.15):.2f}')
        t = time.time()
        print(f'[gpu{gpu}] START {j["name"]}: {cmd}', flush=True)
        with open(f'{out}/stdout.txt', 'w') as f:
            rc = subprocess.run(cmd, shell=True, cwd='/kaggle/working', stdout=f, stderr=subprocess.STDOUT,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONWARNINGS='ignore')).returncode
        tail = open(f'{out}/stdout.txt').read().splitlines()[-3:]
        print(f'[gpu{gpu}] DONE {j["name"]} rc={rc} {(time.time()-t)/60:.1f} min\n    ' + '\n    '.join(l[:600] for l in tail), flush=True)

ths = [threading.Thread(target=worker, args=(g,)) for g in range(N_GPU)]
[t.start() for t in ths]; [t.join() for t in ths]
'''

SUMMARY = r'''
import json, glob, os
rows = {}
for f in sorted(glob.glob('/kaggle/working/runs/*/final.json')):
    r = json.load(open(f)); name = f.split('/')[-2]
    rows[name] = r
    di = lambda k: f"{r[k]['dice_net']:.4f}/{r[k]['iou_net']:.4f}" if k in r else '-'
    splits = [k for k, v in r.items() if isinstance(v, dict) and not k.endswith(('_raw', '_tta'))]
    print(f"{name:34s} ep {r['best_epoch']}/{r.get('epochs_run')} {r['train_hours']:.2f}h " +
          ' '.join(f"| {k} {di(k)} tta {di(k + '_tta')} raw {di(k + '_raw')}" for k in splits))
json.dump(rows, open('/kaggle/working/summary.json', 'w'), indent=1)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--title")
    ap.add_argument("--hours", type=float, default=10.5, help="no new job starts after this")
    ap.add_argument("--hard-hours", type=float, default=11.5, help="running jobs stop and finalize before this")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--user", default="amarnathmadaka")
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    b64, sha = pack()
    # timm encoders named in the jobs' --cfg: download their weights once, before the workers start
    timm_encoders = sorted({e for j in jobs for e in re.findall(r'"encoder":\s*"([\w.]+)"', j["args"])} - {"effb0"})
    cells = [
        "!nvidia-smi -L\n"
        "import base64, io, tarfile, subprocess, sys\n"
        f"CODE_SHA = {sha!r}\n"
        f"tarfile.open(fileobj=io.BytesIO(base64.b64decode({b64!r})), mode='r:gz').extractall('/kaggle/working', filter='data')\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'einops', 'timm'], check=False)\n"
        "import torch, torchvision, timm; print('code', CODE_SHA, 'torch', torch.__version__, torchvision.__version__, 'timm', timm.__version__)\n"
        "from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights\n"
        "efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)  # cache weights once\n"
        f"for name in {timm_encoders!r}:\n"
        "    timm.create_model(name, pretrained=True, features_only=True)\n",
        RUNNER % (json.dumps(jobs), a.hours, a.hard_hours, a.workers),
        SUMMARY,
    ]
    out_dir = os.path.join(HERE, "build", a.slug)
    os.makedirs(out_dir, exist_ok=True)
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                            language_info=dict(name="python")),
              cells=[dict(cell_type="code", metadata={}, execution_count=None, outputs=[], id=f"c{i}",
                          source=src) for i, src in enumerate(cells)])
    json.dump(nb, open(os.path.join(out_dir, f"{a.slug}.ipynb"), "w"), indent=1)
    meta = dict(id=f"{a.user}/{a.slug}", title=a.title or a.slug, code_file=f"{a.slug}.ipynb",
                language="python", kernel_type="notebook", is_private=True, enable_gpu=True,
                enable_internet=True, machine_shape="NvidiaTeslaT4", dataset_sources=DATASETS,
                competition_sources=[], kernel_sources=[])
    json.dump(meta, open(os.path.join(out_dir, "kernel-metadata.json"), "w"), indent=1)
    print(f"wrote {out_dir} ({len(jobs)} jobs, code {sha})")


if __name__ == "__main__":
    main()
