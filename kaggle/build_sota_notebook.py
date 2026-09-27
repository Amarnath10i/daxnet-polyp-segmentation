"""Build ONE self-contained, readable Kaggle notebook that trains and tests DAXNet
(PVTv2-b2 encoder, 352 px, presence gate) on Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB
and PolypGen, and compares every number with the BAMPolyp paper and the best value
in its Table 3.

The DAXNet source is written into the notebook as %%writefile cells (generated from
daxnet/*.py, so the notebook always matches the repository).

    python kaggle/build_sota_notebook.py [--seed 1] [--slug daxnet-sota-all-datasets]
    PYTHONUTF8=1 python kaggle/push_when_free.py daxnet-sota-all-datasets
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODULES = ['__init__.py', 'ssd_axial.py', 'data.py', 'losses.py', 'metrics.py', 'model.py', 'train.py']
DATASETS = ["debeshjha1/kvasirseg", "balraj98/cvcclinicdb", "longvil/cvc-colondb", "kokoroou/polypgen2021"]

INTRO = """# DAXNet: polyp segmentation on Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB and PolypGen

One notebook: builds DAXNet, trains and tests it on all four datasets (two T4 GPUs in
parallel), and compares every result with the BAMPolyp paper (ICCVW 2025).

**Model.** EfficientNet/PVT-style U-Net whose bottleneck and decoder mix context with an exact,
bidirectional, edge-aware (geodesic) axial state-space operator; an image-level presence head
empties the mask of frames judged polyp-free. Encoder PVTv2-b2 (ImageNet), input 352 px.

**Protocol (identical to BAMPolyp).** Per-dataset split `np.random.seed(42)` permutation,
80 / 10 / 10 train / val / test. Metric: per-image Dice and IoU at threshold 0.5, averaged.
The checkpoint with the best validation IoU is kept.
* **Protocol A** = validation score of that checkpoint: the number BAMPolyp reports.
* **Protocol B** = the held-out 10 % test split at the same checkpoint (never used for selection).
* `+TTA` = the same checkpoint with horizontal/vertical flip averaging.

Numbers of methods other than BAMPolyp in its Table 3 come from their own papers and may use
other splits (the BAMPolyp paper says so); they are listed only as reference points.
"""

SETUP = """!nvidia-smi -L
import os, subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'einops', 'timm'], check=False)
WORK, DATA = '/kaggle/working', '/kaggle/input'
os.chdir(WORK)
os.makedirs('daxnet', exist_ok=True)
import torch, timm
print('torch', torch.__version__, '| timm', timm.__version__, '| GPUs', torch.cuda.device_count())
print('datasets:', sorted(os.listdir(DATA)))
"""

CONFIG = """import json, time
ENCODER, SIZE = 'pvt_v2_b2', 352
SEED = {seed}                       # main run: all four datasets
EXTRA_SEEDS = {extra}               # extra replicates on the three small datasets, only if time is left
EPOCHS = {{'polypgen': 100}}         # others: 200 (PolypGen's training set is 3.8x Kvasir's)
EST_H = {{'polypgen': 5.8, 'kvasir': 3.0, 'clinicdb': 1.9, 'colondb': 1.7}}   # measured on 2x T4

def job(ds, seed):
    cfg = json.dumps({{'encoder': ENCODER, 'presence': True}})
    args = (f"--model daxnet --recipe ours --protocol bam --dataset {{ds}} --size {{SIZE}} --seed {{seed}} "
            + (f"--epochs {{EPOCHS[ds]}} " if ds in EPOCHS else '') + f"--cfg '{{cfg}}'")
    return dict(name=f'dax_{{ENCODER}}_{{SIZE}}_{{ds}}_s{{seed}}', dataset=ds, seed=seed, args=args, est_h=EST_H[ds])

JOBS = [job(ds, SEED) for ds in ('polypgen', 'kvasir', 'clinicdb', 'colondb')]
JOBS += [job(ds, s) for s in EXTRA_SEEDS for ds in ('kvasir', 'clinicdb', 'colondb')]
START_BY_H, FINISH_BY_H = 10.5, 11.5   # no job starts after 10.5 h; running jobs finalize before 11.5 h
for j in JOBS: print(f"{{j['name']:40s}} ~{{j['est_h']}} h")

# download the ImageNet weights once, before the parallel workers start
timm.create_model(ENCODER, pretrained=True, features_only=True);
"""

RUNNER = """import queue, shlex, threading
T0 = time.time()
N_GPU = max(1, torch.cuda.device_count())
q = queue.Queue()
for j in JOBS: q.put(j)
os.makedirs(f'{WORK}/runs', exist_ok=True)

def worker(gpu):
    while True:
        try: j = q.get_nowait()
        except queue.Empty: return
        el_h = (time.time() - T0) / 3600
        left_h = FINISH_BY_H - el_h
        if el_h > START_BY_H or j['est_h'] > left_h:
            print(f"[gpu{gpu}] SKIP {j['name']} ({left_h:.1f} h left, needs ~{j['est_h']} h)", flush=True); continue
        out = f"{WORK}/runs/{j['name']}"
        if os.path.exists(f'{out}/final.json'): continue
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, '-m', 'daxnet.train', *shlex.split(j['args']), '--data-root', DATA, '--out', out,
               '--workers', '3', '--max-hours', f'{max(0.1, left_h - 0.15):.2f}']
        t = time.time()
        print(f"[gpu{gpu}] START {j['name']}  (t = {el_h:.2f} h)", flush=True)
        with open(f'{out}/stdout.txt', 'w') as f:
            rc = subprocess.run(cmd, cwd=WORK, stdout=f, stderr=subprocess.STDOUT,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONWARNINGS='ignore')).returncode
        msg = ''
        if os.path.exists(f'{out}/final.json'):
            r = json.load(open(f'{out}/final.json'))
            msg = (f"val {r['val']['dice_net']:.4f}/{r['val']['iou_net']:.4f}  "
                   f"test {r['test']['dice_net']:.4f}/{r['test']['iou_net']:.4f}  best epoch {r['best_epoch']}")
        else:
            msg = 'FAILED: ' + ' | '.join(open(f'{out}/stdout.txt').read().splitlines()[-3:])[:800]
        print(f"[gpu{gpu}] DONE  {j['name']} rc={rc} {(time.time() - t) / 60:.0f} min  {msg}", flush=True)

ths = [threading.Thread(target=worker, args=(g,)) for g in range(N_GPU)]
[t.start() for t in ths]; [t.join() for t in ths]
print(f'all jobs finished in {(time.time() - T0) / 3600:.2f} h')
"""

RESULTS = """import glob
import numpy as np
import pandas as pd
PAPER = {'kvasir': (0.9380, 0.8881), 'clinicdb': (0.9437, 0.8939),
         'colondb': (0.9255, 0.8659), 'polypgen': (0.8683, 0.8211)}
# best value per column of BAMPolyp's Table 3 (Dice, IoU) and who reported it
TABLE3_BEST = {'kvasir': ((0.9380, 'BAMPolyp'), (0.8910, 'SwinE-Net')),
               'clinicdb': ((0.9521, 'PVT-EMCAD-B2'), (0.9238, 'LGPS')),
               'colondb': ((0.9255, 'BAMPolyp'), (0.8659, 'BAMPolyp')),
               'polypgen': ((0.8683, 'BAMPolyp'), (0.8211, 'BAMPolyp'))}
rows = []
for f in sorted(glob.glob(f'{WORK}/runs/*/final.json')):
    r = json.load(open(f)); m = json.load(open(f.replace('final.json', 'meta.json')))
    row = dict(dataset=m['dataset'], seed=m['seed'], best_epoch=r['best_epoch'], hours=round(r['train_hours'], 2),
               params_M=round(r['params_M'], 2))
    for k, name in (('val', 'A'), ('val_tta', 'A+TTA'), ('test', 'B'), ('test_tta', 'B+TTA')):
        row[f'Dice {name}'], row[f'IoU {name}'] = r[k]['dice_net'], r[k]['iou_net']
    rows.append(row)
assert rows, 'no run finished: see the FAILED lines of the previous cell'
df = pd.DataFrame(rows).sort_values(['dataset', 'seed'])
df.to_csv(f'{WORK}/results_per_run.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.precision', 4)
print('Per run:'); display(df)

mark = lambda v, ref: f"{v:.4f} {'✅' if v > ref else '❌'}"
table = []
for ds in ('kvasir', 'clinicdb', 'colondb', 'polypgen'):
    d = df[df.dataset == ds]
    if d.empty: continue
    (pd_, pi), ((t3d, t3dn), (t3i, t3in)) = PAPER[ds], TABLE3_BEST[ds]
    a_d, a_i = d['Dice A'].mean(), d['IoU A'].mean()
    table.append({'dataset': ds, 'seeds': len(d),
                  'BAMPolyp paper Dice/IoU': f'{pd_:.4f} / {pi:.4f}',
                  'DAXNet Dice (Prot. A)': mark(a_d, pd_), 'DAXNet IoU (Prot. A)': mark(a_i, pi),
                  'Table-3 best Dice': f'{t3d:.4f} ({t3dn})', 'vs best Dice': '✅' if a_d > t3d else '❌',
                  'Table-3 best IoU': f'{t3i:.4f} ({t3in})', 'vs best IoU': '✅' if a_i > t3i else '❌',
                  'DAXNet test Dice/IoU (Prot. B)': f"{d['Dice B'].mean():.4f} / {d['IoU B'].mean():.4f}",
                  'test + TTA': f"{d['Dice B+TTA'].mean():.4f} / {d['IoU B+TTA'].mean():.4f}"})
summary = pd.DataFrame(table)
summary.to_csv(f'{WORK}/results_vs_sota.csv', index=False)
print('Mean over seeds, compared with the paper (✅ = higher):'); display(summary)
json.dump({'per_run': rows, 'summary': table}, open(f'{WORK}/results.json', 'w'), indent=1)
"""

CURVES = """import matplotlib.pyplot as plt
runs = sorted(glob.glob(f'{WORK}/runs/*/log.jsonl'))
fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.4), squeeze=False)
for ax, f in zip(axes[0], runs):
    log = [json.loads(l) for l in open(f) if l.strip()]
    log = [r for r in log if 'val' in r]
    ep = [r['epoch'] for r in log]
    ax.plot(ep, [r['val']['dice_net'] for r in log], label='val Dice')
    ax.plot(ep, [r['test']['dice_net'] for r in log], label='test Dice')
    ax.set_title(f.split('/')[-2].replace('dax_', ''), fontsize=9); ax.set_ylim(0.7, 1.0); ax.grid(alpha=.3)
axes[0][0].legend(); plt.tight_layout(); plt.savefig(f'{WORK}/curves.png', dpi=110); plt.show()
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--extra-seeds', default='2', help='comma-separated; empty for none')
    ap.add_argument('--slug', default='daxnet-sota-all-datasets')
    ap.add_argument('--title', default='DAXNet SOTA all datasets')
    ap.add_argument('--user', default='amarnathmadaka')
    a = ap.parse_args()
    extra = [int(s) for s in a.extra_seeds.split(',') if s.strip()]

    md = lambda s: dict(cell_type='markdown', metadata={}, source=s)
    code = lambda s: dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s)
    cells = [md(INTRO), md('## 1. Setup'), code(SETUP),
             md('## 2. DAXNet source\nWritten to `daxnet/` so the training script can run on both GPUs in parallel.')]
    for fn in MODULES:
        src = open(os.path.join(ROOT, 'daxnet', fn), encoding='utf-8').read()
        cells.append(code(f'%%writefile daxnet/{fn}\n' + src))
    cells += [md('## 3. Experiment configuration'), code(CONFIG.format(seed=a.seed, extra=extra)),
              md('## 4. Train and test on all datasets (2 GPUs in parallel)'), code(RUNNER),
              md('## 5. Results vs. BAMPolyp and its Table 3'), code(RESULTS),
              md('## 6. Validation / test curves'), code(CURVES)]
    for i, c in enumerate(cells):
        c['id'] = f'c{i}'
    nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
              metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                            language_info=dict(name='python')))
    out_dir = os.path.join(HERE, 'build', a.slug)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(nb, open(os.path.join(out_dir, f'{a.slug}.ipynb'), 'w'), indent=1)
    meta = dict(id=f'{a.user}/{a.slug}', title=a.title, code_file=f'{a.slug}.ipynb', language='python',
                kernel_type='notebook', is_private=True, enable_gpu=True, enable_internet=True,
                machine_shape='NvidiaTeslaT4', dataset_sources=DATASETS, competition_sources=[], kernel_sources=[])
    json.dump(meta, open(os.path.join(out_dir, 'kernel-metadata.json'), 'w'), indent=1)
    print(f'wrote {out_dir} ({len(cells)} cells, seed {a.seed}, extra seeds {extra})')


if __name__ == '__main__':
    main()
