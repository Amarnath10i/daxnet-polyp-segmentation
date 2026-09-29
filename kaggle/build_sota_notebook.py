"""Build ONE self-contained, readable Kaggle notebook that trains and tests DAXNet
(PVTv2-b2 encoder, 352 px, presence gate) on Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB
and PolypGen over several seeds, saves named checkpoints, re-tests every checkpoint,
and prints the final comparison with the BAMPolyp paper as plain-text tables (so the
results are readable in the Kaggle log as well as in the notebook).

The DAXNet source is written into the notebook as %%writefile cells (generated from
daxnet/*.py, so the notebook always matches the repository).

    python kaggle/build_sota_notebook.py [--seeds 42,1,2] [--slug daxnet-polyp-segmentation]
    PYTHONUTF8=1 python kaggle/push_when_free.py daxnet-polyp-segmentation
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODULES = ['__init__.py', 'ssd_axial.py', 'data.py', 'losses.py', 'metrics.py', 'model.py', 'train.py']
DATASETS = ["debeshjha1/kvasirseg", "balraj98/cvcclinicdb", "longvil/cvc-colondb", "kokoroou/polypgen2021"]

INTRO = """# DAXNet: polyp segmentation on Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB and PolypGen

This single notebook builds DAXNet, **trains** it on all four datasets ({n_seeds} seeds each, two T4 GPUs
in parallel), saves the **checkpoints**, **tests** every checkpoint again from disk, and compares the
results with the BAMPolyp paper (ICCVW 2025), which reports on these four datasets.

**Model.** U-Net with a PVTv2-b2 encoder (ImageNet weights) whose bottleneck and decoder mix context
with an exact, bidirectional, edge-aware (geodesic) axial state-space operator. An image-level presence
head empties the mask of frames judged polyp-free (about 19 % of PolypGen frames contain no polyp).
Input 352 x 352. 28.65 M parameters.

**Protocol (identical to BAMPolyp).** Per-dataset split with `np.random.seed(42)` permutation,
80 / 10 / 10 train / val / test. Metric: per-image Dice and IoU at threshold 0.5, averaged over images.
The checkpoint with the best validation IoU is kept.
* **Protocol A** = validation score of that checkpoint: the number BAMPolyp reports.
* **Protocol B** = the held-out 10 % test split, evaluated with the same checkpoint (never used for selection).
* **+ TTA** = the same checkpoint with horizontal / vertical flip averaging.

Numbers of other methods in BAMPolyp's Table 3 come from their own papers and may use other splits
(the BAMPolyp paper says so); they are shown only as reference points.

| Section | What it does |
|---|---|
| 1 | Setup: GPUs, libraries, datasets |
| 2 | DAXNet source code |
| 3 | Experiment configuration |
| 4 | Training on all datasets and seeds (progress is printed every 30 min) |
| 5 | Checkpoints: saved with readable names |
| 6 | Testing: every checkpoint reloaded from disk and evaluated on val and test |
| 7 | Final results vs. BAMPolyp and its Table 3 |
| 8 | Training curves |
| 9 | Example predictions |
"""

SETUP = """!nvidia-smi -L
import os, subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'einops', 'timm'], check=False)
WORK, DATA = '/kaggle/working', '/kaggle/input'
os.chdir(WORK)
sys.path.insert(0, WORK)            # so the notebook can import the daxnet package written below
os.makedirs('daxnet', exist_ok=True)
import torch, timm
print('torch', torch.__version__, '| timm', timm.__version__, '| GPUs', torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print('datasets:', sorted(os.listdir(DATA)))
"""

CONFIG = """import json, time, glob
ENCODER, SIZE = 'pvt_v2_b2', 352
SEEDS = {seeds}
DATASETS = ['polypgen', 'kvasir', 'clinicdb', 'colondb']
NAMES = {{'kvasir': 'Kvasir-SEG', 'clinicdb': 'CVC-ClinicDB', 'colondb': 'CVC-ColonDB', 'polypgen': 'PolypGen'}}
EPOCHS = {{'polypgen': 100, 'kvasir': 200, 'clinicdb': 200, 'colondb': 200}}   # PolypGen's train set is 3.8x Kvasir's
EST_H = {{'polypgen': 3.4, 'kvasir': 1.85, 'clinicdb': 1.15, 'colondb': 0.75}}   # measured on a T4
START_BY_H, FINISH_BY_H = 10.5, 11.5    # no job starts after 10.5 h; running jobs finalize before 11.5 h

def job(ds, seed):
    cfg = json.dumps({{'encoder': ENCODER, 'presence': True}})
    args = (f"--model daxnet --recipe ours --protocol bam --dataset {{ds}} --size {{SIZE}} --seed {{seed}} "
            f"--epochs {{EPOCHS[ds]}} --cfg '{{cfg}}'")
    return dict(name=f'{{ds}}_seed{{seed}}', dataset=ds, seed=seed, args=args, est_h=EST_H[ds], epochs=EPOCHS[ds])

JOBS = [job(ds, s) for s in SEEDS for ds in DATASETS]     # seed by seed, so every dataset gets its first seed early
print(f'{{len(JOBS)}} training runs, ~{{sum(j["est_h"] for j in JOBS):.1f}} GPU-hours on {{max(1, torch.cuda.device_count())}} GPU(s):')
for j in JOBS: print(f"  {{j['name']:18s}} {{NAMES[j['dataset']]:13s}} {{j['epochs']}} epochs  ~{{j['est_h']}} h")

# download the ImageNet weights once, before the parallel workers start
timm.create_model(ENCODER, pretrained=True, features_only=True);
"""

RUNNER = """import queue, shlex, threading
T0 = time.time()
N_GPU = max(1, torch.cuda.device_count())
q = queue.Queue()
for j in JOBS: q.put(j)
os.makedirs(f'{WORK}/runs', exist_ok=True)
RUNNING, DONE = {}, threading.Event()
hms = lambda s: f'{int(s // 3600)}h{int(s % 3600 // 60):02d}m'

def last_record(out):
    try:
        recs = [json.loads(l) for l in open(f'{out}/log.jsonl') if l.strip()]
        return recs[-1] if recs else None
    except FileNotFoundError:
        return None

def monitor(every_s=1800):
    while not DONE.wait(every_s):
        lines = []
        for name, (out, epochs, t) in list(RUNNING.items()):
            r = last_record(out)
            if r and 'val' in r:
                lines.append(f"    {name:18s} epoch {r['epoch']:3d}/{epochs}  val Dice {r['val']['dice_net']:.4f}  "
                             f"test Dice {r['test']['dice_net']:.4f}  best epoch {r['best_epoch']}  ({hms(time.time() - t)})")
        print(f"[progress {hms(time.time() - T0)}]" + ('\\n' + '\\n'.join(lines) if lines else ' starting'), flush=True)

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
        RUNNING[j['name']] = (out, j['epochs'], t)
        print(f"[gpu{gpu}] START {j['name']:18s} at {hms(time.time() - T0)}", flush=True)
        with open(f'{out}/stdout.txt', 'w') as f:
            rc = subprocess.run(cmd, cwd=WORK, stdout=f, stderr=subprocess.STDOUT,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONWARNINGS='ignore')).returncode
        RUNNING.pop(j['name'], None)
        if os.path.exists(f'{out}/final.json'):
            r = json.load(open(f'{out}/final.json'))
            msg = (f"val Dice/IoU {r['val']['dice_net']:.4f}/{r['val']['iou_net']:.4f}  "
                   f"test Dice/IoU {r['test']['dice_net']:.4f}/{r['test']['iou_net']:.4f}  best epoch {r['best_epoch']}")
        else:
            msg = 'FAILED: ' + ' | '.join(open(f'{out}/stdout.txt').read().splitlines()[-3:])[:800]
        print(f"[gpu{gpu}] DONE  {j['name']:18s} in {hms(time.time() - t)}  {msg}", flush=True)

threading.Thread(target=monitor, daemon=True).start()
ths = [threading.Thread(target=worker, args=(g,)) for g in range(N_GPU)]
[t.start() for t in ths]; [t.join() for t in ths]
DONE.set()
print(f'training finished in {hms(time.time() - T0)}')
"""

CHECKPOINTS = """import shutil
os.makedirs(f'{WORK}/checkpoints', exist_ok=True)
CKPTS = []
print(f"{'checkpoint':45s} {'size':>8s}  {'best epoch':>10s}")
for f in sorted(glob.glob(f'{WORK}/runs/*/final.json')):
    d = os.path.dirname(f); m = json.load(open(f'{d}/meta.json')); r = json.load(open(f))
    name = f"daxnet_{ENCODER.replace('_', '')}_{SIZE}_{m['dataset']}_seed{m['seed']}"
    dst = f'{WORK}/checkpoints/{name}.pt'
    shutil.copy(f'{d}/best.pt', dst)
    info = dict(checkpoint=f'{name}.pt', dataset=m['dataset'], seed=m['seed'], size=m['size'], model_cfg=m['model_cfg'],
                best_epoch=r['best_epoch'], params_M=r['params_M'], selected_on='best validation IoU')
    json.dump(info, open(dst.replace('.pt', '.json'), 'w'), indent=1)
    CKPTS.append(info)
    print(f"{name + '.pt':45s} {os.path.getsize(dst) / 2**20:6.0f}MB  {r['best_epoch']:>10d}")
json.dump(CKPTS, open(f'{WORK}/checkpoints/manifest.json', 'w'), indent=1)
print(f'{len(CKPTS)} checkpoints in {WORK}/checkpoints (each with a .json describing how to rebuild the model)')
"""

TESTING = """from torch.utils.data import DataLoader
from daxnet.data import EvalSet, bam_split, eval_collate
from daxnet.model import build_model
from daxnet.train import evaluate
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TESTED = []
print(f"{'checkpoint':40s} {'val Dice/IoU':>16s} {'test Dice/IoU':>16s} {'test+TTA Dice/IoU':>18s}  matches training log")
for info in CKPTS:
    model = build_model('daxnet', **dict(info['model_cfg'], pretrained=False)).to(dev)
    model.load_state_dict(torch.load(f"{WORK}/checkpoints/{info['checkpoint']}", map_location=dev))
    split = bam_split(DATA, info['dataset'])
    res = {}
    for sp in ('val', 'test'):
        dl = DataLoader(EvalSet(split[sp], info['size']), batch_size=16, shuffle=False, collate_fn=eval_collate)
        for tta in (False, True):
            ev, _, _ = evaluate(model, 'daxnet', dl, dev, amp=dev.type == 'cuda', tta=tta)
            res[sp + ('_tta' if tta else '')] = ev.summary()
    ref = json.load(open(f"{WORK}/runs/{info['dataset']}_seed{info['seed']}/final.json"))
    same = abs(res['test']['dice_net'] - ref['test']['dice_net']) < 2e-3
    TESTED.append(dict(info, **res))
    di = lambda k: f"{res[k]['dice_net']:.4f}/{res[k]['iou_net']:.4f}"
    print(f"{info['checkpoint'][:-3]:40s} {di('val'):>16s} {di('test'):>16s} {di('test_tta'):>18s}  {'yes' if same else 'NO'}")
    del model; torch.cuda.empty_cache()
"""

RESULTS = """import numpy as np
PAPER = {'kvasir': (0.9380, 0.8881), 'clinicdb': (0.9437, 0.8939), 'colondb': (0.9255, 0.8659), 'polypgen': (0.8683, 0.8211)}
TABLE3_BEST = {'kvasir': ((0.9380, 'BAMPolyp'), (0.8910, 'SwinE-Net')),
               'clinicdb': ((0.9521, 'PVT-EMCAD-B2'), (0.9238, 'LGPS')),
               'colondb': ((0.9255, 'BAMPolyp'), (0.8659, 'BAMPolyp')),
               'polypgen': ((0.8683, 'BAMPolyp'), (0.8211, 'BAMPolyp'))}
ORDER = ['kvasir', 'clinicdb', 'colondb', 'polypgen']
PRESENT = [ds for ds in ORDER if any(t['dataset'] == ds for t in TESTED)]

def table(title, header, rows, note=None):
    w = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = '+-' + '-+-'.join('-' * n for n in w) + '-+'
    fmt = lambda r: '| ' + ' | '.join(str(x).ljust(n) for x, n in zip(r, w)) + ' |'
    out = ['', title, line, fmt(header), line.replace('-', '=')] + [fmt(r) for r in rows] + [line]
    if note: out.append(note)
    txt = '\\n'.join(out); print(txt); return txt

def stat(v):
    v = np.array(v); return f'{v.mean():.4f} +- {v.std(ddof=1):.4f}' if len(v) > 1 else f'{v.mean():.4f}'

report = [f"DAXNet ({ENCODER}, {SIZE} px, presence gate) | {TESTED[0]['params_M']:.2f} M parameters | seeds {SEEDS}"]
# 1. every checkpoint
rows = [[NAMES[t['dataset']], t['seed'], t['best_epoch'], f"{t['val']['dice_net']:.4f}", f"{t['val']['iou_net']:.4f}",
         f"{t['test']['dice_net']:.4f}", f"{t['test']['iou_net']:.4f}", f"{t['test_tta']['dice_net']:.4f}", f"{t['test_tta']['iou_net']:.4f}"]
        for t in sorted(TESTED, key=lambda t: (ORDER.index(t['dataset']), t['seed']))]
report.append(table('TABLE 1. Every trained model (tested from its saved checkpoint)',
                    ['Dataset', 'Seed', 'Best epoch', 'Val Dice', 'Val IoU', 'Test Dice', 'Test IoU', 'Test+TTA Dice', 'Test+TTA IoU'], rows))
# 2. vs the paper, Protocol A
rows, summary = [], []
for ds in ORDER:
    ts = [t for t in TESTED if t['dataset'] == ds]
    if not ts: continue
    vd, vi = [t['val']['dice_net'] for t in ts], [t['val']['iou_net'] for t in ts]
    pd_, pi = PAPER[ds]
    beat = all(d > pd_ for d in vd) and all(i > pi for i in vi)
    rows.append([NAMES[ds], len(ts), f'{pd_:.4f}', stat(vd), f'{np.mean(vd) - pd_:+.4f}', f'{pi:.4f}', stat(vi),
                 f'{np.mean(vi) - pi:+.4f}', 'YES (every seed)' if beat else ('yes (mean)' if np.mean(vd) > pd_ and np.mean(vi) > pi else 'NO')])
    summary.append(dict(dataset=ds, seeds=len(ts), paper_dice=pd_, paper_iou=pi, dice_mean=float(np.mean(vd)), iou_mean=float(np.mean(vi)),
                        dice_std=float(np.std(vd, ddof=1)) if len(ts) > 1 else 0.0, iou_std=float(np.std(vi, ddof=1)) if len(ts) > 1 else 0.0,
                        test_dice=float(np.mean([t['test']['dice_net'] for t in ts])), test_iou=float(np.mean([t['test']['iou_net'] for t in ts])),
                        test_tta_dice=float(np.mean([t['test_tta']['dice_net'] for t in ts])), test_tta_iou=float(np.mean([t['test_tta']['iou_net'] for t in ts])),
                        beats_paper_every_seed=beat))
report.append(table('TABLE 2. DAXNet vs. the BAMPolyp paper (Protocol A = the paper\\'s protocol, mean +- std over seeds)',
                    ['Dataset', 'Seeds', 'Paper Dice', 'DAXNet Dice', 'Diff', 'Paper IoU', 'DAXNet IoU', 'Diff', 'Beats paper?'], rows))
# 3. held-out test
rows = [[NAMES[s['dataset']], f"{s['test_dice']:.4f}", f"{s['test_iou']:.4f}", f"{s['test_tta_dice']:.4f}", f"{s['test_tta_iou']:.4f}"] for s in summary]
report.append(table('TABLE 3. Held-out test split (Protocol B, never used for model selection), mean over seeds',
                    ['Dataset', 'Test Dice', 'Test IoU', 'Test+TTA Dice', 'Test+TTA IoU'], rows))
# 4. vs best of BAMPolyp Table 3
rows = []
for s in summary:
    (td, tdn), (ti, tin) = TABLE3_BEST[s['dataset']]
    rows.append([NAMES[s['dataset']], f'{td:.4f} ({tdn})', f"{s['dice_mean']:.4f}", 'YES' if s['dice_mean'] > td else 'no',
                 f'{ti:.4f} ({tin})', f"{s['iou_mean']:.4f}", 'YES' if s['iou_mean'] > ti else 'no'])
report.append(table("TABLE 4. vs. the best value in BAMPolyp's Table 3 (Protocol A)",
                    ['Dataset', 'Best Dice (method)', 'DAXNet', 'Higher?', 'Best IoU (method)', 'DAXNet', 'Higher?'], rows,
                    note='Other methods in that table come from their own papers and may use other splits.'))
n_beat = sum(s['beats_paper_every_seed'] for s in summary)
verdict = f'\\nVERDICT: DAXNet beats the BAMPolyp paper on Dice AND IoU on {n_beat} of {len(summary)} datasets, on every seed.'
print(verdict); report.append(verdict)
open(f'{WORK}/RESULTS.txt', 'w').write('\\n'.join(report) + '\\n')
json.dump({'summary': summary, 'per_checkpoint': TESTED}, open(f'{WORK}/results.json', 'w'), indent=1)
print(f'\\nsaved {WORK}/RESULTS.txt and {WORK}/results.json')
"""

CURVES = """import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(PRESENT), figsize=(4.6 * len(PRESENT), 3.6), squeeze=False)
for ax, ds in zip(axes[0], PRESENT):
    for f in sorted(glob.glob(f'{WORK}/runs/{ds}_seed*/log.jsonl')):
        log = [r for r in (json.loads(l) for l in open(f) if l.strip()) if 'val' in r]
        s = f.split('_seed')[1].split('/')[0]
        ax.plot([r['epoch'] for r in log], [r['val']['dice_net'] for r in log], lw=1, label=f'val, seed {s}')
        ax.plot([r['epoch'] for r in log], [r['test']['dice_net'] for r in log], lw=1, ls='--', label=f'test, seed {s}')
    ax.axhline(PAPER[ds][0], color='k', lw=1, ls=':', label='BAMPolyp paper')
    ax.set_title(NAMES[ds]); ax.set_xlabel('epoch'); ax.set_ylabel('Dice'); ax.set_ylim(0.75, 1.0); ax.grid(alpha=.3)
axes[0][0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f'{WORK}/curves.png', dpi=110); plt.show()
"""

EXAMPLES = """from daxnet.train import predict
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
fig, axes = plt.subplots(len(PRESENT), 6, figsize=(15, 2.7 * len(PRESENT)), squeeze=False)
for row, ds in zip(axes, PRESENT):
    info = min((t for t in CKPTS if t['dataset'] == ds), key=lambda t: SEEDS.index(t['seed']))
    model = build_model('daxnet', **dict(info['model_cfg'], pretrained=False)).to(dev).eval()
    model.load_state_dict(torch.load(f"{WORK}/checkpoints/{info['checkpoint']}", map_location=dev))
    x, m, _, names = next(iter(DataLoader(EvalSet(bam_split(DATA, ds)['test'][:3], info['size']), batch_size=3, collate_fn=eval_collate)))
    with torch.no_grad():
        logit, pres = predict(model, 'daxnet', x.to(dev))
        pred = (torch.sigmoid(logit.float())[:, 0] > 0.5).cpu().numpy()
        if pres is not None: pred &= (torch.sigmoid(pres.float()) >= 0.5).cpu().numpy()[:, :, None]
    for i in range(3):
        img = np.clip(x[i].permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)
        row[2 * i].imshow(img); row[2 * i].set_title(f'{NAMES[ds]}: {names[i][:18]}', fontsize=7)
        ov = img.copy(); ov[m[i].numpy() > 0] = ov[m[i].numpy() > 0] * .5 + np.array([0, 1, 0]) * .5
        ov[pred[i]] = ov[pred[i]] * .5 + np.array([1, 0, 0]) * .5
        row[2 * i + 1].imshow(ov); row[2 * i + 1].set_title('green = truth, red = DAXNet', fontsize=7)
    for a in row: a.axis('off')
    del model
plt.tight_layout(); plt.savefig(f'{WORK}/examples.png', dpi=100); plt.show()
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='42,1,2')
    ap.add_argument('--slug', default='daxnet-polyp-segmentation')
    ap.add_argument('--title', default='DAXNet Polyp Segmentation')
    ap.add_argument('--user', default='amarnathmadaka')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',') if s.strip()]

    md = lambda s: dict(cell_type='markdown', metadata={}, source=s)
    code = lambda s: dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s)
    cells = [md(INTRO.format(n_seeds=len(seeds))), md('## 1. Setup'), code(SETUP),
             md('## 2. DAXNet source code\nWritten to `daxnet/` so the training script can run on both GPUs in parallel.')]
    for fn in MODULES:
        src = open(os.path.join(ROOT, 'daxnet', fn), encoding='utf-8').read()
        cells.append(code(f'%%writefile daxnet/{fn}\n' + src))
    cells += [md('## 3. Experiment configuration'), code(CONFIG.format(seeds=seeds)),
              md('## 4. Training on all datasets and seeds (2 GPUs in parallel)\n'
                 'Each run keeps the checkpoint with the best validation IoU. Progress is printed every 30 minutes.'),
              code(RUNNER),
              md('## 5. Checkpoints\nEvery selected model is saved as `checkpoints/daxnet_<encoder>_<size>_<dataset>_seed<seed>.pt` '
                 'with a `.json` that records how to rebuild it.'), code(CHECKPOINTS),
              md('## 6. Testing\nEach checkpoint is loaded from disk and evaluated on the validation and the held-out test split, '
                 'with and without flip TTA. The last column confirms that the numbers match the ones logged during training.'),
              code(TESTING),
              md('## 7. Final results'), code(RESULTS),
              md('## 8. Training curves (validation and test Dice per epoch)'), code(CURVES),
              md('## 9. Example predictions on the test split'), code(EXAMPLES)]
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
    print(f'wrote {out_dir} ({len(cells)} cells, seeds {seeds})')


if __name__ == '__main__':
    main()
