"""Build a self-contained Kaggle notebook that trains and tests DAXNet
(PVTv2-b2 encoder, 352 px, presence gate) using the PraNet protocol on the
ruchiranalluri/polyp-segmentation-data dataset.

PraNet protocol: train on the 1450 Kvasir+ClinicDB images, test on
Kvasir (100), CVC-ClinicDB (62), CVC-ColonDB (380), ETIS (196), CVC-300 (60).

The DAXNet source is written into the notebook as %%writefile cells (generated
from daxnet/*.py, so the notebook always matches the repository).

    python kaggle/build_pranet_notebook.py [--seeds 42,1,2] [--slug daxnet-pranet-polyp-segmentation]
    kaggle kernels push -p kaggle/build/daxnet-pranet-polyp-segmentation
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODULES = ['__init__.py', 'ssd_axial.py', 'data.py', 'losses.py', 'metrics.py', 'model.py', 'train.py']
DATASET_SOURCES = ["ruchiranalluri/polyp-segmentation-data"]

INTRO = """# DAXNet: polyp segmentation on the PraNet benchmark (polyp-segmentation-data)

This single notebook builds DAXNet, **trains** it on the standard PraNet benchmark
({n_seeds} seeds, two T4 GPUs in parallel), saves the **checkpoints**, **tests** every
checkpoint again from disk, and reports the final results on all five test sets.

**Dataset.** [polyp-segmentation-data](https://www.kaggle.com/datasets/ruchiranalluri/polyp-segmentation-data)
(LSSNet-main layout): 1450 training images (Kvasir-SEG + CVC-ClinicDB combined) and five test sets:
Kvasir (100), CVC-ClinicDB (62), CVC-ColonDB (380), ETIS-LaribPolypDB (196), CVC-300 (60).

**Model.** U-Net with a PVTv2-b2 encoder (ImageNet weights) whose bottleneck and decoder mix context
with an exact, bidirectional, edge-aware (geodesic) axial state-space operator. An image-level presence
head empties the mask of frames judged polyp-free.
Input 352 x 352. 28.65 M parameters.

**Protocol (PraNet benchmark).** Train on the combined 1450 images, test on the five standard test
sets. The checkpoint from the last epoch is kept (no validation-based early stopping, following the
PraNet convention). All seeds use **{epochs} epochs**.

| Section | What it does |
|---|---|
| 1 | Setup: GPUs, libraries, datasets |
| 2 | DAXNet source code |
| 3 | Experiment configuration |
| 4 | Training ({n_seeds} seeds, 2 GPUs in parallel) |
| 5 | Checkpoints: saved with readable names |
| 6 | Testing: every checkpoint evaluated on all 5 test sets |
| 7 | Final results |
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
EPOCHS = {epochs}
EST_H = {est_h}    # measured on a single T4 for {epochs} epochs
START_BY_H, FINISH_BY_H = 10.5, 11.5    # no job starts after 10.5 h; running jobs finalize before 11.5 h
TEST_SETS = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300']

def job(seed):
    cfg = json.dumps({{'encoder': ENCODER, 'presence': True}})
    args = (f"--model daxnet --recipe ours --protocol pranet --size {{SIZE}} --seed {{seed}} "
            f"--epochs {{EPOCHS}} --eval-every 10 --cfg '{{cfg}}'")
    return dict(name=f'pranet_seed{{seed}}', seed=seed, args=args, est_h=EST_H, epochs=EPOCHS)

JOBS = [job(s) for s in SEEDS]
print(f'{{len(JOBS)}} training runs, ~{{sum(j["est_h"] for j in JOBS):.1f}} GPU-hours on {{max(1, torch.cuda.device_count())}} GPU(s):')
for j in JOBS: print(f"  {{j['name']:18s}} {{EPOCHS}} epochs  ~{{j['est_h']}} h")

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
            if r:
                ep = r.get('epoch', '?')
                loss = f"loss {r.get('loss', 0):.4f}" if 'loss' in r else ''
                lines.append(f"    {name:18s} epoch {ep}/{epochs}  {loss}  ({hms(time.time() - t)})")
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
            tests = [k for k in r if k.startswith('test_') and not k.endswith(('_tta', '_raw'))]
            msg = '  '.join(f"{k.split('_', 1)[1]} D{r[k]['dice_full']:.4f}" for k in sorted(tests)[:3])
            print(f"[gpu{gpu}] DONE  {j['name']:18s} in {hms(time.time() - t)}  {msg}", flush=True)
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
print(f"{'checkpoint':50s} {'size':>8s}  {'epochs run':>10s}")
for f in sorted(glob.glob(f'{WORK}/runs/*/final.json')):
    d = os.path.dirname(f); m = json.load(open(f'{d}/meta.json')); r = json.load(open(f))
    name = f"daxnet_{ENCODER.replace('_', '')}_{SIZE}_pranet_seed{m['seed']}"
    dst = f'{WORK}/checkpoints/{name}.pt'
    shutil.copy(f'{d}/best.pt', dst)
    info = dict(checkpoint=f'{name}.pt', seed=m['seed'], size=m['size'], model_cfg=m['model_cfg'],
                epochs_run=r.get('epochs_run', EPOCHS), params_M=r['params_M'],
                selected_on='last epoch (PraNet convention)')
    json.dump(info, open(dst.replace('.pt', '.json'), 'w'), indent=1)
    CKPTS.append(info)
    print(f"{name + '.pt':50s} {os.path.getsize(dst) / 2**20:6.0f}MB  {r.get('epochs_run', '?'):>10}")
json.dump(CKPTS, open(f'{WORK}/checkpoints/manifest.json', 'w'), indent=1)
print(f'{len(CKPTS)} checkpoints in {WORK}/checkpoints (each with a .json describing how to rebuild the model)')
"""

TESTING = """from torch.utils.data import DataLoader
from daxnet.data import EvalSet, pranet_split, eval_collate
from daxnet.model import build_model
from daxnet.train import evaluate
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TESTED = []
split = pranet_split(DATA)
test_keys = sorted([k for k in split if k.startswith('test_')])
header = f"{'checkpoint':45s} " + '  '.join(f'{k.split(\"_\", 1)[1]:>14s}' for k in test_keys)
print(header)

for info in CKPTS:
    model = build_model('daxnet', **dict(info['model_cfg'], pretrained=False)).to(dev)
    model.load_state_dict(torch.load(f"{WORK}/checkpoints/{info['checkpoint']}", map_location=dev))
    res = {}
    for k in test_keys:
        dl = DataLoader(EvalSet(split[k], info['size']), batch_size=16, shuffle=False, collate_fn=eval_collate)
        for tta in (False, True):
            ev, _, _ = evaluate(model, 'daxnet', dl, dev, amp=dev.type == 'cuda', tta=tta)
            res[k + ('_tta' if tta else '')] = ev.summary()
    TESTED.append(dict(info, **res))
    di = lambda k: f"{res[k]['dice_full']:.4f}/{res[k]['iou_full']:.4f}"
    vals = '  '.join(f'{di(k):>14s}' for k in test_keys)
    print(f"{info['checkpoint'][:-3]:45s} {vals}")
    del model; torch.cuda.empty_cache()
"""

RESULTS = """import numpy as np
NAMES = {'Kvasir': 'Kvasir', 'CVC-ClinicDB': 'CVC-ClinicDB', 'CVC-ColonDB': 'CVC-ColonDB',
         'ETIS-LaribPolypDB': 'ETIS', 'CVC-300': 'CVC-300'}

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
report.append(f"Protocol: PraNet benchmark | Training: 1450 images | {EPOCHS} epochs per seed")

# 1. every checkpoint, every test set (Dice / IoU)
test_keys = sorted([k for k in TESTED[0] if k.startswith('test_') and not k.endswith(('_tta', '_raw'))])
header = ['Seed'] + [NAMES.get(k.split('_', 1)[1], k.split('_', 1)[1]) + ' D/I' for k in test_keys]
rows = []
for t in sorted(TESTED, key=lambda t: t['seed']):
    row = [t['seed']]
    for k in test_keys:
        row.append(f"{t[k]['dice_full']:.4f}/{t[k]['iou_full']:.4f}")
    rows.append(row)
report.append(table('TABLE 1. Every seed, Dice/IoU on each test set', header, rows))

# 2. mean +- std per test set
header2 = ['Test set', 'Dice (mean+-std)', 'IoU (mean+-std)', 'Dice+TTA', 'IoU+TTA']
rows2 = []
for k in test_keys:
    name = NAMES.get(k.split('_', 1)[1], k.split('_', 1)[1])
    dices = [t[k]['dice_full'] for t in TESTED]
    ious = [t[k]['iou_full'] for t in TESTED]
    k_tta = k + '_tta'
    dices_tta = [t[k_tta]['dice_full'] for t in TESTED]
    ious_tta = [t[k_tta]['iou_full'] for t in TESTED]
    rows2.append([name, stat(dices), stat(ious), stat(dices_tta), stat(ious_tta)])
report.append(table('TABLE 2. Mean +- std over seeds per test set', header2, rows2))

# 3. also compute MAE
header3 = ['Test set', 'MAE (mean+-std)']
rows3 = []
for k in test_keys:
    name = NAMES.get(k.split('_', 1)[1], k.split('_', 1)[1])
    maes = [t[k]['mae_full'] for t in TESTED]
    rows3.append([name, stat(maes)])
report.append(table('TABLE 3. MAE (full resolution) per test set', header3, rows3))

REF = {'Kvasir': ((0.898, 0.840), (0.917, 0.864)), 'CVC-ClinicDB': ((0.899, 0.849), (0.937, 0.889)),
       'CVC-ColonDB': ((0.709, 0.640), (0.808, 0.727)), 'ETIS-LaribPolypDB': ((0.628, 0.567), (0.787, 0.706)),
       'CVC-300': ((0.871, 0.797), (0.900, 0.833))}
rows4 = []
for k in test_keys:
    tk = k.split('_', 1)[1]
    d = float(np.mean([t[k]['dice_full'] for t in TESTED])); i = float(np.mean([t[k]['iou_full'] for t in TESTED]))
    (pd_, pi), (vd, vi) = REF[tk]
    rows4.append([NAMES.get(tk, tk), f'{pd_:.3f} / {pi:.3f}', f'{vd:.3f} / {vi:.3f}', f'{d:.4f} / {i:.4f}',
                  'YES' if d > vd and i > vi else ('Dice only' if d > vd else 'no'),
                  'YES' if d >= 0.90 and i >= 0.90 else ('Dice only' if d >= 0.90 else 'no')])
report.append(table('TABLE 4. vs. published results on the same benchmark (mean Dice / mean IoU)',
                    ['Test set', 'PraNet (MICCAI 20)', 'Polyp-PVT', 'DAXNet (mean)', 'Above Polyp-PVT?', 'Both >= 0.90?'], rows4,
                    note='Published numbers as reported in the Polyp-PVT paper. ColonDB, ETIS and CVC-300 are never seen in training.'))
txt = '\\n'.join(report)
open(f'{WORK}/RESULTS.txt', 'w').write(txt + '\\n')
json.dump({'per_checkpoint': TESTED}, open(f'{WORK}/results.json', 'w'), indent=1)
print(f'\\nsaved {WORK}/RESULTS.txt and {WORK}/results.json')
"""

CURVES = """import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 1, figsize=(8, 4))
for f in sorted(glob.glob(f'{WORK}/runs/pranet_seed*/log.jsonl')):
    log = [json.loads(l) for l in open(f) if l.strip()]
    s = f.split('_seed')[1].split('/')[0]
    axes.plot([r['epoch'] for r in log], [r.get('loss', 0) for r in log], lw=1, label=f'loss, seed {s}')
axes.set_xlabel('epoch'); axes.set_ylabel('loss'); axes.set_title('Training loss'); axes.grid(alpha=.3)
axes.legend(fontsize=8); plt.tight_layout(); plt.savefig(f'{WORK}/curves.png', dpi=110); plt.show()
"""

EXAMPLES = """from daxnet.train import predict
from daxnet.data import EvalSet, pranet_split, eval_collate
MEAN_V, STD_V = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
split = pranet_split(DATA)
test_keys = sorted([k for k in split if k.startswith('test_')])
info = CKPTS[0]   # first seed
model = build_model('daxnet', **dict(info['model_cfg'], pretrained=False)).to(dev).eval()
model.load_state_dict(torch.load(f"{WORK}/checkpoints/{info['checkpoint']}", map_location=dev))
fig, axes = plt.subplots(len(test_keys), 4, figsize=(12, 2.7 * len(test_keys)), squeeze=False)
for row_idx, tk in enumerate(test_keys):
    row = axes[row_idx]
    pairs = split[tk][:2]
    x, m, _, names = next(iter(DataLoader(EvalSet(pairs, info['size']), batch_size=2, collate_fn=eval_collate)))
    with torch.no_grad():
        logit, pres = predict(model, 'daxnet', x.to(dev))
        pred = (torch.sigmoid(logit.float())[:, 0] > 0.5).cpu().numpy()
        if pres is not None: pred &= (torch.sigmoid(pres.float()) >= 0.5).cpu().numpy()[:, :, None]
    tname = NAMES.get(tk.split('_', 1)[1], tk.split('_', 1)[1])
    for i in range(min(2, len(pairs))):
        img = np.clip(x[i].permute(1, 2, 0).numpy() * STD_V + MEAN_V, 0, 1)
        row[2 * i].imshow(img); row[2 * i].set_title(f'{tname}: {names[i][:18]}', fontsize=7)
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
    ap.add_argument('--epochs', type=int, default=200, help='same epochs for every seed')
    ap.add_argument('--slug', default='daxnet-pranet-polyp-segmentation')
    ap.add_argument('--title', default='DAXNet PraNet Polyp Segmentation')
    ap.add_argument('--user', default='amarnathmadaka')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',') if s.strip()]

    md = lambda s: dict(cell_type='markdown', metadata={}, source=s)
    code = lambda s: dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s)
    cells = [md(INTRO.format(n_seeds=len(seeds), epochs=a.epochs)),
             md('## 1. Setup'), code(SETUP),
             md('## 2. DAXNet source code\nWritten to `daxnet/` so the training script can run on both GPUs in parallel.')]
    for fn in MODULES:
        src = open(os.path.join(ROOT, 'daxnet', fn), encoding='utf-8').read()
        cells.append(code(f'%%writefile daxnet/{fn}\n' + src))
    cells += [md('## 3. Experiment configuration'),
              code(CONFIG.format(seeds=seeds, epochs=a.epochs, est_h=3.5)),
              md('## 4. Training (2 GPUs in parallel)\n'
                 f'Each seed trains for {a.epochs} epochs on the 1450-image training set. '
                 'Progress is printed every 30 minutes.'),
              code(RUNNER),
              md('## 5. Checkpoints\nEvery selected model is saved as '
                 '`checkpoints/daxnet_<encoder>_<size>_pranet_seed<seed>.pt` '
                 'with a `.json` that records how to rebuild it.'), code(CHECKPOINTS),
              md('## 6. Testing\nEach checkpoint is loaded from disk and evaluated on all 5 PraNet '
                 'test sets, with and without flip TTA.'),
              code(TESTING),
              md('## 7. Final results'), code(RESULTS),
              md('## 8. Training curves'), code(CURVES),
              md('## 9. Example predictions on the test sets'), code(EXAMPLES)]
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
                machine_shape='NvidiaTeslaT4', dataset_sources=DATASET_SOURCES,
                competition_sources=[], kernel_sources=[])
    json.dump(meta, open(os.path.join(out_dir, 'kernel-metadata.json'), 'w'), indent=1)
    print(f'wrote {out_dir} ({len(cells)} cells, seeds {seeds}, {a.epochs} epochs)')


if __name__ == '__main__':
    main()
