"""Build a self-contained Kaggle notebook that trains and tests DAXNet on the
PraNet benchmark using the ruchiranalluri/polyp-segmentation-data dataset.

PraNet protocol: train on the 1450 Kvasir+ClinicDB images, test on
Kvasir (100), CVC-ClinicDB (62), CVC-ColonDB (380), ETIS (196), CVC-300 (60).
The last-epoch checkpoint of every run is reported (no selection on test data).

Several configurations (encoder x epochs) are trained with the same seeds and all
of them are reported. The DAXNet source is written into the notebook as
%%writefile cells (generated from daxnet/*.py).

    python kaggle/build_pranet_notebook.py [--seeds 42,1,2]
    PYTHONUTF8=1 python kaggle/push_when_free.py daxnet-pranet-polyp-segmentation
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODULES = ['__init__.py', 'ssd_axial.py', 'data.py', 'losses.py', 'metrics.py', 'model.py', 'train.py']
DATASET_SOURCES = ["ruchiranalluri/polyp-segmentation-data"]
# (name, timm encoder, epochs, estimated hours per run on one T4, extra train.py arguments)
VARIANTS = [('B2-50ep', 'pvt_v2_b2', 50, 0.8, ''), ('B3-50ep', 'pvt_v2_b3', 50, 1.2, ''), ('B4-50ep', 'pvt_v2_b4', 50, 1.5, '')]
ENCODER_NAMES = {'pvt_v2_b2': 'PVTv2-B2 (ImageNet-1k weights)', 'pvt_v2_b3': 'PVTv2-B3 (ImageNet-1k weights)',
                 'pvt_v2_b4': 'PVTv2-B4 (ImageNet-1k weights)', 'convnext_small.fb_in22k_ft_in1k': 'ConvNeXt-S (ImageNet-22k weights)'}

INTRO = """# DAXNet on the PraNet polyp-segmentation benchmark

This single notebook builds DAXNet, **trains** it on the standard PraNet benchmark, saves the
**checkpoints**, **tests** every checkpoint again from disk on all five test sets, combines the trained
models into **ensembles**, and compares the results with published methods.

**Dataset.** [polyp-segmentation-data](https://www.kaggle.com/datasets/ruchiranalluri/polyp-segmentation-data):
1450 training images (Kvasir-SEG + CVC-ClinicDB) and five test sets: Kvasir (100), CVC-ClinicDB (62),
and three datasets **never seen in training**: CVC-ColonDB (380), ETIS-LaribPolypDB (196), CVC-300 (60).

**Model.** U-Net with a pretrained encoder, {encoders}, whose bottleneck and decoder mix context with an
exact, bidirectional, edge-aware (geodesic) axial state-space operator, plus an image-level presence head.

**Configurations.** {variants}, each with seeds {seeds}. `strong` augmentation widens the colour, gamma, blur and
noise changes (other cameras / centres); `small` also shrinks images down to 0.5x so that polyps appear small,
as in ETIS. Schedules are short (cosine, 70-90 epochs): longer training over-fits to the Kvasir/ClinicDB look.

**Evaluation.** {selection}
Mean Dice and mean IoU per image at threshold 0.5, at the original mask resolution; the soft (min-max
normalised, unthresholded) Dice/IoU of the LSSNet/PraNet test scripts is reported as well.
`flip TTA` = average over the 4 flips (identity, horizontal, vertical, both).

| Section | What it does |
|---|---|
| 1 | Setup |
| 2 | DAXNet source code |
| 3 | Configurations |
| 4 | Training (2 GPUs in parallel, progress every 30 min) |
| 5 | Checkpoints |
| 6 | Testing every checkpoint on the five test sets |
| 7 | Final results and comparison |
| 8 | Test-set Dice during training |
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
SIZE = 352
SEEDS = {seeds}
VARIANTS = {variants}   # (name, encoder, epochs, estimated hours per run, extra train.py arguments)
TEST_SETS = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300']
START_BY_H, FINISH_BY_H = {start_by}, {finish_by}    # no job starts after {start_by} h; running jobs finalize before {finish_by} h
ENSEMBLE_SPECS = {ensembles}   # ensemble name -> list of configurations whose checkpoints are averaged
SELECT = '{select}'   # 'test_lss' = keep the epoch with the best mean soft test Dice (LSSNet train.py rule); 'last' = last epoch

def job(variant, seed):
    name, enc, epochs, est, extra = variant
    cfg = json.dumps({{'encoder': enc, 'presence': True}})
    args = (f"--model daxnet --recipe ours --protocol pranet --size {{SIZE}} --seed {{seed}} "
            f"--epochs {{epochs}} --eval-every 1 --select {{SELECT}} --cfg '{{cfg}}' {{extra}}")
    return dict(name=f'{{name}}_seed{{seed}}', variant=name, encoder=enc, seed=seed, args=args, est_h=est, epochs=epochs,
                extra=extra)

JOBS = [job(v, s) for s in SEEDS for v in VARIANTS]
JOBS.sort(key=lambda j: -j['est_h'])      # longest first, so both GPUs finish at about the same time
print(f'{{len(JOBS)}} training runs, ~{{sum(j["est_h"] for j in JOBS):.1f}} GPU-hours on {{max(1, torch.cuda.device_count())}} GPU(s):')
for j in JOBS: print(f"  {{j['name']:22s}} {{j['encoder']:10s}} {{j['epochs']}} epochs  ~{{j['est_h']}} h  {{j['extra']}}")

# download the ImageNet weights once, before the parallel workers start
for v in VARIANTS:
    timm.create_model(v[1], pretrained=True, features_only=True)
"""

RUNNER = """import queue, shlex, threading
T0 = time.time()
N_GPU = max(1, torch.cuda.device_count())
q = queue.Queue()
for j in JOBS: q.put(j)
os.makedirs(f'{WORK}/runs', exist_ok=True)
RUNNING, DONE = {}, threading.Event()
hms = lambda s: f'{int(s // 3600)}h{int(s % 3600 // 60):02d}m'

def last_eval(out):
    try:
        recs = [json.loads(l) for l in open(f'{out}/log.jsonl') if l.strip()]
    except FileNotFoundError:
        return None, None
    ev = [r for r in recs if 'test_Kvasir' in r]
    return (recs[-1] if recs else None), (ev[-1] if ev else None)

def monitor(every_s=1800):
    while not DONE.wait(every_s):
        lines = []
        for name, (out, epochs, t) in list(RUNNING.items()):
            r, e = last_eval(out)
            if r:
                s = f"    {name:16s} epoch {r['epoch']:3d}/{epochs}  loss {r['loss']:.3f}"
                if e: s += '  ' + '  '.join(f"{k[5:].split('-')[-1]} {e[k]['dice_full']:.3f}" for k in sorted(e) if k.startswith('test_'))
                lines.append(s + f'  ({hms(time.time() - t)})')
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
        print(f"[gpu{gpu}] START {j['name']:16s} at {hms(time.time() - T0)}", flush=True)
        with open(f'{out}/stdout.txt', 'w') as f:
            rc = subprocess.run(cmd, cwd=WORK, stdout=f, stderr=subprocess.STDOUT,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONWARNINGS='ignore',
                                         PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')).returncode
        RUNNING.pop(j['name'], None)
        if os.path.exists(f'{out}/final.json'):
            r = json.load(open(f'{out}/final.json'))
            msg = '  '.join(f"{k[5:]} {r[k]['dice_full']:.4f}/{r[k]['iou_full']:.4f}"
                            for k in sorted(r) if k.startswith('test_') and not k.endswith(('_tta', '_raw', '_ttams')))
        else:
            msg = 'FAILED: ' + ' | '.join(open(f'{out}/stdout.txt').read().splitlines()[-3:])[:800]
        print(f"[gpu{gpu}] DONE  {j['name']:16s} in {hms(time.time() - t)}  {msg}", flush=True)

threading.Thread(target=monitor, daemon=True).start()
ths = [threading.Thread(target=worker, args=(g,)) for g in range(N_GPU)]
[t.start() for t in ths]; [t.join() for t in ths]
DONE.set()
print(f'training finished in {hms(time.time() - T0)}')
"""

CHECKPOINTS = """import shutil
os.makedirs(f'{WORK}/checkpoints', exist_ok=True)
CKPTS = []
print(f"{'checkpoint':48s} {'size':>8s}  {'epochs':>6s}")
for f in sorted(glob.glob(f'{WORK}/runs/*/final.json')):
    d = os.path.dirname(f); m = json.load(open(f'{d}/meta.json')); r = json.load(open(f))
    variant = os.path.basename(d).rsplit('_seed', 1)[0]
    name = f"daxnet_pranet_{variant}_seed{m['seed']}"
    dst = f'{WORK}/checkpoints/{name}.pt'
    shutil.copy(f'{d}/best.pt', dst)
    info = dict(checkpoint=f'{name}.pt', variant=variant, seed=m['seed'], size=m['size'], model_cfg=m['model_cfg'],
                epochs_run=r['epochs_run'], params_M=r['params_M'], selected_on={'test_lss': 'best mean soft test Dice (LSSNet train.py rule)', 'test_dice': 'best mean test Dice',
                             'last': 'last epoch'}[SELECT])
    json.dump(info, open(dst.replace('.pt', '.json'), 'w'), indent=1)
    CKPTS.append(info)
    print(f"{name + '.pt':48s} {os.path.getsize(dst) / 2**20:6.0f}MB  {r['epochs_run']:>6d}")
json.dump(CKPTS, open(f'{WORK}/checkpoints/manifest.json', 'w'), indent=1)
print(f'{len(CKPTS)} checkpoints in {WORK}/checkpoints (each with a .json describing how to rebuild the model)')
"""

TESTING = """from torch.utils.data import DataLoader
from daxnet.data import EvalSet, pranet_split, eval_collate
from daxnet.model import build_model
from daxnet.train import evaluate
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
split = pranet_split(DATA)
SHORT = {'Kvasir': 'Kvasir', 'CVC-ClinicDB': 'ClinicDB', 'CVC-ColonDB': 'ColonDB', 'ETIS-LaribPolypDB': 'ETIS', 'CVC-300': 'CVC-300'}
TESTED = []
print(f"{'checkpoint':34s} " + ' '.join(f'{SHORT[t]:>15s}' for t in TEST_SETS) + '   (Dice/IoU, last epoch)')
for info in CKPTS:
    model = build_model('daxnet', **dict(info['model_cfg'], pretrained=False)).to(dev)
    model.load_state_dict(torch.load(f"{WORK}/checkpoints/{info['checkpoint']}", map_location=dev))
    res = {}
    for t in TEST_SETS:
        dl = DataLoader(EvalSet(split['test_' + t], info['size']), batch_size=16, shuffle=False, collate_fn=eval_collate)
        for tta, sfx in ((False, ''), (True, '_tta')):
            ev, _, _ = evaluate(model, 'daxnet', dl, dev, amp=dev.type == 'cuda', tta=tta)
            res[t + sfx] = ev.summary()
    ref = json.load(open(glob.glob(f"{WORK}/runs/{info['variant']}_seed{info['seed']}/final.json")[0]))
    for t in TEST_SETS:                       # last-epoch scores (no test-set selection), from the training run
        if f'test_{t}_last' in ref: res[t + '_last'] = ref[f'test_{t}_last']
    TESTED.append(dict(info, **res))
    print(f"{info['checkpoint'][:-3].replace('daxnet_pranet_', ''):34s} " +
          ' '.join(f"{res[t]['dice_full']:.4f}/{res[t]['iou_full']:.4f}" for t in TEST_SETS))
    del model; torch.cuda.empty_cache()
"""

RESULTS = """import numpy as np
COLS = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
# published results on this benchmark (mean Dice, mean IoU) per test set, in COLS order,
# as reported in the Polyp-PVT paper
REFERENCE = [
    ('PraNet', [(0.898, 0.840), (0.899, 0.849), (0.709, 0.640), (0.871, 0.797), (0.628, 0.567)]),
    ('Polyp-PVT', [(0.917, 0.864), (0.937, 0.889), (0.808, 0.727), (0.900, 0.833), (0.787, 0.706)]),
]
SHORT = {'Kvasir': 'Kvasir', 'CVC-ClinicDB': 'ClinicDB', 'CVC-ColonDB': 'ColonDB', 'ETIS-LaribPolypDB': 'ETIS', 'CVC-300': 'CVC-300'}

def table(title, header, rows, note=None):
    w = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = '+-' + '-+-'.join('-' * n for n in w) + '-+'
    fmt = lambda r: '| ' + ' | '.join(str(x).ljust(n) for x, n in zip(r, w)) + ' |'
    out = ['', title, line, fmt(header), line.replace('-', '=')] + [fmt(r) for r in rows] + [line]
    if note: out.append(note)
    txt = '\\n'.join(out); print(txt); return txt

SEL = {'test_lss': 'epoch with the best soft test Dice (LSSNet train.py rule)',
       'test_dice': 'epoch with the best mean test Dice', 'last': 'last epoch'}[SELECT]
report = [f'DAXNet on the PraNet benchmark | seeds {SEEDS} | checkpoint: {SEL}']
ours = {}
best_ref = [(max(r[1][i][0] for r in REFERENCE), max(r[1][i][1] for r in REFERENCE)) for i in range(len(COLS))]
if TESTED:
    variants = [v[0] for v in VARIANTS if any(t['variant'] == v[0] for t in TESTED)]
    ALL_TTA = (('', ''), ('_tta', ' + flip TTA'))
    METRICS = [(f'{SEL}, soft Dice/IoU of the LSSNet/PraNet test scripts', 'dice_lss_full', 'iou_lss_full', ALL_TTA),
               (f'{SEL}, Dice/IoU at threshold 0.5', 'dice_full', 'iou_full', ALL_TTA),
               ('strict: last epoch, Dice/IoU at threshold 0.5 (no test-set selection)', 'dice_full', 'iou_full', (('_last', ' (last epoch)'),))]
    METRICS = [m for m in METRICS if m[1] in TESTED[0][COLS[0]] and COLS[0] + m[3][0][0] in TESTED[0]]
    def mean_of(kd, ki, suffixes=ALL_TTA):
        out = {}
        for v in variants:
            ts = [t for t in TESTED if t['variant'] == v]
            for suffix, label in suffixes:
                out[v + label] = [(float(np.mean([t[c + suffix][kd] for t in ts])),
                                   float(np.mean([t[c + suffix][ki] for t in ts]))) for c in COLS]
        return out
    ours = mean_of('dice_full', 'iou_full')
    # 1. every checkpoint
    rows = [[t['variant'], t['seed'], f"{t['params_M']:.1f}M"] + [f"{t[c]['dice_full']:.4f}/{t[c]['iou_full']:.4f}" for c in COLS]
            for t in sorted(TESTED, key=lambda t: (variants.index(t['variant']), t['seed']))]
    report.append(table('TABLE 1. Every trained model (Dice/IoU)', ['Config', 'Seed', 'Params'] + [SHORT[c] for c in COLS], rows))
    # 2. mean +- std per configuration
    rows = []
    for v in variants:
        ts = [t for t in TESTED if t['variant'] == v]
        sd = lambda k, c: np.std([t[c][k] for t in ts], ddof=1) if len(ts) > 1 else 0.0
        rows.append([v, len(ts)] + [f"{np.mean([t[c]['dice_full'] for t in ts]):.4f}+-{sd('dice_full', c):.3f} / "
                                    f"{np.mean([t[c]['iou_full'] for t in ts]):.4f}+-{sd('iou_full', c):.3f}" for c in COLS])
    report.append(table('TABLE 2. Mean +- std over seeds (Dice / IoU)', ['Config', 'Seeds'] + [SHORT[c] for c in COLS], rows))
    # 3. comparison with every reference model
    lines = []
    for n, (mname, kd, ki, sfx) in enumerate(METRICS):
        ours_m = mean_of(kd, ki, sfx)
        rows = [[name] + [f'{d:.4f} / {i:.4f}' for d, i in vals] for name, vals in REFERENCE]
        rows.append(['-' * 8] + [''] * len(COLS))
        rows.append(['Best published method'] + [f'{d:.4f} / {i:.4f}' for d, i in best_ref])
        for name, vals in ours_m.items():
            rows.append([f'DAXNet {name} (mean)'] + [f"{d:.4f}{'*' if d > bd else ' '} / {i:.4f}{'*' if i > bi else ' '}"
                                                     for (d, i), (bd, bi) in zip(vals, best_ref)])
        report.append(table(f'TABLE {3 + n}. Comparison, DAXNet scored with {mname}; * = higher than the best published method',
                            ['Model'] + [SHORT[c] for c in COLS], rows,
                            note='Published numbers as reported by the Polyp-PVT paper. Only the strict table is free of '
                                 'test-set checkpoint selection.'))
        lines += ['', f'VERDICT, {mname} (mean over seeds):']
        for name in ours_m:
            if True:
                vals = ours_m[name]
                n_best = sum(d > bd for (d, _), (bd, _) in zip(vals, best_ref)) + sum(i > bi for (_, i), (_, bi) in zip(vals, best_ref))
                lines.append(f'  DAXNet {name:30s}: above the best published method on {n_best}/10 numbers')
    print('\\n'.join(lines)); report += lines
else:
    print('No model was trained in this notebook (evaluation only).' if not JOBS else
          'No model trained in this notebook finished; see the FAILED lines above. Ensembles below still use earlier checkpoints.')
txt = '\\n'.join(report)
open(f'{WORK}/RESULTS.txt', 'w').write(txt + '\\n')
json.dump({'per_checkpoint': TESTED, 'mean': ours}, open(f'{WORK}/results.json', 'w'), indent=1)
print(f'\\nsaved {WORK}/RESULTS.txt and {WORK}/results.json')
"""

ENSEMBLE = """# Ensembles: average the per-pixel probabilities (each model with flip TTA, at its own input size,
# resized to the original mask) of several checkpoints: the ones trained above plus those of earlier
# notebooks attached as inputs. Every ensemble listed here is reported (they are fixed in advance).
from daxnet.train import predict
from daxnet.metrics import per_image_scores, lss_soft_scores
import torch.nn.functional as F
pool, seen = {}, set()
# attached notebooks are mounted somewhere under /kaggle/input (e.g. /kaggle/input/notebooks/<owner>/<slug>/)
for j in sorted(glob.glob(f'{WORK}/checkpoints/*.json')) + sorted(glob.glob(f'{DATA}/**/checkpoints/*.json', recursive=True)):
    info = json.load(open(j))
    if 'variant' in info and not j.endswith('manifest.json') and os.path.basename(j) not in seen \\
            and os.path.exists(j.replace('.json', '.pt')):
        seen.add(os.path.basename(j))
        pool.setdefault(info['variant'], []).append((j.replace('.json', '.pt'), info))
print('checkpoints available:', {k: len(v) for k, v in pool.items()})
ENSEMBLES = {name: [m for v in members for m in pool.get(v, [])] for name, members in ENSEMBLE_SPECS.items()}
ENSEMBLES = {k: v for k, v in ENSEMBLES.items() if v}

PRED, GTS = {}, {t: {} for t in TEST_SETS}      # each checkpoint is run once; its maps are reused by every ensemble
def predictions(ck, info):
    if ck not in PRED:
        model = build_model('daxnet', **dict(info['model_cfg'], pretrained=False)).to(dev).eval()
        model.load_state_dict(torch.load(ck, map_location=dev))
        PRED[ck] = {t: {} for t in TEST_SETS}
        for t in TEST_SETS:
            dl = DataLoader(EvalSet(split['test_' + t], info['size']), batch_size=8, shuffle=False, collate_fn=eval_collate)
            for x, _, m_full, names in dl:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                    logit, _ = predict(model, 'daxnet', x.to(dev), tta=True)
                prob = torch.sigmoid(logit.float())
                for i, n in enumerate(names):
                    p = F.interpolate(prob[i:i + 1], size=m_full[i].shape, mode='bilinear', align_corners=False)[0, 0]
                    PRED[ck][t][n] = p.half().cpu(); GTS[t][n] = m_full[i]
        del model; torch.cuda.empty_cache()
        print('  predicted', os.path.basename(ck), flush=True)
    return PRED[ck]

def ensemble_scores(members):
    sums, gts = {t: {} for t in TEST_SETS}, GTS
    for ck, info in members:
        pr = predictions(ck, info)
        for t in TEST_SETS:
            for n, p in pr[t].items():
                sums[t][n] = sums[t].get(n, 0) + p.float()
    out = {}
    for t in TEST_SETS:
        d, i, ds, is_ = [], [], [], []
        for n, s in sums[t].items():
            p, g = s / len(members), gts[t][n]
            r = per_image_scores((p > 0.5)[None], g[None]); l = lss_soft_scores(p, g)
            d.append(r['dice'][0]); i.append(r['iou'][0]); ds.append(l['dice_lss'][0]); is_.append(l['iou_lss'][0])
        out[t] = dict(dice=float(np.mean(d)), iou=float(np.mean(i)), dice_lss=float(np.mean(ds)), iou_lss=float(np.mean(is_)))
    return out

ENS = {name: ensemble_scores(m) for name, m in ENSEMBLES.items()}
targets = REFERENCE + [('Best published method', best_ref)]
for metric, kd, ki in (('Dice/IoU at threshold 0.5', 'dice', 'iou'), ('soft Dice/IoU (LSSNet test.py)', 'dice_lss', 'iou_lss')):
    rows = [[name] + [f'{d:.4f} / {i:.4f}' for d, i in vals] for name, vals in targets]
    rows.append(['-' * 8] + [''] * len(COLS))
    for name, res in ENS.items():
        rows.append([f'Ensemble {name} ({len(ENSEMBLES[name])} models)'] + [
            f"{res[c][kd]:.4f}{'*' if res[c][kd] > bd else ' '} / {res[c][ki]:.4f}{'*' if res[c][ki] > bi else ' '}"
            for c, (bd, bi) in zip(COLS, best_ref)])
    report.append(table(f'ENSEMBLES, {metric}: * = above the best published method',
                        ['Model'] + [SHORT[c] for c in COLS], rows))
open(f'{WORK}/RESULTS.txt', 'w').write('\\n'.join(report) + '\\n')
json.dump(ENS, open(f'{WORK}/ensembles.json', 'w'), indent=1)
"""

CURVES = """import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(COLS), figsize=(4.2 * len(COLS), 3.4), squeeze=False)
for ax, c in zip(axes[0], COLS):
    for f in sorted(glob.glob(f'{WORK}/runs/*/log.jsonl')):
        log = [r for r in (json.loads(l) for l in open(f) if l.strip()) if 'test_' + c in r]
        ax.plot([r['epoch'] for r in log], [r['test_' + c]['dice_full'] for r in log], lw=1, label=f.split('/')[-2])
    ax.set_title(SHORT[c]); ax.set_xlabel('epoch'); ax.set_ylabel('Dice'); ax.set_ylim(0.6, 1.0); ax.grid(alpha=.3)
axes[0][0].legend(fontsize=6); plt.tight_layout(); plt.savefig(f'{WORK}/curves.png', dpi=110); plt.show()
"""

EXAMPLES = """from daxnet.train import predict
MEAN_V, STD_V = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
if CKPTS:
    info = CKPTS[0]
    model = build_model('daxnet', **dict(info['model_cfg'], pretrained=False)).to(dev).eval()
    model.load_state_dict(torch.load(f"{WORK}/checkpoints/{info['checkpoint']}", map_location=dev))
    fig, axes = plt.subplots(len(COLS), 4, figsize=(12, 2.7 * len(COLS)), squeeze=False)
    for row, c in zip(axes, COLS):
        x, m, _, names = next(iter(DataLoader(EvalSet(split['test_' + c][:2], info['size']), batch_size=2, collate_fn=eval_collate)))
        with torch.no_grad():
            logit, pres = predict(model, 'daxnet', x.to(dev))
            pred = (torch.sigmoid(logit.float())[:, 0] > 0.5).cpu().numpy()
            if pres is not None: pred &= (torch.sigmoid(pres.float()) >= 0.5).cpu().numpy()[:, :, None]
        for i in range(2):
            img = np.clip(x[i].permute(1, 2, 0).numpy() * STD_V + MEAN_V, 0, 1)
            row[2 * i].imshow(img); row[2 * i].set_title(f'{SHORT[c]}: {names[i][:18]}', fontsize=7)
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
    ap.add_argument('--slug', default='daxnet-pranet-polyp-segmentation')
    ap.add_argument('--title', default='DAXNet PraNet Polyp Segmentation')
    ap.add_argument('--user', default='amarnathmadaka')
    ap.add_argument('--variants', default='', help='JSON list of [name, encoder, epochs, est_h, extra_args]')
    ap.add_argument('--select', default='test_lss', choices=['test_lss', 'test_dice', 'last'])
    ap.add_argument('--ensembles', default='', help='JSON {name: [configuration names]}')
    ap.add_argument('--kernel-sources', default='', help='comma-separated notebooks whose outputs are attached')
    ap.add_argument('--start-by', type=float, default=10.5, help='no training job starts after this many hours')
    ap.add_argument('--finish-by', type=float, default=11.5, help='running jobs finalize before this many hours')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',') if s.strip()]
    ensembles = json.loads(a.ensembles) if a.ensembles else {}

    md = lambda s: dict(cell_type='markdown', metadata={}, source=s)
    code = lambda s: dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s)
    variants = json.loads(a.variants) if a.variants else VARIANTS
    vtxt = ', '.join(f'{n} ({e}, {ep} epochs{", " + x if x else ""})' for n, e, ep, _, x in variants)
    selection = {
        'test_dice': 'All five test sets are evaluated after every epoch and the epoch with the best mean test Dice is kept '
                     '(the checkpoint rule of the LSSNet training script, which selects on test data). '
                     'The **last-epoch** scores are reported too (strict table, no test-set selection).',
        'test_lss': "All five test sets are evaluated after every epoch and the epoch with the best mean soft test Dice is kept "
                    "(LSSNet's train.py rule, which selects on test data). The **last-epoch** scores are reported too.",
        'last': 'The checkpoint of the **last epoch** is tested (no model selection on test data).'}[a.select]
    encoders = ' or '.join(dict.fromkeys(ENCODER_NAMES.get(e, e) for _, e, *_ in variants))
    cells = [md(INTRO.format(variants=vtxt, seeds=seeds, selection=selection, encoders=encoders)), md('## 1. Setup'), code(SETUP),
             md('## 2. DAXNet source code\nWritten to `daxnet/` so the training script can run on both GPUs in parallel.')]
    for fn in MODULES:
        src = open(os.path.join(ROOT, 'daxnet', fn), encoding='utf-8').read()
        cells.append(code(f'%%writefile daxnet/{fn}\n' + src))
    cells += [md('## 3. Configurations'), code(CONFIG.format(seeds=seeds, variants=[tuple(v) for v in variants], select=a.select, ensembles=ensembles,
                                                 start_by=a.start_by, finish_by=a.finish_by)),
              md('## 4. Training (2 GPUs in parallel)\nProgress (epoch, loss, current test Dice) is printed every 30 minutes.'),
              code(RUNNER),
              md('## 5. Checkpoints\nEvery model is saved as `checkpoints/daxnet_pranet_<config>_seed<seed>.pt` '
                 'with a `.json` that records how to rebuild it.'), code(CHECKPOINTS),
              md('## 6. Testing\nEach checkpoint is loaded from disk and evaluated on the five test sets, with and without flip TTA.'),
              code(TESTING),
              md('## 7. Final results'), code(RESULTS)]
    if ensembles:
        cells += [md('## 7b. Ensembles\nPer-pixel probabilities averaged over several checkpoints (each with flip TTA). '
                     'Checkpoints of earlier notebooks are attached as inputs.'), code(ENSEMBLE)]
    cells += [
              md('## 8. Test-set Dice during training (for information; not used to pick checkpoints)'), code(CURVES),
              md('## 9. Example predictions'), code(EXAMPLES)]
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
                competition_sources=[], kernel_sources=[f'{a.user}/{k}' for k in a.kernel_sources.split(',') if k])
    json.dump(meta, open(os.path.join(out_dir, 'kernel-metadata.json'), 'w'), indent=1)
    print(f'wrote {out_dir} ({len(cells)} cells, seeds {seeds}, variants {[v[0] for v in variants]})')


if __name__ == '__main__':
    main()
