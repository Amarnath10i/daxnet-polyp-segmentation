"""Compare every downloaded run with the BAMPolyp paper (Table 2/3: Dice / IoU).

    python kaggle/report.py                 # all runs under kaggle/results/*/runs/
    python kaggle/report.py --filter 352    # only run names containing '352'

Columns per run (Dice/IoU at network resolution, BAMPolyp's metric):
  valA   : best-validation-IoU epoch, validation score  (Protocol A = the paper's numbers)
  testB  : held-out 10% test at that epoch                (Protocol B)
  +tta   : the same checkpoint with flip TTA
  raw    : presence gate switched off (models with a presence head)
  last20 : mean test Dice over the final 20 epochs (stability)
A '*' marks a Protocol-A value above the paper's. Runs that differ only in the
seed are also aggregated (mean +- std, n).
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = {'kvasir': (0.9380, 0.8881), 'clinicdb': (0.9437, 0.8939),
         'colondb': (0.9255, 0.8659), 'polypgen': (0.8683, 0.8211)}


def load_runs(pattern=''):
    runs = []
    for d in sorted(glob.glob(os.path.join(HERE, 'results', '*', 'runs', '*'))):
        name = os.path.basename(d)
        if (name.startswith('smoke') or pattern and pattern not in name
                or not os.path.exists(os.path.join(d, 'final.json'))):
            continue
        meta = json.load(open(os.path.join(d, 'meta.json')))
        if meta.get('protocol') != 'bam':
            continue
        log = [json.loads(l) for l in open(os.path.join(d, 'log.jsonl')) if l.strip()]
        runs.append(dict(name=name, dir=d, meta=meta, final=json.load(open(os.path.join(d, 'final.json'))),
                         log=[r for r in log if 'test' in r]))
    return runs


def di(block):
    return None if block is None else (block['dice_net'], block['iou_net'])


def fmt(v, paper=None):
    if v is None:
        return '      -      '
    star = '*' if paper and v[0] > paper[0] and v[1] > paper[1] else ' '
    return f'{v[0]:.4f}/{v[1]:.4f}{star}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--filter', default='')
    a = ap.parse_args()
    runs = load_runs(a.filter)
    by_ds = defaultdict(list)
    for r in runs:
        by_ds[r['meta']['dataset']].append(r)
    groups = defaultdict(list)
    for ds in ('kvasir', 'clinicdb', 'colondb', 'polypgen'):
        if ds not in by_ds:
            continue
        p = PAPER[ds]
        print(f'\n## {ds}   paper (Protocol A): {p[0]:.4f}/{p[1]:.4f}')
        print(f"{'run':34s} {'ep':>7s} {'valA':>14s} {'valA+tta':>14s} {'testB':>14s} {'testB+tta':>14s} "
              f"{'valA raw':>14s} {'testB raw':>14s} {'last20':>6s} {'h':>5s}")
        for r in by_ds[ds]:
            f = r['final']
            t = np.array([x['test']['dice_net'] for x in r['log']], float)
            last20 = np.nanmean(t[-20:]) if len(t) else float('nan')
            print(f"{r['name']:34s} {f['best_epoch']:>3}/{f.get('epochs_run', len(r['log'])):<3} "
                  f"{fmt(di(f.get('val')), p)} {fmt(di(f.get('val_tta')), p)} {fmt(di(f.get('test')))} "
                  f"{fmt(di(f.get('test_tta')))} {fmt(di(f.get('val_raw')))} {fmt(di(f.get('test_raw')))} "
                  f"{last20:.4f} {f['train_hours']:5.2f}")
            groups[(ds, re.sub(r'_s\d+$', '', r['name']))].append(f)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        print('\n## seeds aggregated (mean +- std over n runs)')
        for (ds, cfg), fs in sorted(multi.items()):
            cols = []
            for k in ('val', 'val_tta', 'test', 'test_tta'):
                v = np.array([di(f[k]) for f in fs if k in f])
                if len(v):
                    m, s = v.mean(0), v.std(0, ddof=1) if len(v) > 1 else np.zeros(2)
                    cols.append(f"{k} {m[0]:.4f}+-{s[0]:.4f} / {m[1]:.4f}+-{s[1]:.4f}")
            print(f'{ds:8s} {cfg:30s} n={len(fs)}  ' + '  |  '.join(cols))


if __name__ == '__main__':
    main()
