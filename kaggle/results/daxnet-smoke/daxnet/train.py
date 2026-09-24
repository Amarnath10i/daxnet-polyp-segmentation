# -*- coding: utf-8 -*-
"""Train / evaluate one configuration.

  python -m daxnet.train --model daxnet   --recipe ours --protocol bam --dataset kvasir --data-root /kaggle/input
  python -m daxnet.train --model bampolyp --recipe bam  --protocol bam --dataset kvasir ...   (reproduction)
  python -m daxnet.train --model daxnet   --recipe ours --protocol pranet ...

Selection never looks at test data: 'bam' protocol keeps the epoch with the best
validation IoU (BAMPolyp's rule); 'pranet' protocol keeps the last epoch.
Test metrics are logged every epoch only so that curves can be plotted.
Outputs in --out: log.jsonl, best.pt, final.json, per_image_<split>.npz
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (BamTrainSet, EvalSet, OursTrainSet, bam_split, eval_collate, pranet_split)
from .losses import OursLoss, edge_loss
from .metrics import Evaluator
from .model import build_model

RECIPES = {
    #            bs  epochs  lr     wd
    'bam':  dict(bs=4, epochs=300, lr=1e-4, wd=1e-4),
    'ours': dict(bs=8, epochs=200, lr=2e-4, wd=1e-2),
}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def seed_worker(wid):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)


# --------------------------------------------------------------------------- #
# Model output adapters: 1-logit (ours) <-> 2-class softmax logits (BAMPolyp)
# --------------------------------------------------------------------------- #
def run_model(model, name, x, train):
    if name == 'bampolyp':
        if not train:
            l = model(x, return_deep=False)
            return {'out': l[:, 1:2] - l[:, 0:1]}
        out, ds2, ds3, ds4 = model(x, return_deep=True)
        one = lambda l: l[:, 1:2] - l[:, 0:1]
        return {'out': one(out), 'deep': [one(ds2), one(ds3), one(ds4)], 'edge': None,
                'two': (out, ds2, ds3, ds4)}
    if not train:
        return {'out': model(x)}
    o = model(x, return_aux=True)
    two = lambda z: torch.cat([-z / 2, z / 2], 1)            # softmax == sigmoid(z)
    o['two'] = tuple(two(t) for t in [o['out']] + o['deep'])
    return o


class EMA:
    def __init__(self, model, decay=0.999):
        self.m = copy.deepcopy(model).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        msd = model.state_dict()
        for k, v in self.m.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def param_groups(model, lr, wd, enc_lr_mult):
    enc_prefix = ('stem.', 'stage')
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        enc = n.startswith(enc_prefix)
        no_wd = p.ndim <= 1 or n.endswith('.bias') or 'A_log' in n or 'dt_bias' in n \
            or 'edge_log_gain' in n or n.endswith('.D')
        key = (enc, no_wd)
        groups.setdefault(key, []).append(p)
    return [{'params': ps, 'lr': lr * (enc_lr_mult if enc else 1.0),
             'weight_decay': 0.0 if no_wd else wd} for (enc, no_wd), ps in groups.items()]


@torch.no_grad()
def evaluate(model, name, loader, device, amp):
    model.eval()
    ev = Evaluator()
    for x, m_net, m_full, names in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            logit = run_model(model, name, x, train=False)['out']
        ev.add(logit, m_net, m_full, names)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='daxnet', choices=['daxnet', 'bampolyp'])
    ap.add_argument('--recipe', default='ours', choices=list(RECIPES))
    ap.add_argument('--protocol', default='bam', choices=['bam', 'pranet'])
    ap.add_argument('--dataset', default='kvasir')
    ap.add_argument('--data-root', default='/kaggle/input')
    ap.add_argument('--out', default='runs/debug')
    ap.add_argument('--cfg', default='{}', help='JSON overrides for DAXConfig (ablations)')
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--bs', type=int)
    ap.add_argument('--lr', type=float)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--eval-every', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0, help='debug: cap #images per split')
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--multiscale', type=int, default=1, help="ours recipe: 0.75/1/1.25 sizes")
    ap.add_argument('--enc-lr-mult', type=float, default=0.5)
    ap.add_argument('--edge-w', type=float, default=1.0)
    args = ap.parse_args()
    R = dict(RECIPES[args.recipe])
    for k in ('epochs', 'bs', 'lr'):
        if getattr(args, k) is not None:
            R[k] = getattr(args, k)

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda' and not args.no_amp and args.recipe == 'ours'
    torch.backends.cudnn.benchmark = True

    # ---------------- data ----------------
    if args.protocol == 'bam':
        split = bam_split(args.data_root, args.dataset)
        eval_splits = ['val', 'test']
    else:
        split = pranet_split(args.data_root)
        eval_splits = [k for k in split if k.startswith('test_')]
    if args.limit:
        split = {k: v[:args.limit] for k, v in split.items()}
    TrainSet = BamTrainSet if args.recipe == 'bam' else OursTrainSet
    train_ds = TrainSet(split['train'], size=args.size)
    g = torch.Generator(); g.manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=R['bs'], shuffle=True, drop_last=args.recipe == 'ours',
                          num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker,
                          generator=g, persistent_workers=args.workers > 0)
    eval_dl = {k: DataLoader(EvalSet(split[k], args.size), batch_size=16, shuffle=False,
                             num_workers=0, collate_fn=eval_collate) for k in eval_splits}
    sizes = {k: len(v) for k, v in split.items()}
    print('splits:', sizes, flush=True)

    # ---------------- model / optim ----------------
    cfg = json.loads(args.cfg)
    model = build_model(args.model, **cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    steps = len(train_dl) * R['epochs']
    if args.recipe == 'bam':
        from .baselines.bampolyp_ref import DeepSupervisionLoss
        crit = DeepSupervisionLoss(main_weight=0.6, deep2_weight=0.2, deep3_weight=0.2)
        opt = torch.optim.AdamW(model.parameters(), lr=R['lr'], weight_decay=R['wd'])
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
        ema = None
    else:
        crit = OursLoss(edge_w=args.edge_w)
        opt = torch.optim.AdamW(param_groups(model, R['lr'], R['wd'], args.enc_lr_mult))
        warm = max(1, min(5 * len(train_dl), steps // 10))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(
            0.005, 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
        ema = EMA(model)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    meta = dict(vars(args), recipe_params=R, params_M=n_params / 1e6, splits=sizes,
                model_cfg=getattr(model, 'cfg', None) and model.cfg.to_dict())
    json.dump(meta, open(os.path.join(args.out, 'meta.json'), 'w'), indent=1)
    log = open(os.path.join(args.out, 'log.jsonl'), 'a')
    best, best_ep = -1.0, -1
    t_start = time.time()

    for ep in range(R['epochs']):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if args.recipe == 'ours' and args.multiscale:
                s = int(round(args.size * random.choice([0.75, 1.0, 1.25]) / 32) * 32)
                if s != args.size:
                    x = F.interpolate(x, size=(s, s), mode='bilinear', align_corners=False)
                    y = F.interpolate(y[:, None].float(), size=(s, s), mode='nearest')[:, 0].long()
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                o = run_model(model, args.model, x, train=True)
                if args.recipe == 'bam':
                    loss = crit(o['two'], y)
                    if o.get('edge') is not None:          # the edge head is part of DAXNet
                        loss = loss + args.edge_w * edge_loss(o['edge'].float(), y[:, None].float())
                else:
                    loss = crit(o, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model)
            tot += loss.item(); nb += 1
        rec = {'epoch': ep + 1, 'loss': tot / max(nb, 1), 'lr': opt.param_groups[0]['lr'],
               'train_s': time.time() - t0}
        if (ep + 1) % args.eval_every == 0 or ep + 1 == R['epochs']:
            net = ema.m if ema is not None else model
            for k, dl in eval_dl.items():
                rec[k] = evaluate(net, args.model, dl, device, amp).summary()
            if args.protocol == 'bam':
                score = rec['val']['iou_net']            # BAMPolyp's selection rule
            else:
                score = ep                               # keep the last epoch
            if score > best:
                best, best_ep = score, ep + 1
                torch.save(net.state_dict(), os.path.join(args.out, 'best.pt'))
            rec['best_epoch'] = best_ep
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + '\n'); log.flush()

    # ---------------- final evaluation of the selected checkpoint ----------------
    net = ema.m if ema is not None else model
    net.load_state_dict(torch.load(os.path.join(args.out, 'best.pt'), map_location=device))
    final = {'best_epoch': best_ep, 'params_M': n_params / 1e6,
             'train_hours': (time.time() - t_start) / 3600}
    for k, dl in eval_dl.items():
        ev = evaluate(net, args.model, dl, device, amp)
        final[k] = ev.summary()
        np.savez(os.path.join(args.out, f'per_image_{k}.npz'), names=np.array(ev.names),
                 **{f'{m}_{s}': v for s in ('net', 'full') for m, v in ev.per_image(s).items()})
    json.dump(final, open(os.path.join(args.out, 'final.json'), 'w'), indent=1)
    print('FINAL', json.dumps(final), flush=True)


if __name__ == '__main__':
    main()
