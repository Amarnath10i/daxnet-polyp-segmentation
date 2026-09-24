# -*- coding: utf-8 -*-
"""Per-image segmentation metrics.

`dice` / `iou` follow BAMPolyp's calculate_dice / calculate_iou exactly:
binary masks, per-image score, 1.0 when prediction and GT are both empty,
averaged over images.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def per_image_scores(pred, gt):
    """pred, gt: (B,H,W) bool tensors -> dict of (B,) float64 numpy arrays."""
    pred, gt = pred.bool(), gt.bool()
    tp = (pred & gt).flatten(1).sum(1).double()
    fp = (pred & ~gt).flatten(1).sum(1).double()
    fn = (~pred & gt).flatten(1).sum(1).double()
    ps, gs = tp + fp, tp + fn
    union = tp + fp + fn
    dice = torch.where(ps + gs == 0, torch.ones_like(tp), 2 * tp / (ps + gs).clamp(min=1))
    iou = torch.where(union == 0, torch.ones_like(tp), tp / union.clamp(min=1))
    prec = torch.where(ps == 0, (gs == 0).double(), tp / ps.clamp(min=1))
    rec = torch.where(gs == 0, torch.ones_like(tp), tp / gs.clamp(min=1))
    return {k: v.cpu().numpy() for k, v in
            dict(dice=dice, iou=iou, precision=prec, recall=rec).items()}


class Evaluator:
    """Accumulates per-image metrics at network size ('net', BAMPolyp's metric)
    and at native mask size ('full'), plus MAE at native size."""

    def __init__(self):
        self.rows = {'net': [], 'full': []}
        self.names, self.mae = [], []

    @torch.no_grad()
    def add(self, logit, m_net, m_full_list, names):
        """logit: (B,1,h,w) foreground logits at network size."""
        prob = torch.sigmoid(logit.float())
        self.rows['net'].append(per_image_scores(prob[:, 0] > 0.5, m_net.to(prob.device)))
        for i, mf in enumerate(m_full_list):
            mf = mf.to(prob.device)
            p = F.interpolate(prob[i:i + 1], size=mf.shape, mode='bilinear', align_corners=False)[0, 0]
            self.rows['full'].append(per_image_scores((p > 0.5)[None], mf[None]))
            self.mae.append((p - mf.float()).abs().mean().item())
        self.names += list(names)

    def per_image(self, space='full'):
        rows = self.rows[space]
        return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}

    def summary(self):
        out = {}
        for space in ('net', 'full'):
            for k, v in self.per_image(space).items():
                out[f'{k}_{space}'] = float(v.mean())
        out['mae_full'] = float(np.mean(self.mae))
        return out
