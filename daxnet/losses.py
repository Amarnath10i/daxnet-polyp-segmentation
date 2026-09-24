# -*- coding: utf-8 -*-
"""'ours' recipe loss: one set of weights for every dataset (no per-dataset tuning).

  L = S(out) + sum_i w_i S(deep_i) + lambda_e * E(edge)
  S : structure loss (boundary-weighted BCE + weighted IoU, PraNet)
  E : BCE (positive-reweighted) + Dice on the boundary band at stride 4
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def structure_loss(logit, mask):
    """mask: (B,1,H,W) float in {0,1}."""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(logit, mask, reduction='none')
    wbce = (weit * wbce).sum((2, 3)) / weit.sum((2, 3))
    p = torch.sigmoid(logit)
    inter = ((p * mask) * weit).sum((2, 3))
    union = ((p + mask) * weit).sum((2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def boundary_band(mask, width=5):
    """Morphological gradient of a binary mask: 1 on a `width`-px band around the border."""
    dil = F.max_pool2d(mask, width, stride=1, padding=width // 2)
    ero = -F.max_pool2d(-mask, width, stride=1, padding=width // 2)
    return (dil - ero).clamp(0, 1)


def edge_loss(edge_logit, mask):
    k = mask.shape[-1] // edge_logit.shape[-1]
    tgt = F.max_pool2d(boundary_band(mask), k) if k > 1 else boundary_band(mask)
    pos = tgt.sum().clamp(min=1.0)
    pw = ((tgt.numel() - pos) / pos).clamp(max=20.0)
    bce = F.binary_cross_entropy_with_logits(edge_logit, tgt, pos_weight=pw)
    p = torch.sigmoid(edge_logit)
    dice = 1 - (2 * (p * tgt).sum((2, 3)) + 1) / (p.sum((2, 3)) + tgt.sum((2, 3)) + 1)
    return bce + dice.mean()


class OursLoss(nn.Module):
    def __init__(self, deep_w=(0.5, 0.3, 0.2), edge_w=1.0):
        super().__init__()
        self.deep_w, self.edge_w = deep_w, edge_w

    def forward(self, outs, target):
        """outs: dict(out=(B,1,H,W), deep=[s4, s8, s16], edge=(B,1,h,w) or None);
        target: (B,H,W) long."""
        m = target[:, None].float()
        loss = structure_loss(outs['out'], m)
        for w, d in zip(self.deep_w, outs['deep']):
            loss = loss + w * structure_loss(d, m)
        if outs.get('edge') is not None and self.edge_w > 0:
            loss = loss + self.edge_w * edge_loss(outs['edge'].float(), m)
        return loss
