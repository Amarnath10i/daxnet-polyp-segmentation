# -*- coding: utf-8 -*-
"""DAXNet: Domain-transform AXial state-space Network for polyp segmentation.

Kept identical to BAMPolyp so that comparisons isolate the contribution:
  * EfficientNet-B0 encoder (ImageNet V1 weights), same five stages;
  * the same U-Net decoder (ConvTranspose + DoubleConv, D1 fused with the input);
  * deep supervision at strides 16 / 8 / 4.
Changed:
  1. The Bi-Axial Mamba bottleneck + 6 flattened residual Mamba blocks are
     replaced by `n_bottleneck` geodesic bidirectional axial SSD blocks.
  2. Axial SSD blocks are also placed in the decoder at stride 16 and 8.
  3. A light boundary head on encoder features predicts an edge map that sets
     the SSD decay (edge-aware geodesic distance) and is supervised with GT
     boundaries.
  4. (optional) An image-level presence head on the globally mixed bottleneck
     predicts whether the frame contains a polyp; at inference a frame judged
     polyp-free gets an empty mask. ~19% of PolypGen frames have an empty GT
     mask, and the per-image Dice scores those 1 or 0 on the presence of a
     single false-positive pixel.
Every change has a switch in `DAXConfig` for the ablations.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from .ssd_axial import AxialGeoSSD


@dataclass
class DAXConfig:
    axial_scales: tuple = ('s32', 's16', 's8')   # where axial SSD blocks live
    n_bottleneck: int = 2
    bidirectional: bool = True
    edge_aware: bool = True
    mode: str = 'parallel'                       # 'parallel' | 'sequential'
    d_state: int = 32
    pretrained: bool = True
    encoder: str = 'effb0'                       # 'effb0' | 'pvt_v2_b0' | 'pvt_v2_b2'
    presence: bool = False                       # image-level presence head (gates the mask)

    def to_dict(self):
        return asdict(self)


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class EdgeHead(nn.Module):
    """Boundary logits at stride 4 from encoder stages E2 (s4) and E3 (s8)."""

    def __init__(self, c2=24, c3=40, mid=32):
        super().__init__()
        self.r2 = nn.Conv2d(c2, mid, 1, bias=False)
        self.r3 = nn.Conv2d(c3, mid, 1, bias=False)
        self.body = nn.Sequential(
            nn.BatchNorm2d(2 * mid), nn.ReLU(inplace=True),
            nn.Conv2d(2 * mid, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1))

    def forward(self, e2, e3):
        u3 = F.interpolate(self.r3(e3), size=e2.shape[2:], mode='bilinear', align_corners=False)
        return self.body(torch.cat([self.r2(e2), u3], 1))


def _up(x, ref):
    return x if x.shape[2:] == ref.shape[2:] else F.interpolate(
        x, size=ref.shape[2:], mode='bilinear', align_corners=False)


class EffB0Encoder(nn.Module):
    """BAMPolyp's split of EfficientNet-B0 -> E2..E5 at strides 4/8/16/32 (24/40/80/320 ch)."""
    channels = (24, 40, 80, 320)

    def __init__(self, pretrained=True):
        super().__init__()
        f = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None).features
        self.stem, self.stage1, self.stage2, self.stage3, self.stage4 = f[0], f[1], f[2], f[3], f[4]
        self.stage5 = nn.Sequential(f[5], f[6], f[7])

    def forward(self, x):
        e2 = self.stage2(self.stage1(self.stem(x)))
        e3 = self.stage3(e2)
        e4 = self.stage4(e3)
        return [e2, e3, e4, self.stage5(e4)]


class TimmEncoder(nn.Module):
    def __init__(self, name, pretrained=True):
        super().__init__()
        import timm
        self.body = timm.create_model(name, pretrained=pretrained, features_only=True,
                                      out_indices=(0, 1, 2, 3))
        self.channels = tuple(self.body.feature_info.channels())

    def forward(self, x):
        return self.body(x)


class DAXNet(nn.Module):
    def __init__(self, cfg: DAXConfig | None = None, num_classes=1):
        super().__init__()
        self.cfg = cfg = cfg or DAXConfig()
        self.encoder = (EffB0Encoder(cfg.pretrained) if cfg.encoder == 'effb0'
                        else TimmEncoder(cfg.encoder, cfg.pretrained))
        c2, c3, c4, c5 = self.encoder.channels

        ax = dict(bidirectional=cfg.bidirectional, edge_aware=cfg.edge_aware,
                  mode=cfg.mode, d_state=cfg.d_state)
        self.bottleneck = nn.ModuleList(
            [AxialGeoSSD(c5, d_inner=192, headdim=32, **ax) for _ in range(cfg.n_bottleneck)]
            if 's32' in cfg.axial_scales else [])
        self.bottleneck_refine = nn.Sequential(
            nn.Conv2d(c5, 320, 3, padding=1, bias=False), nn.BatchNorm2d(320), nn.ReLU(inplace=True))

        self.upconv4, self.decoder4 = nn.ConvTranspose2d(320, 160, 2, 2), DoubleConv(160 + c4, 160)
        self.upconv3, self.decoder3 = nn.ConvTranspose2d(160, 80, 2, 2), DoubleConv(80 + c3, 80)
        self.upconv2, self.decoder2 = nn.ConvTranspose2d(80, 40, 2, 2), DoubleConv(40 + c2, 40)
        self.upconv1, self.decoder1 = nn.ConvTranspose2d(40, 24, 2, 2), DoubleConv(24 + 3, 24)
        self.ax16 = AxialGeoSSD(160, d_inner=160, headdim=32, **ax) if 's16' in cfg.axial_scales else None
        self.ax8 = AxialGeoSSD(80, d_inner=96, headdim=16, **ax) if 's8' in cfg.axial_scales else None

        self.edge_head = EdgeHead(c2, c3) if cfg.edge_aware else None
        self.presence_head = nn.Sequential(
            nn.Linear(2 * 320, 64), nn.ReLU(inplace=True), nn.Linear(64, 1)) if cfg.presence else None
        self.final_conv = nn.Conv2d(24, num_classes, 1)
        self.deep_sup4 = nn.Conv2d(160, num_classes, 1)
        self.deep_sup3 = nn.Conv2d(80, num_classes, 1)
        self.deep_sup2 = nn.Conv2d(40, num_classes, 1)

    @staticmethod
    def _edge_at(edge, ref):
        if edge is None:
            return None
        k = edge.shape[-1] // ref.shape[-1]
        e = F.max_pool2d(edge, k) if k > 1 else edge          # keep thin boundaries
        return _up(e, ref)

    def forward(self, x, return_aux=False):
        e2, e3, e4, e5 = self.encoder(x)

        edge_logit = self.edge_head(e2, e3) if self.edge_head is not None else None
        edge = torch.sigmoid(edge_logit) if edge_logit is not None else None

        b = e5
        for blk in self.bottleneck:
            b = blk(b, self._edge_at(edge, b))
        b = self.bottleneck_refine(b)
        presence = (self.presence_head(torch.cat([b.mean((2, 3)), b.amax((2, 3))], 1))
                    if self.presence_head is not None else None)

        d4 = self.decoder4(torch.cat([_up(self.upconv4(b), e4), e4], 1))
        if self.ax16 is not None:
            d4 = self.ax16(d4, self._edge_at(edge, d4))
        d3 = self.decoder3(torch.cat([_up(self.upconv3(d4), e3), e3], 1))
        if self.ax8 is not None:
            d3 = self.ax8(d3, self._edge_at(edge, d3))
        d2 = self.decoder2(torch.cat([_up(self.upconv2(d3), e2), e2], 1))
        d1 = self.decoder1(torch.cat([_up(self.upconv1(d2), x), x], 1))

        out = _up(self.final_conv(d1), x)
        if not return_aux:
            return out
        return {'out': out,
                'deep': [_up(self.deep_sup2(d2), x), _up(self.deep_sup3(d3), x),
                         _up(self.deep_sup4(d4), x)],
                'edge': edge_logit,
                'presence': presence}


def build_model(name='daxnet', **kw):
    if name == 'daxnet':
        return DAXNet(DAXConfig(**kw))
    if name == 'bampolyp':
        from .baselines.bampolyp_ref import build_bampolyp
        return build_bampolyp(**kw)
    raise ValueError(name)
