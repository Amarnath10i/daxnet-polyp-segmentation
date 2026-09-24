# -*- coding: utf-8 -*-
"""Geodesic bidirectional axial state-space mixer (the core operator of DAXNet).

For one axis (a row or a column of length L) with per-position, per-head
log-decay lam_t >= 0, input u_s = dt_s * x_s and selective projections
B_s, C_t in R^N, the mixer computes

    y_t = sum_s  exp(-|Lam_t - Lam_s|) * (C_t . B_s) * u_s  +  D * x_t,
    Lam_t = sum_{k<=t} lam_k .

For s <= t this is exactly the Mamba recurrence h_t = exp(-lam_t) h_{t-1} + B_t u_t
run forwards; for s > t it is the same recurrence run backwards. The decay
between two pixels is the exponential of their *geodesic* distance along the
axis, which is the Domain Transform recursive filter (Gastal & Oliveira, 2011)
with learned, input-dependent distances. The edge prior e_t enters only the
decay, lam_t = dt_t * a_h * (1 + g_h * e_t), so a predicted boundary acts as a
barrier that context cannot cross cheaply.

Axial sequences are short (L = H or W <= ~100), so the operator is evaluated in
its quadratic matrix ("SSD dual") form: exact, stable (every exponent is <= 0),
and a few batched matmuls. `sequential_reference` gives the equivalent linear
time forward + backward scans and is used by the unit tests.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def geodesic_mix(x, dt, lam, Bm, Cm, bidirectional=True):
    """
    x   : (S, L, Hh, P)  per-head inputs
    dt  : (S, L, Hh)     input step sizes (>0)
    lam : (S, L, Hh)     log-decays (>=0)
    Bm, Cm : (S, L, N)
    returns (S, L, Hh, P)
    """
    dtype = x.dtype
    acc = torch.promote_types(lam.dtype, torch.float32)          # decays never in half
    lam = lam.to(acc)
    cum = torch.cumsum(lam, dim=1)                                # (S, L, Hh)
    dist = cum[:, :, None, :] - cum[:, None, :, :]                # (S, t, s, Hh)
    if bidirectional:
        dec = torch.exp(-dist.abs())
    else:                                                         # causal (Mamba) ablation
        L = x.shape[1]
        causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
        dec = torch.exp(-dist.clamp(min=0.0)) * causal[None, :, :, None]
    cb = torch.einsum('stn,sun->stu', Cm.to(acc), Bm.to(acc))     # (S, t, s)
    kern = (dec * cb[..., None]).to(dtype)                        # (S, t, s, Hh)
    u = x * dt[..., None].to(dtype)
    return torch.einsum('stuh,suhp->sthp', kern, u)


def sequential_reference(x, dt, lam, Bm, Cm, bidirectional=True):
    """Linear-time forward + backward scans; must equal `geodesic_mix`."""
    S, L, Hh, P = x.shape
    u = x * dt[..., None]
    a = torch.exp(-lam)                                           # (S, L, Hh)
    h = x.new_zeros(S, Hh, P, Bm.shape[-1])
    yf = []
    for t in range(L):
        h = a[:, t, :, None, None] * h + u[:, t, :, :, None] * Bm[:, t, None, None, :]
        yf.append((h * Cm[:, t, None, None, :]).sum(-1))
    y = torch.stack(yf, 1)
    if not bidirectional:
        return y
    g = x.new_zeros(S, Hh, P, Bm.shape[-1])
    yb = [None] * L
    for t in reversed(range(L)):
        decay = a[:, t + 1, :, None, None] if t + 1 < L else 0.0
        g = decay * g + u[:, t, :, :, None] * Bm[:, t, None, None, :]
        yb[t] = (g * Cm[:, t, None, None, :]).sum(-1)
    yb = torch.stack(yb, 1)
    diag = (Cm * Bm).sum(-1)[:, :, None, None] * u                # counted twice
    return y + yb - diag


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class AxialGeoSSD(nn.Module):
    """
    Residual block:  x + out_proj( norm(Y_h + Y_w) * silu(z) )
    with Y_h / Y_w the geodesic mixer along columns / rows.

    mode = 'parallel'   : both axes read the same input (BAMPolyp style sum)
           'sequential' : the width pass reads the height pass output, so one
                          block already couples every pixel with every other.
    """

    def __init__(self, dim, d_inner=None, headdim=32, d_state=32,
                 bidirectional=True, edge_aware=True, mode='parallel',
                 dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        d_inner = d_inner or 2 * dim
        assert d_inner % headdim == 0, (d_inner, headdim)
        self.d_inner, self.headdim, self.d_state = d_inner, headdim, d_state
        self.nheads = d_inner // headdim
        self.bidirectional, self.edge_aware, self.mode = bidirectional, edge_aware, mode

        self.norm_in = LayerNorm2d(dim)
        self.in_proj = nn.Conv2d(dim, 2 * d_inner, 1, bias=False)
        self.dwconv = nn.Conv2d(d_inner, d_inner, 3, padding=1, groups=d_inner)
        n_axis = 2 * d_state + self.nheads
        self.axis_proj = nn.ModuleDict({a: nn.Conv2d(d_inner, n_axis, 1, bias=False)
                                        for a in ('h', 'w')})
        # Mamba-2 initialisation: dt ~ logU(dt_min, dt_max), A = 1..16 per head
        for a in ('h', 'w'):
            dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
                           + math.log(dt_min))
            self.register_parameter(f'dt_bias_{a}',
                                    nn.Parameter(dt + torch.log(-torch.expm1(-dt))))
            self.register_parameter(f'A_log_{a}', nn.Parameter(
                torch.log(torch.linspace(1, 16, self.nheads))))
            # edge gain g = exp(edge_log_gain), starts at 1 (decay doubled on edges)
            self.register_parameter(f'edge_log_gain_{a}',
                                    nn.Parameter(torch.zeros(self.nheads)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm_out = LayerNorm2d(d_inner)
        self.out_proj = nn.Conv2d(d_inner, dim, 1, bias=False)
        nn.init.zeros_(self.out_proj.weight)       # block starts as identity

    def _axis(self, v, edge, axis):
        """v: (B, d_inner, H, W) -> mixed along `axis`, same shape."""
        B_, _, H, W = v.shape
        proj = self.axis_proj[axis](v)
        Bm, Cm, dt = torch.split(proj, [self.d_state, self.d_state, self.nheads], dim=1)
        dt = F.softplus(dt.float() + getattr(self, f'dt_bias_{axis}')[None, :, None, None])
        A = torch.exp(getattr(self, f'A_log_{axis}'))[None, :, None, None]
        lam = dt * A
        if self.edge_aware and edge is not None:
            gain = torch.exp(getattr(self, f'edge_log_gain_{axis}'))[None, :, None, None]
            lam = lam * (1.0 + gain * edge.float())
        if axis == 'h':      # sequences run down each column
            pat_in, pat_out = 'b c h w -> (b w) h c', '(b w) h c -> b c h w'
            kw = dict(b=B_, w=W)
        else:                # sequences run along each row
            pat_in, pat_out = 'b c h w -> (b h) w c', '(b h) w c -> b c h w'
            kw = dict(b=B_, h=H)
        seq = lambda t: rearrange(t, pat_in)
        xs = seq(v).unflatten(-1, (self.nheads, self.headdim))
        y = geodesic_mix(xs, seq(dt), seq(lam), seq(Bm), seq(Cm),
                         bidirectional=self.bidirectional)
        y = y + xs * self.D[:, None].to(y.dtype)
        return rearrange(y.flatten(-2), pat_out, **kw)

    def forward(self, x, edge=None):
        v, z = self.in_proj(self.norm_in(x)).chunk(2, dim=1)
        v = F.silu(self.dwconv(v))
        if self.mode == 'parallel':
            y = self._axis(v, edge, 'h') + self._axis(v, edge, 'w')
        elif self.mode == 'sequential':
            y = self._axis(self._axis(v, edge, 'h'), edge, 'w')
        else:
            raise ValueError(self.mode)
        y = self.norm_out(y) * F.silu(z)
        return x + self.out_proj(y)
