import torch
from daxnet.ssd_axial import geodesic_mix, sequential_reference, AxialGeoSSD

def _rand(S=3, L=37, Hh=4, P=8, N=16, big=False):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(S, L, Hh, P, generator=g, dtype=torch.float64)
    dt = torch.rand(S, L, Hh, generator=g, dtype=torch.float64) + 0.05
    lam = torch.rand(S, L, Hh, generator=g, dtype=torch.float64) * (20.0 if big else 1.0)
    Bm = torch.randn(S, L, N, generator=g, dtype=torch.float64)
    Cm = torch.randn(S, L, N, generator=g, dtype=torch.float64)
    return x, dt, lam, Bm, Cm

def test_matrix_form_equals_scans():
    for bi in (True, False):
        for big in (False, True):          # big: strong decays that break the original scan
            args = _rand(big=big)
            a, b = geodesic_mix(*args, bidirectional=bi), sequential_reference(*args, bidirectional=bi)
            err = ((a - b).norm() / b.norm()).item()
            assert err < 1e-10, (bi, big, err)

def test_block_shapes_and_identity_init():
    blk = AxialGeoSSD(40, d_inner=64, headdim=16, d_state=8).eval()
    x = torch.randn(2, 40, 12, 20); e = torch.rand(2, 1, 12, 20)
    y = blk(x, e)
    assert y.shape == x.shape and torch.allclose(y, x)   # out_proj zero-init
    for mode in ('parallel', 'sequential'):
        blk = AxialGeoSSD(40, d_inner=64, headdim=16, d_state=8, mode=mode)
        torch.nn.init.normal_(blk.out_proj.weight, std=0.02)
        y = blk(x, e); y.sum().backward()
        assert torch.isfinite(y).all()
        assert blk.edge_log_gain_h.grad is not None and blk.edge_log_gain_h.grad.abs().sum() > 0

def test_bfloat16_autocast():
    blk = AxialGeoSSD(32, d_inner=64, headdim=16, d_state=8)
    torch.nn.init.normal_(blk.out_proj.weight, std=0.02)
    x = torch.randn(2, 32, 16, 16)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y = blk(x, torch.rand(2, 1, 16, 16))
    assert torch.isfinite(y.float()).all()
