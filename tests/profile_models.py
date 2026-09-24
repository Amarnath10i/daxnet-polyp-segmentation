import torch, time, sys
from torch.utils.flop_counter import FlopCounterMode
from daxnet.model import build_model
torch.manual_seed(0)
def prof(name, **kw):
    m = build_model(name, **kw).eval()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        y = m(x)
    t = time.time()
    with torch.no_grad():
        for _ in range(3): m(x)
    print(f"{name:10s} {str(kw):60s} params {n:5.2f}M  FLOPs {fc.get_total_flops()/1e9:5.2f}G  "
          f"out {tuple(y.shape)}  cpu {1000*(time.time()-t)/3:.0f} ms")
    return m
prof('bampolyp')
prof('daxnet')
prof('daxnet', axial_scales=('s32',))
prof('daxnet', mode='sequential')
m = build_model('daxnet').train()
o = m(torch.randn(2, 3, 256, 256), return_aux=True)
print({k: (tuple(v.shape) if torch.is_tensor(v) else [tuple(t.shape) for t in v]) for k, v in o.items()})
