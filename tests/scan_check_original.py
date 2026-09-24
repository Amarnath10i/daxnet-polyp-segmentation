import torch, math, torch.nn.functional as F
torch.manual_seed(0)
def selective_scan(u, delta, A, B, C, D):          # verbatim from repo_original/bampolyp.py
    A = torch.clamp(A, min=-5.0, max=5.0)
    dA = torch.einsum('bld,dn->bldn', delta, A)
    dB_u = torch.einsum('bld,bld,bln->bldn', delta, u, B)
    dA_cumsum = torch.cat([dA[:, 1:], torch.zeros_like(dA[:, :1])], dim=1)
    dA_cumsum = torch.flip(dA_cumsum, dims=[1])
    dA_cumsum = torch.cumsum(dA_cumsum, dim=1)
    dA_cumsum = torch.clamp(dA_cumsum, max=15.0)
    dA_cumsum = torch.exp(dA_cumsum)
    dA_cumsum = torch.flip(dA_cumsum, dims=[1])
    x = dB_u * dA_cumsum
    x = torch.cumsum(x, dim=1) / (dA_cumsum + 1e-6)
    y = torch.einsum('bldn,bln->bld', x, C)
    return y + u * D
def reference(u, delta, A, B, C, D):               # exact recurrence h_t = exp(dA_t) h_{t-1} + dB_t u_t
    A = torch.clamp(A, -5, 5); b,l,d = u.shape; h = torch.zeros(b,d,A.shape[1],dtype=u.dtype); ys=[]
    for t in range(l):
        h = torch.exp(delta[:,t,:,None]*A)*h + delta[:,t,:,None]*B[:,t,None,:]*u[:,t,:,None]
        ys.append((h*C[:,t,None,:]).sum(-1))
    return torch.stack(ys,1) + u*D
d, n = 192, 96                                      # model_internal_dim, model_states of the original ModelArgs
A = -torch.exp(torch.log((torch.arange(1,n+1).float()/n*3).repeat(d,1)))   # original init
for L in [8, 16, 32, 64, 256]:
    u = torch.randn(4,L,d); B=torch.randn(4,L,n); C=torch.randn(4,L,n); D=torch.ones(d)
    delta = F.softplus(torch.randn(4,L,d)*0.5)     # ~ softplus(dt_proj(.)) at init
    for dt in [torch.float32]:
        y0 = selective_scan(u,delta,A,B,C,D); y1 = reference(u,delta,A,B,C,D)
        rel = ((y0-y1).norm()/y1.norm()).item()
        # AMP: emulate fp16 for the exp/cumsum part
        y2 = selective_scan(u.half().float(),delta,A,B,C,D)
        print(f"L={L:4d}  rel.err(orig parallel scan vs exact) = {rel:.3e}")
# fp16 (as used under torch.autocast in bampolyp2) at L=64
L=64; u = torch.randn(2,L,d); B=torch.randn(2,L,n); C=torch.randn(2,L,n); delta=F.softplus(torch.randn(2,L,d)*0.5)
y1 = reference(u,delta,A,B,C,torch.ones(d))
y16 = selective_scan(u.half(),delta.half(),A.half(),B.half(),C.half(),torch.ones(d).half()).float()
print("fp16, L=64: rel.err =", ((y16-y1).norm()/y1.norm()).item(), " nan:", torch.isnan(y16).any().item())
