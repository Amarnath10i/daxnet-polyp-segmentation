# DAXNet — Domain-transform AXial state-space network for polyp segmentation

Work in progress. Successor study to BAMPolyp (ICCVW 2025): same EfficientNet-B0 encoder and
decoder, with the bi-axial Mamba bottleneck replaced by an exact, bidirectional, edge-aware
("geodesic") axial SSD operator, placed at strides 32/16/8.

- `daxnet/ssd_axial.py` — the operator (+ linear-time reference scan, tested to 1e-10)
- `daxnet/model.py` — DAXNet; `daxnet/baselines/bampolyp_ref.py` — verbatim BAMPolyp baseline
- `daxnet/train.py` — protocols `bam` (BAMPolyp split), `pranet` (standard benchmark), `polypgen_c6`
- `kaggle/` — self-contained Kaggle notebook builder, job lists, watcher
- `RESEARCH_NOTES.md` — verified findings about the baseline, method, protocols, experiment log

```
PYTHONPATH=. python -m pytest tests/
PYTHONPATH=. python -m daxnet.train --model daxnet --recipe ours --protocol bam --dataset kvasir --data-root <data>
```
The original BAMPolyp code is expected in `repo_original/` (git clone https://github.com/farhad324/BAMPolyp).
