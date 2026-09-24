# DAXNet: research notes

Goal: beat BAMPolyp (Islam et al., ICCV Workshops 2025, CVAMD, pp. 1093-1103) and
produce a publishable paper. Paper PDF: `refs/bampolyp_iccvw2025.pdf` (CVF Open
Access). Original code: `repo_original/` (github.com/farhad324/BAMPolyp @ bebb67b).

## 1. Verified findings about the baseline (each has a script/evidence)

| # | Finding | Evidence |
|---|---------|----------|
| F1 | The Mamba `selective_scan` is not the Mamba recurrence. The `/ (exp(R)+1e-6)` trick underflows: relative error vs. the exact scan is 8% at L=8 (the axial bottleneck), 65% at L=64 (the flattened 6-block residual Mamba), and 80% at L=256. | `tests/scan_check_original.py` |
| F2 | Reported numbers are the best **validation** score over 300 epochs. The same split is used for model selection and reporting, and the 10% test split is never evaluated by the released code. The paper says "checkpoint with highest validation IoU is retained"; Fig. 4 marks the best validation epoch. | `repo_original/bampolyp.py` L1054-1058, L1163 |
| F3 | Validation sets are tiny: ColonDB val = **38** images, ClinicDB = 61, Kvasir = 100, PolypGen = 376. A max over 300 noisy epochs on n=38 inflates the score. | `daxnet.data.bam_split` (verified identical to original split) |
| F4 | PolypGen split leakage: 220 / 376 validation frames (59%) come from the 23 video sequences whose other frames are in train. | PolypGen listing, seed-42 split |
| F5 | Loss weights in the code (`DeepSupervisionLoss`: bce .2, dice .3, focal .2, tversky .2, iou .3, boundary .1, Tversky 0.5/0.5, gamma 2) do not match paper Table 1 (dataset-specific). | `bampolyp.py` L388-397 vs Table 1 |
| F6 | The "T0=10" cosine warm-restart scheduler is stepped **per batch**, so restarts happen after 10, 30, 70, ... iterations, not epochs. | `bampolyp.py` L491-492 |
| F7 | Complexity is under-counted: thop at 224×224 ignores the einsum/cumsum SSM ops (3.13 GFLOPs claimed). With `torch.utils.flop_counter` at the training size of 256: **3.67 GFLOPs, 6.66M params**. | `tests/profile_models.py` |
| F8 | Table 3 compares in-domain ColonDB training (0.9255) against *zero-shot* ColonDB numbers of PraNet / Polyp-PVT etc. (0.709-0.825), which were trained on Kvasir+ClinicDB only. | Table 3 vs. PraNet protocol |

## 2. Method: DAXNet

Same encoder (EfficientNet-B0), same U-Net decoder, same deep supervision as
BAMPolyp, so differences isolate the operator.

* **Geodesic bidirectional axial SSD** (`daxnet/ssd_axial.py`)
  y_t = Σ_s exp(-|Λ_t − Λ_s|) (C_t·B_s) Δ_s x_s + D x_t,  Λ_t = Σ_{k≤t} λ_k.
  - s ≤ t is the forward Mamba(-2) recurrence and s > t the backward one. This is
    exactly equivalent to two linear-time scans (unit test to 1e-10).
  - Evaluated in quadratic "SSD dual" form because axial L ≤ ~100: exact,
    stable (all exponents ≤ 0), matmul-based, AMP-safe (decays in fp32).
  - It fixes BAM's causal blind spot: a pixel in a causal H-scan never sees pixels below it.
* **Edge-aware decay (learned Domain Transform)**: λ_t = Δ_t·a_h·(1 + g_h·e_t),
  where e is a boundary map predicted from E2/E3 and supervised with GT
  boundaries. Context propagates within the polyp and is damped across its
  border. The operator is the Domain Transform recursive filter
  (Gastal & Oliveira, SIGGRAPH 2011), whose H/V, forward/backward passes it
  mirrors, with learned, input-selective distances.
* **Multi-scale placement**: 2 blocks at stride 32 + 1 block at stride 16 and 8 (decoder).
* Params 6.15M (< BAMPolyp 6.66M), 3.89 GFLOPs @256, 2.2× faster on CPU.

Related work that must be cited and distinguished:
- Hydra (NeurIPS'24): quasiseparable bidirectional SSM. Ours is a symmetric geodesic mask, applied axially.
- Boundary-conditioned Δ (BASM in CS-MUNet, arXiv 2603.19659; GARDEN, Front. Plant Sci. 2025).
  Ours: edge enters *decay only*, not the input step; it gives the DT interpretation; and it is axial and exact.
- Polyp Mamba models: Polyp-Mamba (MICCAI'24), PolyMamba (MICCAI'25), SAM-MaGuP (MICCAI'25),
  CSG-Mamba (arXiv'26), PolyMamba-Net (Front. Med. 2026).

## 3. Protocols

- **A (BAMPolyp's)**: seed-42 80/10/10, best-val-epoch val score. Only for a like-for-like comparison with the paper's numbers.
- **B (clean)**: same split; select on val, report the held-out 10% test; 3 seeds; paired Wilcoxon on per-image Dice.
- **B' (PolypGen grouped)**: video sequences kept whole (fixes F4). TODO
- **C (standard)**: PraNet train (1450) → 5 test sets, last epoch, 352 px.

## 4. Experiment log

See `kaggle/results/*/summary.json`. Kernel builder: `kaggle/build_notebook.py`.
