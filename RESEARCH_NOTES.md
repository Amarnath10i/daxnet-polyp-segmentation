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
| F9 | ~19% of PolypGen frames have an **empty GT mask** (71/376 val, 70/377 test). The per-image Dice scores such a frame 1 if the prediction is empty and 0 if it has a single foreground pixel, so ~19% of the PolypGen score is a polyp-presence test. DAX (main-a) has 13 false-positive empty frames on val vs BAM 5: that is 0.021 of its 0.026 val gap (and 0.013 of 0.015 on test). On frames with a polyp the two are equal (val .8539 vs .8600, median .955 vs .958). | per_image_*.npz: dice=0, recall=1, precision=0 |
| F10 | Protocol A's best-of-300-epochs is dominated by epoch-to-epoch noise. Std of val Dice over the last 50 epochs: BAM .006-.015, DAX (EMA) .0003-.002. Mean val Dice in the last quarter of training is equal on PolypGen (DAX .8405, BAM .8394) though val* differs by .026; on ColonDB, BAM's val* is .9286 (s42) vs .9015 (s1). | log.jsonl curves, main-a/b |

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
* **Presence gate** (`--cfg '{"presence": true}'`, from main-c on): GAP+GMP of the mixed
  bottleneck → MLP → P(polyp), BCE against [mask non-empty], weight 0.5. At inference a
  frame with P < 0.5 gets an empty mask (fixed threshold, no tuning). Motivated by F9;
  ungated metrics are logged as `<split>_raw` so the gate's effect is measured in every run.
* **Encoders** (`"encoder"`): effb0 (BAMPolyp's), pvt_v2_b0, pvt_v2_b2 (timm). With presence
  head: effb0 6.19M; pvt_v2_b0 **5.88M** (< BAMPolyp), 8.3 GFLOPs @352; pvt_v2_b2 28.6M, 25.8 GFLOPs @352.
* Final evaluation also reports flip TTA (h, v, hv) as `<split>_tta`.

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

### main-a (2026-09-25, code 110dfc5d8ab2, seed 42, Protocol A split)

Dice/IoU (`_net`). "val*" = best val epoch (Protocol A). "test@sel" = held-out 10% at that
epoch (Protocol B). "last20" = mean test Dice over the final 20 epochs (stability).

| run | val* | test@sel | last-epoch test | last20 | max test over epochs |
|---|---|---|---|---|---|
| BAM Kvasir (repro; paper .9380/.8881) | .9326/.8804 | .8989/.8420 | .9011/.8432 | .8948 | .9149 |
| DAX Kvasir | .9191/.8655 | .9009/.8406 | .9037/.8444 | .9040 | .9048 |
| BAM ClinicDB (paper .9437/.8939) | .9240/.8814 | .9155/.8709 | .9231/.8834 | .9219 | .9295 |
| DAX ClinicDB | .9320/.8857 | .9278/.8808 | .9278/.8808 | .9273 | .9293 |
| BAM ColonDB (paper .9255/.8659) | .9286/.8708 | .8908/.8277 | .8803/.8249 | .8807 | .9074 |
| DAX ColonDB | .9081/.8525 | .8913/.8267 | .8907/.8261 | .8907 | .9005 |
| DAX PolypGen (paper .8683/.8211) | .8469/.8024 | .8679/.8197 | .8702/.8218 | .8745 | .8818 |
| abl no-edge Kvasir | .9278 | .8961 | .8974 | .8970 | |
| abl causal Kvasir | .9249 | .8985 | .8950 | .8949 | |
| abl s32-only Kvasir | .9268 | .8995 | .9036 | .9035 | |

- Held-out test: DAX >= BAM on all three datasets with 0.51M fewer params and 2-3x less
  training time, but paired Wilcoxon on per-image test Dice is not significant
  (Kvasir p=.32, ClinicDB p=.66, ColonDB p=.84). Need seeds.
- BAM test fluctuates a lot across epochs (ColonDB .880-.907); DAX is flat. Best-of-300 on val inflates BAM.
- Under Protocol A, BAM's val* beats DAX on Kvasir/ColonDB: DAX val peaks early (ep 51-69) while its test keeps improving.
- Ablations (last20): edge-aware +0.7, bidirectional +0.9 Dice; multi-scale placement adds nothing (.9035 vs .9040). Single seed.

### main-b (pushed 2026-09-25, code 0b8aa3ae3221)
BAM PolypGen; recipe cross-over (DAX w/ BAM recipe, BAM w/ our recipe) on Kvasir/ClinicDB; seed-1 replicates of BAM and DAX on Kvasir/ClinicDB/ColonDB.

### main-b results (2026-09-25)
Skipped by the 9.5 h guard: dax_ours_{kvasir,clinicdb,colondb}_s1, x2_bam_oursrecipe_clinicdb_s42
(BAM PolypGen alone took 10.0 h).

| run | val* (Prot. A) | test@sel (Prot. B) | last20 test |
|---|---|---|---|
| BAM Kvasir s1 | .9286/.8787 | .8888/.8253 | .8886 |
| BAM ClinicDB s1 | .9266/.8818 | .9208/.8774 | .9227 |
| BAM ColonDB s1 | .9015/.8473 | .8980/.8366 | .8852 |
| BAM PolypGen s42 | .8731/.8283 | .8825/.8379 | .8760 |
| DAX w/ BAM recipe Kvasir | **.9331/.8839** | .8908/.8286 | .8890 |
| DAX w/ BAM recipe ClinicDB | **.9305/.8833** | .9214/.8770 | .9273 |
| BAM w/ our recipe Kvasir | NaN at ep 23 (fp16 + BAM's unstable scan, cf. F1); best before: .9100 / .8858 | | |

- Test@sel, BAM 2-seed mean vs DAX s42: Kvasir .8939 vs .9009; ClinicDB .9182 vs .9278; ColonDB .8944 vs .8913.
- PolypGen: BAM beats DAX significantly (test .8825 vs .8679, Wilcoxon p=6e-5, n=377). DAX's "ours" recipe underfits there?
- With BAM's recipe DAX beats BAM's val* under Protocol A (Kvasir .9331 vs .9326/.9286; ClinicDB .9305 vs .9240/.9266), but test drops: our recipe is the better one for test.
- TODO: rerun x2_bam_oursrecipe with --no-amp; the skipped DAX seed-1 runs.
- Paired Wilcoxon, DAX (BAM recipe) vs BAM on per-image Dice: every p > 0.17 (val and test, both BAM seeds).
- The PolypGen "loss" is mostly F9 (false positives on empty frames), not segmentation quality.

### main-c (pushed 2026-09-25 21:51, code 35b24ad10680)
Goal: beat the paper's 8 numbers (Dice/IoU × 4 datasets, Protocol A) with a model that also wins
on the held-out test (Protocol B). Pilot to choose the final encoder; every run is seed 42 with the
final protocol, so it doubles as the seed-42 entry of the final table.
PVTv2-b2 / PVTv2-b0 at 352 px + presence gate on all 4 datasets (PolypGen: 100 epochs, its training
set is 3.8× Kvasir's); effb0 @352 (resolution-only control) if time is left.
Runner now passes `--max-hours` so every job finalizes before Kaggle's 12 h limit, and skips jobs
whose `est_h` does not fit. Report: `python kaggle/report.py`.

### main-d (pushed 2026-09-25 22:27, runs alongside main-c, code f6ae5bd7e290 = same files as main-c)
Controlled comparison in BAMPolyp's own setting (effb0 encoder, 256 px, same split): DAX + presence
gate, seeds 42/1/2 on all 4 datasets (PolypGen 100 epochs), and BAMPolyp seed 2 on Kvasir/ClinicDB/ColonDB
(→ 3 BAM seeds there). Independent of main-c's outcome. From this build on, the code sha is
deterministic (content-only tar.gz); earlier shas include the gzip timestamp.

### main-c / main-d results (2026-09-26)
Protocol A (val*, Dice/IoU, net res.; native-res values agree to ±0.0015). Paper: Kvasir .9380/.8881,
ClinicDB .9437/.8939, ColonDB .9255/.8659, PolypGen .8683/.8211.

| config (params) | Kvasir | ClinicDB | ColonDB | PolypGen |
|---|---|---|---|---|
| DAX PVTv2-b2 @352 + gate, s42 (28.6M) | **.9474/.9076** | **.9524/.9111** | **.9353/.8814** | **.8851/.8420** |
|   same, held-out test (Prot. B) | .9126/.8634 | .9467/.9093 | .9281/.8681 | .8849/.8403 |
| DAX PVTv2-b0 @352 + gate, s42 (5.88M) | .9347/.8873 | .9357/.8914 | **.9313/.8740** | .8612/.8160 |
| DAX effb0 @256 + gate, 3 seeds (6.19M) | .9215/.8682 | .9303/.8842 | **.9289/.8701** | .8646/.8186 |
| BAM repro, 3 seeds (1 on PolypGen) (6.66M) | .9310/.8803 | .9259/.8822 | .9172/.8602 | .8731/.8283 |

- PVTv2-b2 beats all 8 paper numbers (single seed) and BAM on held-out test by +2.0 / +2.7 / +3.7 / +0.2 Dice.
- Same-setting comparison (effb0 @256, 3 seeds, test): DAX vs BAM Kvasir .8927 vs .8925, ClinicDB .9202 vs .9193,
  ColonDB .8962 vs .8907, PolypGen .8767 vs .8825 (BAM 1 seed): on par.
- Presence gate on PolypGen val: +0.5 (effb0), +0.5 (b0), +0.8 (b2) Dice; no effect on the other datasets.
- PolypGen timings @352/100 ep: b2 5.7 h, b0 4.4 h. effb0 @352 controls were skipped (deadline).
- NEXT: PVTv2-b2 seeds 1, 2 on all 4 datasets; operator ablation at that scale (no SSD / BAM bottleneck on PVTv2-b2).

TODO (verify before claiming): ClinicDB (612 frames) and ColonDB (380 frames) are video frames;
random frame splits then put neighbouring frames of the same video in train and val/test (cf. F4).
