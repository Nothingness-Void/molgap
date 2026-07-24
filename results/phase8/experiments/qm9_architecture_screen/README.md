# QM9 Architecture Screen

## Purpose

Use QM9 as a cheap architecture-elimination gate before PubChemQC 100K and
learning-curve validation. This experiment does not select a production model.

## Fixed first-level protocol

- Targets: HOMO, LUMO, and Gap in eV.
- Formal split: 100,000 train / 10,000 validation / 10,000 test, split seed 42.
- First pass: one training seed per candidate.
- Promotion: retain only two or three materially competitive candidates, then
  repeat those with three seeds.
- Candidates: GINE6, GPS7, GPS9, SchNet, equal GINE+GPS prediction blend, and
  GPS+SchNet standard embedding fusion.
- Inspiration arm: TensorNet reuses the prior PubChemQC A/B winner to test
  whether its advantage transfers to QM9 DFT and deployment-like ETKDG geometry.
- All encoders export frozen embeddings so combinations do not retrain them.
- Compute gate: the lightweight SchNet `176/160/6` training time is `1.00`.
  Candidates measured above `2.00` on the same data protocol are stopped
  regardless of accuracy.

## Geometry control

- `DFT`: official QM9 optimized coordinates; diagnostic geometry-quality arm.
- `ETKDG`: canonical SMILES -> ETKDGv3 -> MMFF, through
  `src/molgap/graphs.py`; this is the primary deployment-relevant 3D arm.
- ETKDG graphs are atomically cached in 2,000-molecule shards. An interrupted
  build resumes from completed shards instead of regenerating the full split.
- Molecules that cannot become a valid deployment SMILES or an ETKDG conformer
  are recorded explicitly. Paired 2D/3D claims use only aligned test rows.

## Run status

- GINE6 seed 42: complete; test average MAE 0.10152 eV.
- GPS7 seed 42: complete; test average MAE 0.09160 eV.
- GPS9 seed 42: complete; test average MAE 0.08634 eV.
- Equal GINE6+GPS9 prediction blend: complete; test average MAE 0.08830 eV,
  worse than GPS9 alone.
- SchNet DFT-geometry seed 42: complete; test average MAE 0.05638 eV.
- SchNet ETKDG-geometry seed 42: complete. ETKDG succeeded for
  111,451/120,000 rows (92.88%); test average/Gap MAE is
  0.09440/0.11130 eV. DFT geometry is therefore an optimistic diagnostic,
  not a deployment estimate.
- GPS9+SchNet-ETKDG standard fusion: all frozen-encoder head seeds 42/43/44
  beat the same aligned GPS9 baseline. Mean test average/Gap MAE is
  0.07653 +/- 0.00024 / 0.09107 +/- 0.00029 eV, versus aligned GPS9
  0.08419/0.10036 eV. Mean average-MAE gain is 0.00767 eV. This proves head
  stability only; encoder seeds 43/44 have not been trained.
- Same-row frozen-head architecture matrix (mean +/- sample standard deviation,
  three head seeds, 9,306 test rows):

| Inputs | Average MAE (eV) | Gap MAE (eV) |
|---|---:|---:|
| GPS9 baseline | 0.08419 | 0.10036 |
| GPS9 + GPS7 | 0.08231 +/- 0.00027 | 0.09787 +/- 0.00037 |
| GPS9 + SchNet-ETKDG | 0.07653 +/- 0.00024 | 0.09107 +/- 0.00029 |
| GPS9 + GPS7 + SchNet-ETKDG | **0.07556 +/- 0.00048** | **0.08967 +/- 0.00083** |

GPS7 improves every paired triple-head seed, but its mean marginal gain after
GPS9+SchNet is only 0.00096 eV average and 0.00140 eV Gap. This motivates a
validation-selected routed GPS7 call rather than assuming GPS7 must run for
every molecule.
- A validation-selected threshold on base predicted Gap chose a 100% GPS7
  route, so simple Gap routing provides no compute saving on QM9. The routed
  test result equals the always-triple result.
- DFT-geometry diagnostic fusion reaches 0.05141 eV for GPS9+SchNet and
  0.05129 eV for GPS9+GPS7+SchNet. The latter is 0.02428 eV better than the
  ETKDG triple fusion, while GPS7 adds only 0.00012 eV average and slightly
  regresses Gap under DFT geometry. The dominant remaining bottleneck is
  single-conformer ETKDG noise, motivating a maximum-two-conformer test.
- Two-conformer diagnostic completed. Seed 43 succeeded on 111,463/120,000
  rows; the exact two-view intersection contains 92,228 train, 9,233
  validation, and 9,238 test molecules. Frozen SchNet prediction averaging
  improves test average/Gap MAE from 0.09431/0.11121 and
  0.09447/0.11114 for the individual views to 0.09146/0.10774 eV.
- Width screen: GPS9-160 and GPS9-128 keep the nine-layer/four-head protocol,
  batch size, optimizer, and epochs fixed; only 2D hidden/embedding width changes
  from the 192-dimensional reference.
- GPS9-160 completed with 2.599M parameters, test average/Gap MAE
  0.08856/0.10621 eV, and 767 seconds training. It saves 30.4% parameters but
  is slower and less accurate than GPS9-192. Its triple fusion also regresses
  to 0.07675/0.09070 eV, so it is only a memory-constrained fallback.
- GPS9-128 was stopped after the one-epoch timing gate: 1.668M parameters and
  24.8 seconds/epoch provided no runtime gain over GPS9-192 and had a worse
  first-epoch validation trajectory than GPS9-160.
- GPS11-160 completed with 3.167M parameters, test average/Gap MAE
  0.08642/0.10308 eV, and about 31 seconds/epoch. It is effectively tied with
  GPS9-192 alone but slower. Replacing GPS7 with GPS11 in the triple system is
  promising: GPS9+GPS11+SchNet-ETKDG reaches three-head-seed mean
  **0.07441 +/- 0.00035 / 0.08836 +/- 0.00038 eV**, versus
  0.07556/0.08967 for GPS9+GPS7+SchNet. This is a QM9 precision candidate, not
  a production promotion; its extra complete-system compute must be weighed in
  PubChemQC 100K.
- GPS9-192 mean+max pooling completed at 0.08687/0.10385 eV versus
  0.08634/0.10347 for mean pooling. It is closed.
- Frozen GPS9 multiscale layers 2/4/9 and 5/7/9 preserve the original GPS9
  prediction path but do not replace a second GPS expert. With two-conformer
  3D and a 256-wide gate they reach 0.07484/0.08879 and
  0.07545/0.08935 eV, respectively, versus 0.07392/0.08781 for
  GPS9+GPS7 and 0.07341/0.08707 for GPS9+GPS11 on head seed 42.
- Exact-intersection conformer matrix (three frozen-head seeds):

| 2D inputs | 3D input | Average MAE (eV) | Gap MAE (eV) |
|---|---|---:|---:|
| GPS9 | one conformer | 0.07644 +/- 0.00030 | 0.09088 +/- 0.00053 |
| GPS9 | two-conformer mean | 0.07544 +/- 0.00034 | 0.08954 +/- 0.00048 |
| GPS9 + GPS7 | one conformer | 0.07548 +/- 0.00021 | 0.08940 +/- 0.00045 |
| GPS9 + GPS7 | two-conformer mean | 0.07382 +/- 0.00033 | 0.08758 +/- 0.00076 |
| GPS9 + GPS11-160 | one conformer | 0.07466 +/- 0.00032 | 0.08882 +/- 0.00047 |
| GPS9 + GPS11-160 | two-conformer mean | **0.07333 +/- 0.00037** | **0.08700 +/- 0.00053** |

All nine paired two-conformer comparisons improve both average and Gap MAE.
Two-conformer embedding concatenation is worse than embedding averaging.
- On the best frozen inputs, plain concatenation fusion regresses to
  0.07460/0.08846 eV versus gated fusion at 0.07333/0.08700 eV.
- Gate-head width 128 uses 125,699 parameters and matches width 192:
  0.07329/0.08700 versus 0.07333/0.08700 eV. Width 256 uses 366,083
  parameters and improves to **0.07285 +/- 0.00013 /
  0.08629 +/- 0.00028 eV**. Width 192 is dominated.
- Input LayerNorm was screened once and closed after regressing the 256-wide
  gate by 0.00089/0.00130 eV on average/Gap.
- Two-conformer SchNet training augmentation completed at 1,114.9 seconds,
  1.87 times the 594.9-second SchNet reference. On the exact two-view
  intersection its seed42/seed43/averaged test average MAE is
  0.08725/0.08712/0.08438 eV and Gap MAE is
  0.10368/0.10324/0.10035 eV.
- An equal-step duplicate-seed42 control took 1,061.2 seconds and reached only
  0.09293/0.10993 eV on seed42 and 0.09323/0.11003 eV on seed43. Averaging
  its two conformer predictions reaches 0.08897/0.10520 eV. The augmented
  model's 0.00458/0.00485 eV advantage over that averaged control isolates a
  real conformer-diversity benefit rather than merely twice as many updates.
- The two-conformer-trained lightweight SchNet is a much better standalone 3D
  predictor but a worse
  direct fusion expert. GPS9+GPS11+augmented-SchNet two-view fusion reaches
  0.07403/0.08818 eV, versus 0.07285/0.08629 for the single-conformer-trained
  lightweight SchNet. Encoder MAE alone is therefore not a fusion promotion
  criterion.
- The single-conformer-trained and two-conformer-trained lightweight SchNet
  checkpoints remain complementary when each is run once on the primary
  conformer. Their complete fusion predictions are blended using one
  validation-selected global weight:

| 2D inputs | Single-conformer SchNet weight | Average MAE (eV) | Gap MAE (eV) | Decision |
|---|---:|---:|---:|---|
| GPS9 | 0.45 | 0.07188 +/- 0.00034 | 0.08548 +/- 0.00021 | PubChemQC 100K minimal candidate |
| GPS9 + GPS7 | 0.46-0.47 | 0.07146 +/- 0.00016 | 0.08493 +/- 0.00028 | PubChemQC 100K cost candidate |
| GPS9 + GPS11-160 | 0.46-0.49 | **0.07081 +/- 0.00053** | **0.08427 +/- 0.00049** | PubChemQC 100K precision candidate |
| GPS9 + GPS11-160 + GPS7 | 0.45-0.49 | 0.07041 +/- 0.00038 | 0.08378 +/- 0.00044 | Accuracy ceiling only |

Adding GPS7 to the minimal row improves only 0.00043/0.00055 eV, so both are
retained until PubChemQC measures whether that small gain transfers. The
three-GPS row improves only 0.00040/0.00050 eV over the GPS11 row while adding
the full GPS7 forward, so it is not promoted. All uncertainty above is from
frozen-head seeds 42/43/44; encoder seeds have not been repeated.
- Selecting separate weights for the two lightweight SchNet checkpoints for
  HOMO, LUMO, and Gap provides no gain over one global weight. The global
  blend is retained.
- An unconstrained per-target affine calibration improves the precision row by
  only 0.00003 eV average MAE, below head-seed variation. It is not retained.
- Bounded residual identity path (`+/-0.20 eV`) is stable and improves GPS9 by
  0.00705 eV average, but at 0.07714 eV it is 0.00061 eV worse than the
  standard gate. Tighter `+/-0.05` and `+/-0.10 eV` bounds lose more accuracy.
- TensorNet DFT-geometry seed 42: stopped after epoch 0. Its 103.4 seconds per
  epoch is about 5.22 times SchNet on the same QM9 DFT run. The prior
  PubChemQC 10K A/B also measured 3.73 times SchNet, so it exceeds the `2.00`
  hard gate on both same-protocol comparisons.
- EGNN DFT-geometry: closed after the formal run. It used 0.467M parameters
  and 457 seconds for 30 epochs (about 0.77 times SchNet), but test average
  MAE was 0.09301 eV versus SchNet-DFT's 0.05638 eV. It passes compute and
  fails accuracy, so no ETKDG arm is authorized.

Entrypoint and exact commands: `scripts/architecture/README.md`.
