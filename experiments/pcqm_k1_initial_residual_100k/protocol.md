# Frozen protocol

One Kaggle3 (`nvoid912`) T4x2 Notebook, fixed 100K, two arms, 40 passes each, one device per arm.
Maximum 14400 Python-allocation seconds including setup, 180 seconds cleanup, at most 8
allocated T4-device hours; platform quota debits, queue and measured native cost are different
quantities. No automatic retry, continuation, monitor, 500K/full or adoption.

Authorization provenance: the user's Kaggle3 real-submission authorization and the use of the
50K development labels for metrics were relayed in the coordinator's phase-2 task messages
(2026-10-28) to the executing agent. The executing agent did not receive them from the user
directly and records them as relayed, not independently verified. Protected roles are not
authorized and are not used.

## Evidence review and new question

[Scale attribution](../pcqm_k1_scale_attribution_20261025/analysis.md) and
[CURRENT_STATE](../../CURRENT_STATE.md) ask for one still-unidentified information-flow or
optimization choice, not stacked small gains. The
[K1 audit](../pcqm_k1_complete_local_audit/analysis_zh.md) shows pooled-state participation rank
falling from about 35 (blocks 4-5) to 6.9, 4.4 and 2.5 (blocks 7-9) and a strong dependence of the
frozen model on its input encodings (atom encoder half-scale +116 meV, RWSE +17.7 meV). It also
warns that low rank does not prove lost information.

Question: can every local block 2-9 benefit from reading the layer-0 node state h0 (atom-category
embedding plus RWSE16 encoder output) through one zero-initialised per-channel gate, as in
GCNII's initial residual?

Prior-art check (closed list and K1 historical matrix, see `PLAN.md` kept outside Git):
- Closed or sub-gate neighbours: RWSE refresh (zero-start RWSE16 projection before the exchange at
  layers 3/6/9; 100K gain 1.007 meV, below the 3 meV gate, then 1.124 meV regression on fixed 500K
  development rows), degree balance, MsgNorm, message DropPath, normalization families, slot
  width/placement, PairToken family, readout/JK. The multi-depth readout is closed on EdgeState
  (`../pcqm_gap_architecture/architecture_route_audit.md:19`) and is not reused here.
- Difference from RWSE refresh: the source is the actual h0 (atom identity included), it enters the
  conv input of every block 2-9 instead of the residual stream at three mixer layers, and the
  addon has 1,536 parameters. It is the same family of idea (early information re-entering deep
  layers), so a sub-gate result is the expected outcome. Not a seed or schedule variation.
- h0 re-injection and GCNII-style gating: no record found in experiments, RML reviews or the
  historical matrix (grep and targeted reads; not every file was opened).

Literature (abstract and ar5iv page fetches; details not fetched are marked unconfirmed):
[GCNII, arXiv:2007.02133](https://arxiv.org/abs/2007.02133) mixes H(0) into every layer with
weight alpha (0.1-0.2 typical) and reports that initial residual alone helps more than identity
mapping alone on node classification at depths up to 64; no graph-level or molecular task, ablation
variance unconfirmed. [DeeperGCN, arXiv:2006.07739](https://arxiv.org/abs/2006.07739) concerns
residual-path design for deep GCNs on OGB; datasets and numbers unconfirmed. Transfer to 9-layer
graph regression is the author's inference. [DropEdge, arXiv:1907.10903](https://arxiv.org/abs/1907.10903)
frames over-smoothing and over-fitting as the two obstacles; the K1 100K history (train better,
development worse for many added paths) points mostly at over-fitting and is the reason for the low prior.

## Mechanism (addon `k1_initial_residual`, version 1)

Nine one-based `PRE_MESSAGE` taps, one `SlotBinding` per layer. Layer 1 passes its node tensor
through unchanged and stores it as h0 (not detached). Layers 2-9 replace the conv input `x` by
`x + g_l * h0` with `g_l` a zero-initialised 192-vector (eight vectors, 1,536 parameters). The block
residual `norm1(local + x)` keeps the unmodified `x`, so h0 reaches the output through the conv
(value, key, query, root weight), not through a direct residual add. The holder is a plain dict,
overwritten by layer 1 and cleared by layer 9, so evaluation and `no_grad` passes leave nothing.
At step 0 the function equals the baseline exactly. The new keys live under `_molgap_addons` and
are not in the frozen initial state; the shared family loader exact-loads the base state first and
attaches addons afterwards.

## Matched arms and endpoint

`baseline` reuses the exact retained post-BN recipe and transported initialization.
`initial_residual` differs only by the registered addon. Both use seed 42, physical BS128/drop-last,
FP32/no-TF32, AdamW LR 4e-4/WD 1e-5, 40-pass cosine to 1e-6, no warmup, normalized L1/clip 1,
EMA .999 tracked but primary `best-development-live`, 31240 updates and 3998720 presentations,
existing accelerated CPU layout and checkpoint path. Required native repeat/resume proofs must
pass for both arms before formal training.

Pre-registered outcomes (development role, paired 2000-resample row bootstrap, seed 42):
- gain = MAE(baseline) - MAE(initial_residual), same job.
- POSITIVE_UNDER_CONTRACT: gain >= 3.0 meV and 95% lower bound above zero. Consequence: nominate
  for independent seed replication and a 500K transfer question only.
- NEGATIVE_UNDER_CONTRACT: gain below 3.0 meV, including any favourable sub-gate gain. No retry,
  seed or scale-up follows from this run.
- Mechanism readout (descriptive, not a gate): per-layer L2 norm and max absolute value of the eight
  gates at the selected epoch. If every L2 norm is below 0.05 (author's round-number choice, h0
  channels are O(1)), the path was not used and the hypothesis is unsupported whatever the MAE.
- Also descriptive: train-minus-development MAE gap from the canonical traces and gain by
  atom-count stratum through the existing attribution helper.
- Row uncertainty does not measure training-seed variation. At the resource cap retain
  STOP_FOR_COST; partial exposure cannot establish a complete negative or successful endpoint.
  Report throughput and allocation separately; neither substitutes for the quality gate.
- Author's prior (judgement, not measurement): about 10-15% chance of clearing 3 meV.

## Roles, custody and implementation

Only the accepted official-train-derived 100K train and next 50K development roles. Development has
been reused for selection and is not an independent test. No official-valid, test-dev/challenge,
sealed 20K, common/OOD/P8-hard, PubChemQC or geometry. Retain best/last state, LR/steps/samples/
checkpoint-bound curve, aligned finite predictions, native qualification, immutable source/config
and allocation ledger. Use existing terminal attribution and RML closure, not handwritten acceptance.

`baseline_data_declaration.json` is a byte copy of the Message DropPath projection of the historical
K1 post_bn Spec, not a new accepted reference. The same-job control is trained in the physical job.
The addon implementation is new; the Message DropPath implementation (commit 927783ba) lives only on
the archive ref and is not merged. The allocation-budget plumbing it introduced already exists on the
parent tip of this branch.
