# Frozen protocol

Authorized 2026-10-29: one Kaggle3 (nvoid912) T4x2 Notebook, fixed 100K, two arms,
40 passes each, maximum 18000 Python-allocation seconds including setup and 180 seconds
cleanup, at most 10 allocated T4-device hours. The authorization to spend kaggle3 quota and
read 50K development labels was relayed by the coordinating agent, not received directly from
the user. The wall limit and hour ceiling were set by the coordinator from a measured step-time
ratio; the user gave no specific numbers. No automatic retry, continuation, monitor, protected
roles, 500K/full or adoption. If the SAM arm hits the wall limit it stays STOP_FOR_COST; the
coordinator decides any continuation.

## Evidence review and new question

Closed K1 families (2026-10-29 prior-art scan over experiments, server and archive refs):
initial residual (-0.708 meV), head-side inputs, JK readout, normalization
(BN/LN/RMS/BRN/prenorm/exposure), dropout, DropPath (+2.373 meV), MsgNorm, hop-shell,
LayerScale/adaptive gain, degree balance, RWSE refresh, SSMA, K1-G, K4, slot96, atom256, FLAG,
consistency, EMA, tail weight decay. No SAM, Lion, Adan, Lookahead or layer-wise-LR run on K1
was found. FLAG perturbs the input embedding; this question perturbs the weights.
Adjacent: every regulariser-type change at 100K has been negative or below the 3 meV gate.
Two zero-GPU checks on development predictions rejected PNA-style neighbour statistics
(increment 0.393 meV) and a long-range-deficit story (error falls with graph diameter).

Source: Foret et al., arXiv:2010.01412 (read through an automated page summary, not verified
against raw text). Question: does a weight-space sharpness penalty (rho 0.05, one value, no
tuning) reduce the train/development gap of the unchanged post-BN K1 at fixed 100K?
Alternatives: no effect; BN interaction harms; doubled gradient work changes nothing useful; a
gain is seed variation. Cheapest falsifier: one same-job 40-pass paired screen.

## Matched arms and endpoint

`baseline` reuses the retained post-BN recipe and transported initialization (preset
k1-post-bn-100k-v1). `sam` differs only by execution mode `sam`: gradient at w, perturbation
rho*g/||g||_2 (rho 0.05, global L2 over all parameters), gradient at the perturbed weights
applied to the restored weights, clip 1 on that second gradient, AdamW unchanged. Seed 42,
BS128 drop-last, FP32, LR 4e-4, WD 1e-5, 40-pass cosine to 1e-6, 31240 updates,
best-development-live.

Known confounds, stated so the comparison is not read as "only SAM":
- The perturbed forward consumes extra CUDA RNG, so later dropout masks differ from the
  baseline. Data order is fixed by the CPU sampler and is unaffected.
- The perturbed forward runs BatchNorm in train mode with momentum set to 0 and
  num_batches_tracked restored, so running statistics are not changed by it. Batch statistics
  inside that pass still use the perturbed weights.
- One seed. Row bootstrap is not training-seed variance.

Primary gate: complete candidate-minus-reference Gap MAE <= -0.003 eV and paired 2000-resample
row-bootstrap 95% upper bound below zero. Otherwise NEGATIVE_UNDER_CONTRACT, including a
favourable sub-gate gain. No retry, rho sweep, LookSAM, seed or scale-up from this run.

Declared cost readout (not a gate): wall-time ratio SAM/baseline. Measured locally on an RTX
5060, real 128-graph development batch, median of 12 steps: baseline 0.0374 s, SAM 0.0870 s,
ratio 2.34. The T4 ratio is not measured. Estimated SAM arm 12000-12400 s; the 14400 s used by
earlier screens leaves too little margin, hence 18000 s. The preflight overhead gate for `sam`
allows a step-time ratio up to 2.6.

Descriptive readouts: train-minus-dev gap per arm from canonical traces; paired curve over 40
passes; gain by atom-count stratum via experiment_attribution.analyze_family_pair.

## Roles, custody and implementation

Only the accepted official-train-derived 100K train and next 50K development roles.
Development has been reused for selection and is not an independent test. Official validation,
test-dev, test-challenge and common/OOD/P8-hard roles are not read.

Implementation: `_k1_sam_step` in `src/molgap/family_model_ops.py`; mode registered in
`k1_screen_training.py`; catalog entry `k1_sam` (fixed rho literal). The reference path is
unchanged: its 6-step CPU state digest before and after the edits is
439913965b931a23b7c11d7066d3372a1cff03f40615dcbba769d23b548cf77d (identical).
