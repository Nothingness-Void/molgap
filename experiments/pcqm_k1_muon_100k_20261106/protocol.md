# K1 optimizer comparison: 100K

Desktop-owned Track B question, 2026-11-06 JST. The user authorized Kaggle2
submission after reviewing the two-fresh-training-arm design. Shared Muon
infrastructure is not implemented at the planning source 986db6ea; its separate
implementation approval and qualification are pending. This protocol is a draft
until the exact supported optimizer implementation and recipe are committed.

## A: proposal and evidence

Question: does replacing AdamW's hidden-matrix updates with an explicitly
specified Muon/AdamW combination improve a complete K1 training endpoint without
adding model parameters or inference work?

RML queries on desktop 986db6ea and server b4443dab found no Muon experiment.
Existing K1 initialization500K evidence rejects the tested retained pretraining
initialization; late-WD and additional-tail evidence does not justify repeating
those interventions. Neither establishes that K1 needs a different optimizer.
Muon is a new hypothesis, not an identified cause of K1's scale behavior.

The [Muon paper](https://arxiv.org/html/2502.16982v1) motivates matrix-wise
orthogonalized updates. Its language-model findings are not molecular evidence.
The original100K qualified baseline is evidence
`ev-terminal-T-pcqm-k1-message-droppath-100k-s42-baseline` (selected Live
0.1412944608205557 eV). It is a drift check, not a replacement for either fresh arm.
The current SwiGLU owner and server RnC proposal remain separate questions.

## B: objections and C: ruling

[Fantastic Pretraining Optimizers](https://proceedings.iclr.cc/paper_files/paper/2026/file/ea4f0a6ad7088c9b93f22d74c0c9e8c2-Paper-Conference.pdf)
shows that tuning and the completed decay horizon can change optimizer rankings.
This pair evaluates two frozen recipes, not optimally tuned optimizer families.
[Optimizer-switching evidence](https://proceedings.mlr.press/v306/qu26e.html)
motivates starting both models from the same random initialization rather than
switching a trained Adam checkpoint to Muon.

PyTorch2.11's native Muon internally converts Newton-Schulz tensors to BF16.
The implementation must not be described as FP32 merely because model tensors
are FP32. This run retains the approved FP32/noTF32 condition; precise optimizer
arithmetic, update scaling and parameter assignment must be resolved and frozen
before source publication. The existing shared owner currently constructs AdamW.
No experiment-specific replacement trainer or runtime monkeypatch is permitted.

Ruling: qualify a complete100K pair first. No500K or full release, no automatic
successor, no early scientific victory based on an intermediate curve.

## Two newly trained arms

- `adamw`: original K1 post-BN, native AdamW, unchanged preset.
- `muon`: identical original K1, reviewed Muon on explicitly enumerated hidden
  matrices; embeddings, biases, normalization parameters, prediction head and
  non-matrix learned states retain AdamW. A shape-only `ndim == 2` rule is not a
  semantic parameter partition. Every trainable parameter has exactly one owner.

One Kaggle2 (`kaseichou`) T4x2 job, one arm per device. Both load the same original
random initialization (state SHA256 8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd),
not trained reference weights. Both execute one supervised forward per update.
No SwiGLU, RnC, teacher, consistency objective or architectural addon is included.

## Shared scientific conditions

Use the existing `k1-post-bn-100k-v1` preset: fixed TRAIN rows[0,100000),
development[100000,150000), OGB atom9/bond3/RWSE16, Gap eV, normalized L1,
physical batch128/drop-last, seed42/python-epoch-shuffle,40passes,
31240updates/3998720presentations. Keep post-BN, existing dropout and unchanged
single-forward model; FP32/noTF32 and the pinned Torch2.11 runtime.

AdamW: LR4e-4, WD1e-5, betas(.9,.999), epsilon1e-8. Shared global clip1,
cosine40passes to1e-6, no warmup. EMA.999 tracking is descriptive; primary
selection stays best development Live. All optimizer groups follow the frozen
schedule. Proposed Muon settings are momentum.95,5Newton-Schulz iterations,
epsilon1e-7 and Moonlight update scaling0.2*sqrt(max(matrix dimensions)), with
base LR4e-4/WD1e-5. These are a literature-motivated fixed configuration, not
K1-tuned settings; their final encoding waits for the reviewed shared interface.

## Endpoint and cost

Effect is candidate minus reference selected development MAE. Nomination needs
effect <=-3meV and the2000-resample paired-row95% interval upper bound<0, with
negative final-Live and last-five matched-pass mean deltas. A row interval is
conditional on the fitted models and reused development rows, not seed variance.
Report below-gate effects, uncertain effects and adverse effects without changing
the gate after observing them. Success would justify considering500K, not prove
scale transfer or authorize it.

Report selected/final MAE, complete per-pass curves, actual full-pass and worker
wall times, allocation wall/device-hours and native qualification overhead.
Checkpoint events do not count as epochs. Do not compare normalized stochastic
TRAIN loss directly with clean eV development MAE as a generalization gap.

Planning estimate: approximately2-3h allocation wall,4-6T4-device-hours, subject
to unmeasured Muon overhead. Proposed maximum4h wall/8allocated T4-device-hours
with180s cleanup reserve. A budget pause is STOP_FOR_COST, not a complete40-pass
endpoint. No hidden exposure, batch, precision or forward change to fit the cap.

## Existing workflow and closure

Use shared family training, source freezing, atomic checkpoints, runtime/native
repeat/resume qualification, epoch logging, submit_workflow.py, strict selective
retrieve_family_outputs.py and accept_run.py. The shared optimizer extension must
retain both suboptimizers' state and all group schedules under that lifecycle.
Only required strict proof files are retrieved to D:.

Each arm independently needs its complete trace/exposure, selected model,
finite aligned50000-row development predictions, actual resume/native proof,
measured native cost and observed role history. Strict comparison must permit
the declared optimizer intervention; no verifier is bypassed. Close with two
distinct complete replay entries and no exclusions before claiming replay-ready.
Official validation/test roles remain sealed; record observed data access.
Luna owns status checks. No new cron, shutdown, automatic retry or promotion.
