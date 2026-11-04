# K1 post-BN 100K same-init perturbation floor

Prospective server measurement, designed 2026-11-04. This is a measurement: it has
no pass/fail gate, and no automatic verdict, promotion or retained lead can follow.
It makes no claim that a scalar gain helps.

## Why

Init-only noise on this setup is SD 1.074 meV (n=5; paired SD 1.519 meV; the
3 meV gate is 1.98 paired SD). The 14 sub-gate server K1 100K screens (mean delta
−1.98 meV, SD 0.66 meV) each compared a zero-start perturbation against one shared
reference at the same initial state. Their relevant noise is the same-init
perturbation floor, which has not been measured.

## Design (option B)

- One kaggle2 (`kaseichou`) T4x2 session, one arm per device, as a strict same-run
  pair: `reference` (the frozen preset retrained) and `readout_scale` (addon
  `k1_readout_scale/1`, config null: one learnable scalar gain on the pre-readout
  embedding, initialised to 1).
- Both arms use the same preset (`experiments/presets/k1-post-bn-100k-v1.json`),
  the initial state `k1_reference_s42.pt` (8ef6d4ba...), seed 42, 100K train /
  50K development, 40 passes, FP32 and physical batch 128. Data order and dropout
  are fixed.
- The planned design A (two gain arms, readout and aggregate, compared offline
  with ref_s42) cannot be compiled with the current front door. A same-run
  reference must keep the preset's empty addon list, and an unpaired K1 arm needs a
  retained reference plan, which the post-BN preset does not provide (`reference: null`).
  The pair therefore gives one strict floor sample. As a by-product, the reference
  arm is a third seed-42 rerun, measuring rerun drift against the retained 141.2945 meV.

## Step-0 equivalence (CPU, 2026-11-04)

Both addons were checked through the owning CPU path (`load_family_model` and
`FamilyKernel` from the post-BN recipe and `k1_reference_s42.pt`, torch 2.7.1 CPU,
deterministic). The fixed 100K shards are not on this host, so the batch was
synthetic: 16 graphs, 5-11 atoms.

- Shared state equal. The only extra parameter is the gain, value 1.0.
- Forward output: max abs diff 0.0 (bit-equal) for both `k1_readout_scale` and
  `k1_aggregate_scale`.
- One `FamilyKernel.step`: loss bit-equal to the reference (0.23396946). The gain
  is in the optimizer and receives a nonzero gradient: readout 1.79e-3 (gain 0.9996
  after the step), aggregate −5.49e-4 (gain 1.0004).

Multiplying by a gain of exactly 1.0 is exact in IEEE arithmetic, so the arms are
identical at step 0 and diverge only through the gain's own updates.

## What is reported

- Selected development MAE per arm (meV).
- The delta readout_scale − reference with its paired-row bootstrap 95% CI.
- Reference rerun drift against 141.2945 meV.
- Comparison with init-only noise (SD 1.074, paired SD 1.519) and with the 14
  sub-gate deltas (mean −1.98, SD 0.66).
- If design A is run later, also the two deltas' mean and absolute maximum.

Falsification: the hypothesis "the same-init floor is below 1.5 meV" is falsified
if |delta| ≥ 1.5 meV. One sample (two with design A, df=2) is weak evidence. The
row CI is not training noise. Development rows are reused for selection, which is
optimistic.

## Limits

No successor, extra seeds, continuation, 500K, full training or adoption. No
protected roles. Closure follows the usual pair shape: the reference closes
`CLOSED`, and the candidate's outcome is a human decision under this no-gate protocol.
