# K1 post-BN 100K combination screen: frozen protocol

Prospective server screen, designed 2026-11-04. One same-run seed-42 pair on
the frozen K1 post-BN recipe; the gate below is frozen before any result.

## Why

Fourteen server K1 100K screens each stayed below the 3 meV material gate (mean
delta -1.98 meV, SD 0.66 meV; [perturbation-floor protocol](../k1_perturbation_floor_100k/protocol.md)).
Only one K1 mechanism cleared the gate at 100K: PairToken on the K1-v4 recipe,
-3.04 meV with a row interval of [-3.93, -2.17] meV
([decision](../pcqm_k1_pair_token_100k/decision.md)). It has never run on the
post-BN K1/3 recipe, and no two sub-gate candidates have been composed on the
shared family path. The single same-init readout-gain sample was -0.81 meV,
row interval [-1.70, +0.06] meV ([decision](../k1_perturbation_floor_100k/results/completed/decision.md)).

## Choice of combination

The candidates were taken from the recorded screens and filtered by whether they
compose as ordered slot addons on the frozen K1/3 input ABI (`ogb-atom9-bond3-rwse16-v1`)
without changing the objective, optimizer or inputs:

| Recorded candidate | Used | Reason |
|---|---|---|
| PairToken (K1-v4 100K, -3.04 meV) | yes, as `k1_pair_token/1` | Only gate-clearing K1 mechanism; zero-start; reuses the qualified module |
| Readout gain (post-BN same-init, -0.81 meV) | yes, `k1_readout_scale/1` | Only sub-gate candidate measured on this recipe and initial state; acts at a different seam |
| MoSE family (-1.12 to -2.05 meV) | no | Needs the RWSE16+MoSE31 input sidecar, a different input ABI; stacked on PairToken it was +0.33 meV versus PairToken alone ([decision](../pcqm_k1_pair_token_mose_100k/decision.md)) |
| Functional-group and chemistry-typed tokens | no | Need chemistry inputs outside the frozen ABI |
| One-shot triplet, SPD and node-return PairToken | no | Variants of the same relation group, none better than the original PairToken |
| Joint atom reconstruction | no | Auxiliary objective, not a composable slot |
| Weight decay 1e-3 (-0.19 meV) | no | Optimizer change; mixing it with architecture would confound this architecture comparison |
| Two-pass mean training views | no | Registered `single_addon` modes; cannot compose |

PairToken plus MoSE showed that stacking can be sub-additive, so only two
mechanisms at different seams are combined, in one arm.

## Design

- One kaggle3 (`nvoid912`) T4x2 session, one arm per device, as a strict same-run
  pair: `reference` (the frozen preset retrained) and `combination` (addons in
  order `k1_pair_token/1`, `k1_readout_scale/1`, both with reviewed defaults).
- Both arms use the preset `experiments/presets/k1-post-bn-100k-v1.json`, the
  initial state `k1_reference_s42.pt` (8ef6d4ba...), seed 42, 100K train / 50K
  development, 40 passes (31,240 updates), FP32, no TF32, physical batch 128,
  selected-live development policy. Data order and dropout are fixed.
- `k1_pair_token/1` places the original PairToken module (32 pair channels, one
  learned all-pair token, zero-initialised return projection, 22,848 parameters)
  on the layer-7 `pre_message` slot. That slot receives the layer-6 mixer output,
  the state the original PairToken read; unlike the original, the update enters
  the layer-7 message input rather than the residual stream. Its parameters are
  under `_molgap_addons.pair_token`, built from a private CPU RNG seeded 42.
- The initial state loads strictly into the base model first; the addons attach
  afterwards and the substrate is sealed before the optimizer is built.

## Step-0 equivalence (CPU, 2026-11-04)

Checked through `FamilyKernel` built from the floor run's frozen recipe with the
pinned torch version replaced by the local 2.7.1 CPU build, and the frozen
`k1_reference_s42.pt`. The fixed 100K shards are not on this host, so batches
were synthetic: 128 graphs of 8-24 atoms.

- Global RNG state after model construction equal for both arms; shared state
  equal; 15 extra tensors, 22,849 extra parameters (pair token plus gain).
- Eval forward: max abs diff 0.0 (bit-equal).
- First `FamilyKernel.step`: loss bit-equal (0.94828725). Both addons are in the
  optimizer and move: gain 1.0004, pair return projection nonzero after the step.
- The family sandbox passes for `k1_pair_token/1` alone and composed with the gain.

## Cost

The perturbation-floor session used 6,840 s wall and 3.80 T4-hours with a
6,515 s readout-gain arm. The CPU step-time ratio of combination to reference
was 1.027 on the batches above. No T4 measurement of this slot exists: the
40% slowdown recorded on Kunshan DCU for the legacy PairToken trainer is not
established for T4. Estimate: 1.95 h wall, 3.9 T4-hours. Authority: 7,680 s
wall x 2 devices, ceiling 4.27 T4-hours, the same class as the floor session.

## Frozen gate

Effect = combination minus reference selected development MAE. The combination
retains a material lead only if all hold:

1. effect <= -3.0 meV (the frozen material gate);
2. the 2,000-draw paired row bootstrap 95% upper bound < 0;
3. final-live delta < 0;
4. mean delta over the last five observations < 0.

Other finite results take the usual labels as measured: `POSITIVE_BELOW_GATE`
(effect < 0 and upper bound < 0, but not all of 1-4), `INCONCLUSIVE` (interval
includes 0), `NEGATIVE_UNDER_CONTRACT` (effect > 0 with lower bound > 0).
For scale only: init-only SD 1.074 meV and paired SD 1.519 meV
([decision](../k1_init_noise_server_100k/results/completed/decision.md)); the gate
is 1.98 paired SD. The row interval is not seed uncertainty, and development
rows are reused for selection.

## Stop points

- Cost: the frozen 7,680 s wall. Reaching it pauses the run (`STOP_FOR_COST`);
  no extension, continuation or retry without new authority.
- Science: if |effect| < 1.5 meV (the floor protocol's stop point), the
  combination of these two addons closes at 100K with no successor variant.
- The reference is expected to reproduce 141.2945 meV bit-identically; any drift
  is reported, and the gate is still read against the same-run reference.

## What is reported

Selected development MAE per arm, the effect with its row interval, final-live
and last-five deltas, the trained gain value, reference drift against 141.2945
meV, and wall and T4-hours from the allocation ledger.

## Limits

No automatic seeds 43/44: multi-seed confirmation needs a material lead and an
explicit compute decision. No 500K or full training: the PairToken-to-500K
promotion is closed ([decision](../pcqm_k1_pair_token_scale_attribution/results/round3_decision.md)).
No protected roles. Closure follows the usual pair shape: the reference closes
`CLOSED` and the combination's outcome is read from the gate above.
