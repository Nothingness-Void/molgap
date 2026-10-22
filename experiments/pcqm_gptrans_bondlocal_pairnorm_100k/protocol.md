# Frozen protocol and evidence review

Authorized2026-10-23 JST: Kaggle3 T4x2, one same-job reference/candidate,
fixed100K,60passes, at most4hours. Quota observed7.653729 platform GPU hours;
use a tighter13500second Python-allocation ceiling (3.75wallhours, at most
7.5allocated T4-device hours), including setup/preflight and180seconds cleanup.
Platform quota debits are not measured device costs. No retry/automatic resume.

## Evidence and new question

- [PairNorm100K](../pcqm_gptrans_pair_norm_100k/decision.md) selected
  `pair_update_norm`; accumulated-state `pair_post_norm` was worse.
- [PairNorm500K](../pcqm_gptrans_pair_norm_500k/decision.md) improved1.140meV,
  below3meV and NEGATIVE_UNDER_CONTRACT, not an adopted mechanism.
- [Resistance attribution](../pcqm_structural_bias_norm_100k/terminal_acceptance/attribution.md)
  records BondLocal EMA.999 and resistance's1.119meV weak signal. It explicitly
  lacks structural cohorts; no evidence establishes70% simple molecules,
  projection-noise causality, exponential variance or universal architecture ceilings.
- [K1 exposure](../pcqm_k1_norm_exposure_100k/terminal_decision.md) already tests
 160passes/lowerLR/warmup. Other active K1 owner branches are independent;
 do not duplicate them or assume removing BN guarantees accuracy/speed.
- Targeted RML and owning-source search found no BondLocal+pair-update-norm
  training. Historical noisy-node+PairNorm is a different intervention.

The new hypothesis is conditional: normalize each fresh pair update across
its32channels before adding to persistent pair, retaining BondLocal. This may
help, have no effect, or suppress useful magnitude. It does not prove bounded
logits or zero computation cost. Do not sum historical gains or promise an endpoint.

## Matched execution

Reference `bond_local`: retained degree-scaled BondLocal,5871201parameters,
12layers/node256/pair32/local64, degree initialization scale.0897.
Candidate `pair_update_norm`: identical tensors/schema/count, with
`pair = pair + normalize_pair(pair_update)` in each wrapped core. Reuse the
existing `normalize_pair`; do not normalize accumulated pair state.

Reuse the transported initialization and old fixed-data recipe exactly:
seed42,100000train/next50000development,46860optimizer steps,
5998080presentations, physicalBS128/drop-last, FP32/noTF32, AdamW LR.001,
WD.05, four-pass warmup/cosine60/minLR1e-6, L1/clip1, EMA.999,
best-development-EMA. Both arms retain identical16update profiler windows.
New all-arm native repeat/resume qualification is required before formal training.
Shared family/session/checkpoint/output/retention owners remain unchanged.

Primary decision: complete paired candidate-minus-reference MAE <=-.003eV,
with2000row-bootstrap95%CI entirely below0. Report numerical effect and CI
separately. Absolute .1415/.1425 scores are context, not substitute gates.
No early-stop from partial curves. At cap retain STOP_FOR_COST partial evidence,
not a complete or negative scientific endpoint. Row bootstrap is not seed variance.

## Roles, retention and follow-up

Only official-train-derived fixed train/development roles. They are reused
selection roles, not independent held-out evaluation. No3D, official validation,
test-dev/test-challenge, sealed20K or PubChemQC evaluation.
Retain immutable source/config, exact job/version, native proofs, allocation
ledger, best/last state, full LR/step/sample/checkpoint-bound curve, aligned
predictions and final manifest. Use existing minimal retention/retrieval and RML.
Mechanical acceptance, science and replay readiness stay separate.
After terminal acceptance write failure-mode attribution before another module.
Fixed fusion may be described only from permitted aligned predictions later;
learned gating needs genuine OOF or separately frozen fit/evaluation roles and
is not authorized here. No production/default change or500K/full launch.
