# Node-query linear attention on the K1 local backbone

## Authority and one question

The user authorized continued pure-2D architecture discovery on Kaggle1 on
2026-09-26. One candidate, one seed, no baseline retraining; this is server
Track C and grants no desktop/full/official-role or successor authority.
V5 evidence and immutable K1-v4 scientific identity apply. Only the global
exchange architecture changes. All exact settings are in `training_contract.json`.

## Evidence funnel and alternative explanations

The desktop scale-transfer reassessment at commit cb53e973 found that early
advantages eroded during matched 500K training. The final full comparisons
were not architecture-only controls: exposure and optimization differed.
This does not prove that global attention or pure 2D is exhausted.
Authority: https://github.com/Nothingness-Void/molgap/blob/cb53e973/experiments/pcqm_scale_transfer_reassessment/decision.md

Local closures considered before this release:

- `../pcqm_k1_variants_100k/decision.md`: molecular strength gate and relation slot.
- `../pcqm_k1_portability_dual_100k/STATUS.md`: topology refresh and degree calibration
  gave sub-gate selected-role gains and negative frozen-500K portability.
- `../pcqm_k1_spd_pair_token_100k/decision.md`: shortest-path conditioning failed
  to improve the parent PairToken; do not reopen path buckets.
- `../pcqm_k1_sparse_triplet_100k/decision.md`: stronger training fit but worse
  development; do not add another persistent triplet state.
- Multi-slot, multi-head, dynamic-query, return-allocation and slot-processor
  closures are indexed by `../../ROADMAP.md`; none tested per-node kernel retrieval.

Hypothesis: a single shared slot may restrict *recipient-specific retrieval*,
even when its source pooling is adequate. Alternatives: added query freedom
overfits; a smooth positive kernel averages too uniformly; 100K is too small
to rank this architecture; selection-role reuse creates optimism. The
experiment can reject this implementation, not establish universal bottlenecks.

## Paper mechanism, not a paper-name transplant

Katharopoulos et al., ICML 2020, sections 3.2/3.2.1, equations 5 and 7:
https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf

Use phi(x)=ELU(x)+1. Compute a per-molecule K-transpose-V statistic and key sum;
each atom's query reads a separately normalized mixture. No dense N-by-N
attention tensor is formed in the encoder. Complexity is O(N*C*C), with C=64;
this is **not** a speed guarantee over K1 or dense attention on small molecules.
The paper's sequence-generation speedups are not molecular evidence.

Polynormer (https://arxiv.org/abs/2403.01232) is relevant context, not our
implementation: its polynomial local-to-global architecture and node-classification
results do not establish this molecular-regression hypothesis.

Replace K1 exchanges at layers 3/6/9. Preserve nine local GINE/GPS blocks,
64-channel real-bond EdgeState, RWSE16, OGB atom/bond features, mean pooling,
and direct Gap head. Replace rather than retain dead slot parameters. Construct
the frozen K1 first; untouched local state has identical initialization bytes.
New return maps are zero-start, so initial predictions equal K1. No residual
target, prediction ensemble, teacher, geometry or pretraining.

## Qualification and telemetry

One candidate requests one P100; there is no independent second candidate to
occupy a second GPU. Another allocated CUDA device may qualify with the same
FP32 contract. Record actual hardware; fail on numerical/identity/memory issues,
not solely card name. Pin the accepted Torch 2.4.1/cu121, PyG 2.6.1 tuple.

Remote train-role-only preflight checks: identical initial function/local bytes,
parameter identity, kernel versus explicit equation, graph separation, padding
and permutation invariance, query sensitivity, finite gradients after two
optimizer steps, and bitwise resume equivalence. At least 15% memory headroom.
No local model execution. Local tests cover syntax, manifest and policy wiring.

Record actual optimizer steps and rows, live train/dev metrics and learning rate,
each checkpoint SHA, single-device allocated interval time including evaluation
and checkpoint IO, observed role access, source/runtime identities. Atomic last
checkpoint stores optimizer, schedule, RNG; recovery archives every ten epochs.
Canonical trace is written during execution, not reconstructed from epoch guesses.
The canonical trace and role snapshot are included in recovery and completion hashes.

## Selection, portability, stopping

Selected-role material gate: >=0.003 eV mean paired gain versus frozen K1,
with favorable paired interval. This is this experiment's prospective conservative
gate, not a universal V5 threshold or an estimated seed variance. Keep terminal
and matched late-prefix metrics; report early gain erosion even if best improves.

After a mechanically complete training terminal, reproduce both best checkpoints
on the original 50K internal development role, then infer the separate fixed500K
50K internal development role once. Reuse the established frozen audit without
optimizer or checkpoint selection; ten atomic 5K chunks/model/role. The latter
role is **reused internal diagnostic data**, not sealed independent confirmation.
This tests frozen prediction portability, not 500K-trained performance.

Shortlist requires the selected-role gate, frozen audit gain >=0.001 eV and at
least 50% retention of the selected-role gain. Failure closes this exact mechanism.
Passing only proposes a separately authorized matched, late-horizon 500K bridge;
it cannot guarantee full-scale gain or release full training. Future bridge must
retain at least half the 100K gain at the preregistered late exposure and >=0.001 eV.

Training safety timeout six hours; audit timeout 1.5 hours, plus setup overhead.
Estimated 3 training GPU-hours + 1 audit GPU-hour, unverified on the new candidate.
Timeout is STOP_FOR_COST, not scientific failure; preserve outputs, no automatic
retry/extension. No healthy-run manual polling. Existing Luna B monitors only
this bound server job and hands terminal/fault evidence to A.

RML closure is pending until real artifacts pass: native prospective plan and
policy/knowledge snapshots -> observed canonical trace/roles/cost -> reference-bound
strict acceptance -> terminal finalization -> validate/rebuild -> explicit replay
entry. NO_TRAIN audit is a separate diagnostic and never labeled STRICT_CAUSAL.
