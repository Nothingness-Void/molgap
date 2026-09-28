# Prospective protocol — one-layer chemistry-separated K1 local propagation

Frozen question: does delaying the mixing of differently typed directed-bond
messages at layer 6 improve K1's PCQM Gap generalization, and is atom-pair
typing more useful than an equal-capacity bond-type partition?

The motivating local evidence is the accepted [GPS++ local-adapter negative]
(../pcqm_k1_gpspp_local_100k/decision.md): nine extra local adapters fitted
train but worsened development. The [relation-resolution audit]
(../pcqm_k1_relation_resolution_100k/audit/decision.md) found three small
original-role gains that all vanished on disjoint fixed500K molecules. The
[joint reconstruction decision](../pcqm_k1_joint_atom_reconstruction_100k/decision.md)
shows another original-role gain without portable superiority. The distinct
hypothesis comes from [MMGNN's atom-type-pair separation]
(https://arxiv.org/html/2606.20906v1); this experiment is not paper-faithful.

Both arms start from the identical K1 seed42 state. Only one zero-start
64-channel directed-bond adapter is added after local block 6, before the
unchanged K1 molecular exchange. The same message network and four per-node
mean pools are used in both arms:

- `neural_atom_k1_atom_pair_local`: C–C, C–N/O/S, N/O/S–N/O/S, other.
- `neural_atom_k1_bond_type_local`: OGB single, double, triple,
  aromatic/other. This is an equal-parameter category control, not a new
  benchmark reference.

All categories derive deterministically from the accepted OGB atom/bond
features. Every directed real bond is assigned once. No sidecar, geometry,
teacher, new target, loss term, prediction ensemble, changed K1 global
exchange, full-depth adapter, or official PCQM role is involved. K1's accepted
3,658,817-parameter reference is reused, not retrained. Each candidate has
3,717,121 parameters (+58,304, 1.59%).

The immutable [training contract](training_contract.json) binds accepted
cross-platform graph manifests, train/selection source-index roles, FP32 with
TF32 disabled, physical batch 128, seed42, 40 epochs, 31,240 optimizer steps,
3,998,720 sample presentations per arm, and direct Gap. Kaggle2 T4x2 runs
two independent workers with one visible GPU, optimizer, RNG, checkpoint
directory and log each. Maximum worker wall is six hours. Every 10 epochs an
atomic recovery archive is retained; a failure does not authorize a blind
retry. Preflight checks shared K1 initialization/output, real-bond category
coverage, finite updates, candidate gradients, resume equivalence and >=15%
memory reserve. Shape/permutation/disconnected/isolated synthetic checks are
required before remote release.

For this one prospectively frozen study, original-role nomination requires
gain versus immutable K1 >=0.003 eV and a favorable paired row interval.
`0.003` is a conservative *policy threshold*, not measured training variance.
The two candidate modes are compared with one another only as attribution.
The original selection role is reused and cannot serve as an independent
holdout. This study does not change any historical gate.

A separate prospective `NO_TRAIN` action may run only after independently
accepted training. It reproduces each selected model's saved original-role
predictions to <=0.001 eV per row, then infers on the accepted 50,000-row
fixed500K internal-development role with the same FP32 transform/batch.
Reference predictions are reused by checkpoint/payload hash. It reports
paired rows, bootstrap interval, gain retention and actual cost. Nomination
for a separately authorized 500K training bridge additionally requires
fixed500K gain >=0.001 eV and >=50% retention of original gain. This is
selection evidence, not full-scale superiority, and releases no automatic
training, seeds43/44, desktop handoff, or protected-role access.

Alternative explanations to record: the partition merely adds regularized
capacity; sparse color buckets become high-variance; K1 local EdgeState
already preserves the same chemistry; new gains depend on repeated dev use.
Stop on insufficient category coverage, non-nested preflight, memory failure,
terminal artifact failure, sub-threshold original gain, or failed portability.
Store source/archive/runtime/cache/prediction/checkpoint hashes, step/sample
trace, role events, costs and terminal RML separately for both arms and audit.
