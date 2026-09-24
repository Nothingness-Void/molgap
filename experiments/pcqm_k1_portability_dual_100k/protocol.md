# K1 topology-portability dual screen — prospective contract

## Prior evidence and question

The canonical RML K1 reference improves over dense GPS at both fixed 100K and
500K. Many added relationship channels fit 100K but miss its material gate;
PairToken's selected 100K advantage attenuated on the disjoint 500K internal
development molecules before retraining. Slot count, return gates, relation
slots, pair selector/value variants, generic local adapters and extra geometry
are closed by their own decisions. This screen does not reopen them.

Two independent, small interventions probe separate missing invariances:

1. `neural_atom_k1_rwse_refresh`: a zero-start RWSE16 projection is inserted
   before the unchanged K1 molecular exchange at layers 3/6/9. Hypothesis:
   the topology signal added only at input is attenuated by nine local blocks.
2. `neural_atom_k1_degree_balance`: a zero-start, per-channel multiplier
   calibrates each *existing* local update by log directed-bond degree relative
   to the same graph's mean. Hypothesis: one shared local operator handles
   different local branching/density regimes poorly. No new message is sent.

Both use the accepted nine OGB atom categories, three real-bond categories and
RWSE16, with no coordinates, teacher, pretraining, HOMO/LUMO target, or
prediction fusion. The mechanisms are not stacked, and each is exactly K1 at
initialization. The frozen K1-v4 reference is not retrained.

## Training and post-training audit

Two independent T4 workers train the fixed 100K prefix with seed42, FP32 with
TF32 disabled, physical BS128, drop-last, AdamW 4e-4/WD1e-5, cosine40,
40 complete epochs (31,240 steps and 3,998,720 presentations per arm). Each
has separate model, RNG, optimizer, checkpoint and GPU. The data identity is
the accepted cross-platform fixed100K cache. Selection uses only its 50K
internal development role. Official validation and all test roles are sealed.

Only after *both* 100K terminal records exist, a logically separate NO_TRAIN
audit runs. It verifies the frozen original K1 checkpoint, both selected arm
checkpoints, their SHA-256 values, the accepted 100K training-only target
transform, and exact source indices. All three reproduce their saved original
50K development predictions (max absolute discrepancy <=0.001 eV). If any
reproduction fails, the fixed500K role stays unread. On success, the audit
opens only indices 500000–549999 of the accepted fixed500K internal
development shard, never its train prefix. It writes independently retrievable
atomic 5K prediction chunks for K1 and each arm, with SHA-256 and role flags.
The audit creates no optimizer and makes no model/parameter update. The 500K
role cannot tune the model; it is a once-read portability diagnostic, not an
independent untouched test or a 500K training claim.

Terminal acceptance is separate for training and NO_TRAIN audit. Each arm
must receive a prospective training trajectory, then a linked audit trajectory
with model/payload/role/chunk/cost identity. No replay-ready claim is made
until terminal evidence is validated and the derived RML check passes.

## Stop rules

The 100K material-gain gate is prospectively 0.003 eV relative to immutable
K1-v4, with a favorable aligned-row interval and finite accepted artifacts.
This is local to this contract; it does not redefine V5. The disjoint 500K
audit is used only to determine whether the observed 100K direction survives
without retraining. A 100K non-win or nonportable gain closes that exact arm.
Neither result automatically authorizes seeds43/44, 500K/full training,
official-role access, or desktop submission. A Kaggle scheduler/infrastructure
failure may be repaired only without changing this scientific contract.
