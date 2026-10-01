# G1 follow-up: complementary input versus EMA lag

## Scientific questions

A: With the accepted degree-scaled G1 initialization held fixed, does adding
the independently accepted G2 shortest-path chemical bond mean improve Gap MAE?
G1 and G2 previously improved the same original comparator, but additivity is
unmeasured. LayerNorm or duplicate chemical content could erase this benefit.

B: With the exact G1 architecture and initialization held fixed, does shortening
EMA decay from 0.9999 to 0.999 reduce selected-weight lag? G1 terminal live MAE
was lower than its EMA MAE, and late EMA was still improving. This suggests a
test, not proof of an optimal decay. The live training trajectory is a negative
control: EMA is not in the gradient path and should not change live updates.

## Frozen comparison

Both arms use accepted fixed100K train rows 0:100000, internal development
100000:150000, seed42, FP32 without TF32, physical BS128, 60 passes,
46,860 optimizer steps, 5,998,080 sample presentations, original AdamW/schedule,
and portable train-only normalization. Both start from the same accepted
degree-scaled initial tensor state. No baseline retraining or pretrained input.
Two independent processes have one visible T4, independent RNG/optimizer and
atomic per-epoch checkpoints; retained checkpoint chunks every ten epochs.
No graph construction, geometry, official validation or protected test access.

Each arm compares with frozen G1 selected EMA MAE 0.15088489169538022 eV.
A is a mechanism comparison (only path input); B is an EMA comparison (decay
and the decay-qualified selection identifier, not the selection algorithm).
They do not constitute an A/B decomposition of path and EMA together.

Prospective material gate: improvement strictly above 0.003 eV and paired
saved-prediction bootstrap supports positive sign. This is a conservative
decision threshold, NOT measured seed variance; earlier gates are unchanged.
Positive below gate is reported separately, never promoted automatically.
No automatic seeds43/44, 500K, full training or desktop handoff.

## Budget and decision

One private Kaggle3 T4x2 notebook, estimated about four wall-hours / eight
allocated T4-hours; hard local deadline six wall-hours / twelve allocated
T4-hours. Allocation cost includes both devices for the entire notebook.
Optimizer-inclusive preflight estimate must fit 4.5 training hours per arm.
Deadline failure preserves independent chunks and reports incomplete evidence.

## Replay-first closure

Freeze the real accepted G1 candidate-reference view before release. Its
original candidate trajectory, observations, scientific result and costs remain
unchanged; reuse is a hash-verified comparison-world view, not another run.
Full live/EMA trace, initial/source/data/role identities, predictions, row/target
hashes, actual runtime qualification, native allocation cost, checkpoint chunks,
paired analysis and terminal evidence are mandatory per arm.

A uses an ordinary matched-recipe replay world. B requires an explicitly
qualified EMA-intervention replay world; its differing decay must remain visible
and must not be silently treated as an architecture-only comparison. Terminal
Replay-Ready requires the actual accepted candidate/reference pair in the
rebuilt pool. Missing evidence is incomplete, not fabricated or inferred.
