# Terminal decision: K1 PairToken value decoupling 100K

Kaggle1 kernel `nothingnessvoid/molgap-pcqm-k1-pair-value-s42` version 2
completed 40 epochs at 31,240 optimizer steps. The retained source archive,
checkpoint, model, runtime certificate, trace, and 50,000 aligned development
predictions passed the local hash and arithmetic audit in
`results/acceptance_v2.json`. Version 1 failed before training because the
selective source package omitted a cache deserialization type; version 2 added
that dependency without changing the scientific training contract.

The prediction payload recomputes to **0.13896268 eV**. Against frozen K1-v4
at **0.14137363 eV**, the point gain is **0.00241095 eV**, below the required
0.003 eV. The original PairToken at **0.13833006 eV** is better than this
candidate by **0.00063263 eV**, whereas the frozen rule requires a further
0.001 eV gain. The value-decoupling route is closed; no 500K, extra seed,
official-role, or full-scale action follows from this result.

The artifact audit is complete, but strict mechanical and comparison replay
are **not** established. The frozen mechanical gate requires deterministic
checkpoint resume, which was not independently replayed. The prospective
trajectory froze no reference evidence IDs, and no locally accepted aligned
K1-v4 prediction payload is available for its required paired interval.
Those historical gaps cannot be repaired by editing the prospective snapshot.
The RML outcome is therefore `INCONCLUSIVE` for strict contract qualification,
while the observed point comparison fails both numerical advancement floors.
The canonical candidate trace is retained with an explicit replay exclusion.

Only the fixed 100K training and 50K internal development roles were used.
Official validation, test-dev, and test-challenge remain untouched. Native
device hours and complete job wall time were not measured in retained evidence;
the prior budget estimate is not reclassified as measured cost.
