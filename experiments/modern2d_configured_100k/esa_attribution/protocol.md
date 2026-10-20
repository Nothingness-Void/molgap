# ESA optimization attribution

Authorized2026-10-21 JST for Kaggle2/3, at most three slots including the
separately bound PPGT short profile. The two ESA notebooks each have an8-hour
wall/16 allocated T4-hour ceiling (including both devices and startup), below
the human10-hour maximum. Aggregate new ESA ceiling is32 T4-hours. Do not
reserve all remaining quota merely to consume it. No automatic successors.

## Question and controls

The prior ESA pass37 log retained train0.13192 eV and Common-Dev0.21544 eV,
with no completed60-pass endpoint. This motivates optimization/regularization
attribution, not a conclusion that either cause is proven or the family failed.

One shared reference uses LR1e-3/WD1e-5. The LR slot contains3e-4 and1e-4,
with WD1e-5. The WD slot contains the reference and LR1e-3/WD1e-3. Compare each
candidate with that one completed reference; do not compare the old partial
endpoint or infer a combined low-LR/high-WD benefit. The two low-LR arms also
give a directly paired peak-LR contrast. No extra seeds or architecture change.

All four arms retain identical ESA isolate-v3/interface4, hidden192, five M/S
encoder blocks then PMA32, frozen initialization, seed42, physical BS128,
FP32/TF32off, AdamW betas/epsilon, clipping, normalized direct Gap, fixed
100K/Common-Dev50K, complete-pass shuffle, four-pass warmup/cosine shape and
46860 optimizer updates/5998080 presentations. Every arm starts from scratch;
no prior selected checkpoint is warm-started. Learning-rate and decay deltas
are explicit; they do not masquerade as architecture changes.

The Kaggle3 selected-role mirror must match the accepted Kaggle2 input files
byte-for-byte before release. No graph rebuilding or other data roles. Source
commit, recipe, target/feature/row identities, runtime and allocation are pinned.
Cross-slot effects remain seed42 contextual observations until independently
accepted; initial Modern2D context-only workflows do not manufacture a frozen
scientific reference or STRICT_CAUSAL/Replay claim.

## Execution observation and stopping

Use the existing Session profiler in every arm: native repeat-a captures two
scratch updates and must agree with unprofiled repeat/resume. During training,
warm up8 updates and capture8, with shapes/memory/stacks disabled. Retain
profile.json, atomic Chrome trace, native proofs, hardware/peak allocation,
canonical time/step/presentation/LR/checkpoint axes and per-pass train/dev logs.
Session process timing is host observation, not measured GPU kernel busy time;
allocation ledger separately charges both devices. Report instrumented windows
separately from capture-off training throughput, including loader/checkpoint cost.

At the native gate, require finite numerics, exact resume/repeat identity and
accepted input bindings. Unexpected runtime/schema/budget failure blocks the
slot rather than spawning a replacement. Stop at8 hours with the existing
600-second reserve and retain partial outputs; partial is INCONCLUSIVE, not a
completed endpoint. Do not alter BS, precision, passes or LR mid-run to finish.

## Terminal analysis

Verify actual source/job/version, producer manifests, selected/last states,
aligned predictions, role events, full allocation observations and profile hashes.
Compare train/dev curves at matched exposure; compute endpoint differences and
row-paired intervals only after alignment. Separate optimization, generalization
and execution observations; no one-seed/bootstrap-only scale conclusion.
Retain prospective, observed traces, role/cost, terminal evidence and decision
through existing RML owners. Missing finalization/reference capabilities remain
explicit; never relabel incomplete runs or force Replay. No500K/full/test role,
teacher, geometry, extra seed or automatic successor is authorized here.
