# Prospective Contract

Owner: desktop; branch: codex/exp/scale-fit-retained; parent tip: 75751991.
Question: same-cohort terminal fit and weight semantics, not causal size/exposure.

Primary NEW discriminator: fixed same training-cohort fit versus development
for both actual 100K-trained and actual 500K-trained reference/joint models.
This is NOT another EMA selection experiment. Prior trajectory
TB-gptrans-local-selection-probe-20260925 and evidence
pcqm-gptrans-local-selection-probe-20260925 already evaluated epoch59
100K LIVE/EMA on the entire 50K common development role. Its accepted
1.443826 meV EMA-minus-LIVE gain effect was not material alone; preserve
that conclusion. Prior contract, decision, result, acceptance and trajectory
are hash-bound during freeze. Canonical records expose aggregates, not saved
commondev row payloads; staged development_predictions.pt files instead
contain historical 100K development rows and cannot be sliced as commondev.
The bounded 5K commondev reruns only complete the new fixed-fit comparison.

Approved role scope: source_idx [0,5000) training prefix and [500000,505000)
reused internal development, ascending order, exactly the same graphs/targets
for all six states. Never use historical 100K development [100000,150000).
The development role is consumed selection evidence, not an untouched holdout.
Reading graph shards necessarily deserializes containing 50000-row allowed-role
shards; only the approved 5000-row subsets may be evaluated or scored.
No official validation, sealed roles, selection/tuning, training, optimizer
updates, geometry generation, synthetic EMA, or automatic successor.

States: terminal epoch59 100K reference LIVE/EMA, 100K joint LIVE/EMA,
500K reference LIVE, 500K joint LIVE. 500K EMA is missing. Target transforms
come from each retained run; 500K selected bundles supply transform metadata
only, never selected weights. Strict state loading and parameter counts apply.
100K joint numerical mean/std were not independently serialized. Its transform
authority is the same hash-bound producer source and training manifest, with
both frozen trainers calling _target_stats(train_shards). This is a verified
shared derivation, not an independent numerical checkpoint-value comparison;
parent must review that limitation before execution.
Every file SHA must match independent inputs.json pins before deserialization;
all execution/source/authority files are bound by frozen_plan.json. Existing
factory implementations are frozen separately from historical producer source
identities; no full training replay/runtime qualification is claimed.

FP32, TF32 disabled, deterministic algorithms, one local CUDA device, batch128,
no autocast. Ceiling: 900 seconds from before the first model construction
through final prediction publication, including loading/collation/transfers.
The process watchdog enforces the ceiling; cooperative checks guard batches.
Maximum 60000 molecule forwards. Stop on any mismatch or non-finite value.
Incomplete output is not an accepted diagnostic or scientific failure.

Retain 12 aligned prediction bundles, exact indices/targets, tensor-state hashes,
transform identities, checkpoint/producer/frozen-source identities and file
hashes. Report per-state/cohort MAE and paired candidate/reference row errors;
row uncertainty is not training-seed variance. Do not infer under/overfitting
from endpoint absolute MAE alone or claim causality from this non-factorial
comparison. Existing paired_metrics is reused without promoting its material
gate boolean. No new gate is created.

Costs: native GPU name and software; total script wall time; synchronized
GPU-resident elapsed seconds (includes idle/loader time, not active compute);
CUDA-event forward seconds (not utilization or allocation time). Keep native
device-hours, CPU-hours and wall-hours separate. CPU-hours unknown, queue N/A.
The 0.25 device-hour ceiling is estimated, not measured. Retain partial costs
on cooperative failure and a watchdog marker if the process is terminated.

Preparation approval is not run approval. Parent must review frozen bytes,
focused tests, source equivalence limits and RML publication scope first.
