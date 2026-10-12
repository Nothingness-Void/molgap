# Detailed retained-output review

2026-10-12 JST. Supplements [decision](decision.md), without changing its gate,
launching work or claiming terminal RML closure. Machine-readable checks:
[detailed audit](detailed_audit.json). No training/model inference or new
molecular-role consumption; tensor checkpoint inspection ran on CPU.

## Identity and qualification

Four execution owners match the frozen `cd615aa4` source byte-for-byte after
Git newline normalization: K1 trainer, determinism helper, CPU-layout helper,
Kaggle pair runtime. All28 retained metadata SHA256 pins were rechecked.
Both arms use observed Torch2.11.0+cu128 and identical runtime/software/
determinism/fixture fingerprints. Existing `validate_runtime_preflight` passes
on the actual retained provenance, architecture and certificate, not merely a
summary boolean. Candidate reference-step fingerprints match for both native
fixture steps, including gradients, model/BN, AdamW moments and RNG; resume and
selected-state roundtrip deltas are zero. Qualification remains fixture-scoped.

The actual final optimizer has one native AdamW group,313 parameter states,
foreach=null, fused=null, weight-decay1e-5 and LR1e-6. Every optimizer state
acknowledges31240steps. Both terminal schedulers/cursors agree. No AMP, TF32,
EMA, extra stochastic forward or schedule reset is part of this candidate.
Final trace LR is the pre-scheduler-step LR; optimizer checkpoint LR is after
the40th scheduler update, so that difference is intentional, not a mismatch.

## Numerical coverage and comparison boundary

All50000 source indices are exactly100000..149999 in order. Source IDs, targets
and predictions are byte-equal, not just rounded-score equal. All367 selected
and final model-state tensors, all retained final optimizer tensors, scheduler,
RNG and cursor agree. Per-epoch LR, train/dev metrics and exposure agree for
all40 observations. There is no retained fingerprint for every one of the31240
training steps; do not claim independently verified per-step bitwise equality.

Recomputed MAE is0.141294464468956eV with float32 reduction and
0.1412944608205557eV with float64 reduction. This explains the tiny trace versus
inspection decimal difference; it is not arm drift. No prediction differences
exist to localize to a molecular slice.

"Original" means the registered shared trainer's `reference` mode, not the
earliest historical script. Both arms already share selective H2D transfer and
deferred scalar metric collection. For this remote Spec `_execution_scope`
returns the common default execution options; the local-only
`original_execution=True` path is not selected. The incremental candidate adds
CPU layout reuse and scoped TRAIN safety-fill removal, retains native AdamW
and restores fill after the step; clean evaluation remains common. Therefore
7.31% is the incremental saving over this shared baseline, not the total saving
versus every old K1 implementation. This does not change the frozen pair gate.

## Timing structure

The frozen trainer synchronizes CUDA at TRAIN end and development end. TRAIN
timing includes loading, transfers, forward/backward, optimizer and scalar
collection; it is not GPU-kernel-only time. Pipeline minus TRAIN/development
is24.164054s for reference and24.153392s for candidate across40epochs: selected/
last IO and bookkeeping do not explain the roughly379s TRAIN difference.
Native fixture median step time decreases3.34%; that fixture is not a full
epoch workload. The local synthetic23% estimate is not a T4 full-run result.

These are post-hoc descriptive windows, not replacement gates or independent
replications:

| Epoch window | Reference TRAIN,s | Candidate TRAIN,s | Time reduction |
|---|---:|---:|---:|
|1-10|1338.503631|1214.511410|9.26%|
|11-20|1326.896224|1201.659318|9.44%|
|21-30|1319.615867|1200.794633|9.00%|
|31-40|1201.918304|1190.680844|0.93%|

Reference TRAIN epochs1-37 mostly take130-134s after the first epoch; epochs38-40
take94-95s. Candidate is relatively stable near118-123s. Cumulative pipeline
intervals place candidate completion around5911s and reference epoch37 completion
around5930s from each arm's own epoch-loop origin. This is consistent with the
reference suffix experiencing less peer contention, but exact shared wall
overlap is not recorded: loading/start offsets and per-epoch absolute timestamps
are absent. There are no CPU utilization, GPU frequency/power, device crossover
or solo-allocation controls. Peer exit, resource contention and hardware effects
cannot be causally separated. Do not diagnose a CPU-only bottleneck or convert
this into a measured whole-allocation saving.

The complete40epoch TRAIN gate still uses7.31237%, not a chosen prefix. Even
the post-hoc first30epoch reduction9.23586% is below10%. There is no justification
for changing the gate, declaring a larger speedup, or extrapolating A100/500K.

## Remaining evidence gaps

The per-arm mechanical checks and native numerical qualification pass; joint
runtime nomination fails only on measured speed. Runtime preference/defaults
remain unchanged. The improvement is observed with no numerical loss in this
specific run, rather than evidence that all scales and graphs are safe.

Formal terminal package/comparison-readiness publication, observed canonical
role/cost events, RML finalization and Git routing remain pending. The role plan
still says consumption pending; it is a declaration, not an observed ledger.
Do not relabel this record replay-ready until the existing closure validators
accept the actual custody. Source inspection and outputs support the released
TRAIN/development execution, not a new protected-role authorization. No missing
ledger, utilization or platform-version observation was invented.

## Local regression verification

Executed with the project venv, owner `PYTHONPATH` and `CUDA_VISIBLE_DEVICES=''`:
`test_k1_initialized_execution.py`, `test_k1_execution_layout.py`,
`test_k1_screen_training.py`, `test_experiment_family_workflow.py`.
Result:153 passed in193.26s, one upstream PyG deprecation warning. These are
local CPU/synthetic regressions, not a rerun of native Torch2.11 qualification.
Production implementation was not edited. Documentation whitespace checking
passed with the repository's CR-at-EOL preservation convention.
