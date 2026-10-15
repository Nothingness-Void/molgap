# K1 execution protocol — 2026-10-15

The user authorized one dual-arm100K execution comparison and profiler capture
after reviewing the desktop port. Use one private Kaggle3 T4x2 allocation, one
independent worker/device per arm, seed42, no extra arm/seed or automatic retry.
The selected account must pass an actual quota check immediately before POST.

`original` executes the retained K1/2 epoch loop in `k1_screen_training.py`;
`optimized` executes K1/3 through `family_training_runner.py`. These are
interface versions of the same3,658,817-parameter K1, not two architectures.
The original path retains per-epoch loaders, GPU dense-layout construction,
whole-batch transfer and scalar metric synchronization. The shared path uses
persistent inputs, CPU layout/mask reuse, selective transfer, buffered metrics
and atomic session outputs. Its observer is opt-in; ordinary training stays plain.
Both execute their existing loops; no replacement training loop is introduced.
This compares the delivered infrastructure as a whole, not the mask change alone.
The old entry is retained executable code, not an execution of the entire old
repository checkout. Frozen package bytes bind both paths and the observers.

## Matched scientific fields

Both: unchanged single-forward normalized-Gap L1, raw seed42 initialization
state SHA8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd,
accepted fixed100K manifest1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d,
train0..99999/internal development100000..149999, atom9/bond3/RWSE16, no
geometry model input, physicalBS128/drop-last, identical Python epoch order,
40passes/31,240updates/3,998,720presentations, FP32/noTF32, AdamWlr4e-4,
wd1e-5, native defaults, clip1.0, epoch cosine40 to1e-6, noEMA, best-development
live selection. Use the frozen original float32 train mean5.3383002281188965
and sample std1.275090217590332, not a newly reduced transform.
Both checkpoint each complete pass. The shared finite guard every50updates
checks existing math; it does not change the update. Pin Python3.12,
Torch2.7.1+cu126, PyG2.6.1, OGB1.3.6 through the existing environment owner.
No official validation/test/shadow/500K/full role is authorized.

Loader process counts/cache/CPU RNG ownership and output implementations differ
as part of the execution intervention. GPU RNG and full-state native repeat/
serialized-resume are qualified per owner; compare actual saved predictions
and selected epochs rather than assuming a bitwise full-run endpoint.
The old evaluator reports float32 MAE; the shared buffered evaluator and final
independent comparison use float64 reductions. Record this reporting difference.
CPU contention and device assignment remain single-allocation limitations.

## Measurements and acceptance

Both recipes skip120 updates then capture16 with shapes/stacks/memory disabled.
The profile observes actual training updates, not disposable extra formal work.
Each owner must pass its source-bound native repeat/resume gate; the profiled
two-step repeat must match plain model/optimizer/RNG state and emit real CUDA
events. An all-arm barrier prevents either formal arm starting before both pass.

Rank phase host-self and operator CPU/device time, retain Chrome traces and
report actual calls, copies, masks/indexing, dispatch waits and memory evidence
when observed. Do not add overlapping device times or treat host phases as GPU
utilization. Unvisited validation/checkpoint phases are unmeasured.

Primary speed: per-arm train worker wall from the same parent clock. Also
report full observed allocation, preflight/setup and idle tail separately.
A retained complete-pass span (first loader request through last update return)
after capture has no active Kineto
but still has passive Python observer wrappers; label this explicitly, not as
a pristine profiler-free control. Synchronize only its boundaries. Compare
medians/spread across complete post-capture passes, graphs/s and wall reduction;
passes from one run are not independent seed replicates. No extrapolated gain
is an observed gain, and no new speed threshold is invented after results.

Independently verify manifests, hashes, complete exposure, row/target alignment,
finite selected/last weights and optimizer/RNG state. Recompute both50K MAEs
in float64, signed delta, max/mean prediction delta, selected-pass meaning and
trace completeness without model inference. Keep execution success, numerical
comparison and scientific promotion separate. This is a noncausal execution
diagnostic; no strict architecture Replay/promotion or successor is authorized.

Ceiling:7wall hours/14allocated T4 hours including setup, with60s stop reserve.
At the ceiling stop and retain acknowledged checkpoints; no automatic resume.
Existing output hooks persist atomic checkpoint/trace and recovery chunks.
Remote output is independently retrievable; no long transient-only continuation.
A missing/unknown submission outcome requires exact reconciliation before retry.
