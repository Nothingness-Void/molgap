# Protocol - 2026-10-14

Question: where does the latest shared execution spend time for the selected
K1 and GPTrans recipes on Kaggle T4, including CPU waits and checkpoint work?
The user selected fixed100K complete training plus profiling. Fresh training
is explicitly authorized to measure current execution and retain its endpoints;
it does not reopen rejected scientific modules or retrain a missing strict reference.

Two independent arms in one T4x2 Notebook, one GPU each, seed42, FP32/noTF32,
physicalBS128, drop-last, direct PCQM4Mv2 Gap, existing OGB9/bond3 caches.
K1 retains RWSE16, nine local layers and its one molecular slot; mean2 uses
two stochastic supervised forwards, native AdamW4e-4/wd1e-5, cosine40 and
best-development-live selection. GPTrans retains12x256/pair32, corrected
degree initialization,64-channel true-bond residual, foreach AdamW1e-3/wd.05,
warmup4/cosine60, EMA.999 and best-development-EMA selection.
Exact values, initialization identities and normalization remain in config/recipes.
There is no pretrained model load, geometry generation or ensemble in this run.

Dataset: accepted Kaggle1 private fixed100K version identity pinned by manifest
SHA256 in config. Train0..99999, development100000..149999; both derive from
official train. Internal development is repeatedly selected, not independent
held-out generalization. Official validation/test-dev/test-challenge forbidden.

K1 completes31240 updates/3998720 presentations; GPTrans46860/5998080.
Schedules differ by family. Cross-family MAE/cost is descriptive, not a strict
causal architecture comparison or an early-stop calibration.
Each arm keeps atomic best/last model, optimizer/schedule/EMA/RNG/cursor state,
canonical trace, aligned predictions, runtime and cost identity under
`/kaggle/working/experiment/<arm>`. Source/init/prospective records remain
independently retrievable in the private source dataset.

Native all-arm barrier: two-step repeats and serialized resume must match
exactly. One repeat enables the profiler; its report/trace hashes and actual
CUDA events must validate before formal training. No unsupported profiler fallback.
Formal capture skips120 updates and records16, spanning checkpoint128.
It adds no optimizer update or extra data role; no per-phase synchronization.
Shapes/stacks/memory recording off. Evaluation outside the window remains
unmeasured by the profiler; canonical epoch traces retain the broader run.
Instrumentation time cannot be called unprofiled throughput or GPU busy time.

Notebook ceiling7wall hours with180s cleanup reserve; at most14 assigned
T4 device-hours, not platform billing or an observed cost. This retains the
bounded six-hour-per-arm/one-hour setup envelope of the GPTrans100K route.
At the ceiling, checkpoint and STOP_FOR_COST; no automatic continuation/retry.
Estimated3-5wall hours, CPU/queue unknown. CPU parsing/cache admission precedes
GPU qualification; accelerators only serve native training/profiling.

Accept actual completeness/exposure, sealed-role flags, source/package/init/
recipe/cache hashes, native proof, finite predictions, trace/cost and profile
report/trace SHA. Attribution names measured hotspots and remaining gaps.
Only a separately controlled, profiler-off equivalence/throughput test can
justify an execution optimization. This run cannot change production/defaults.
