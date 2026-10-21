# Modern2D execution repair diagnostic

The human authorized audit, repair and further bounded experiments on2026-10-21
JST. This declaration releases one Kaggle3 T4x2 execution diagnostic, not another
incomplete long screen. One PPGT and one ESA worker each consume8 train-only
scratch optimizer updates: candidate repeat2+2, serialized1+1, reference2.
Full allocation ceiling:1800 seconds/1 allocated T4-hour, including startup,
native profiling, reference comparison and export. No automatic successor.

## Frozen question and evidence

The [PPGT profile](../ppgt_profile/retry_v2/results/decision_2026_10_21.md)
observed989,230 launches in two updates and graphwise whole-network dispatch.
The ESA [LR](../esa_attribution/lr/results/decision_2026_10_21.md) and
[WD](../esa_attribution/wd/results/decision_2026_10_21.md) profiles retained
forward-dominated host windows, repeated per-graph categorical/token work,
roughly166-168 capture-off train graphs/s and29/60-pass budget stops.

PPGT's9,536,769-parameter MolGap budget configuration was not the author's
published small benchmark recipe: it retained16 backbone and8 stem blocks,
hidden256, RRWP24 and10 sinusoidal bases. No claim that the PPGT family was
scientifically bad followed from one incomplete pass. ESA retained5,015,233
parameters; lower LR3e-4 was a stronger partial lead than larger decay, but
no completed matched family comparison existed. These findings supported
execution and optimization diagnosis before model-size/scale expansion.

## Changes and controls

PPGT candidate: group equal-node-count graphs and batch the existing pair stem,
backbone and head. Do not pad graphs, change attention scaling/URPE, drop edges,
change RRWP/SPE, add a feature, change parameter names or narrow the network.
The original graphwise forward remains the scientific default and reference.
Stochastic-depth/dropout configurations keep their original graphwise RNG path.

ESA candidate: build categorical one-hot features, directed-edge tokens,
isolated-atom tokens and padding in bulk. Attention/MLPs/readout, duplicate-edge
order, isolation handling and state schema remain unchanged. Retain the original
token path as forward_reference. The packed topology splitter retains all
existing geometry, protected-role, membership, categorical and safety-cap guards.

Seed42, physical BS128, FP32/TF32off, original frozen initialization, AdamW,
clipping, direct normalized Gap and accepted cross-platform input remain fixed.
ESA uses the separately evidenced LR3e-4; its reference uses exactly the same
LR, not the earlier LR1e-3. These are scratch probes, not a changed scientific
60-pass schedule or retrospective continuation of any earlier task.

## Numerical and resource gates

Synthetic CPU tests require original/packed input identity, output and gradient
closeness, state-schema preservation, companion/order invariance and fail-closed
roles. A first PPGT CPU test passed forward/gradient tolerance but failed the
strict post-AdamW state gate at attention.P.bias. That negative result was
preserved, not repaired by increasing tolerance or hiding parameters.

PPGT bucketed execution is therefore explicitly diagnostic-only. The native
worker reports strict model-update differences using unchanged atol3e-6/
rtol2e-5. UPDATE_DRIFT_RECORDED is a valid diagnostic finding, not equivalent
execution or permission for scientific training. Forward closeness cannot
establish optimizer/RNG equivalence. Both candidates must independently pass
exact repeat/resume endpoint equality; schema/nonfinite failures abort.

Retain graphwise reference state, candidate states, runtime/source/role/data
bindings, profile summaries/raw traces, peak allocation and full allocation
ledger. Cold2-update reference/candidate elapsed includes setup/checkpoint IO;
it is not a steady-state speedup. No throughput or memory benefit is presumed.
Missing full billing/provisioning/queue time stays missing, not zero.

## Role, evidence and follow-up boundaries

Only accepted train100K payload is accessed. Common-Dev permit metadata is
required by the existing configured substrate but no Common-Dev payload/metric
is allowed. No protected roles, geometry, teacher, extra seed,500K/full or local
model training/inference is released. GPU work uses existing immutable graphs;
there is no graph rebuilding. Workers checkpoint atomically and expose pinned
independent output files through the existing Session/platform owners.

Each execution_diagnostic prospective is frozen before submission, then retains
role/cost/proof artifacts and terminal NO_TRAIN evidence. It is not a training
Replay-Ready claim. The audit found that the Modern2D configured compiler uses
comparison_prelaunch=None and no accepted reference by design, not merely a
missing post-run checkbox. Scientific training is blocked until a separately
reviewed baseline/reference and trace/comparison-readiness binding exists.
The earlier partial trajectories are never relabeled or retroactively frozen.

Luna B monitors the exact server-owned job every30min, silent while healthy,
and hands terminal/error to existing A. A accepts the numerical/runtime
evidence before choosing another authorized action. B does not retry, change
code or expand the experiment. This diagnostic's authority has no automatic
retry or successor. The human's broader repair request is not a promise that
either family can beat K1/GPTrans within the compute budget.
