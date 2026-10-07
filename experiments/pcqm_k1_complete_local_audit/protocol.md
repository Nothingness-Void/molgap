# Complete K1 retained-evidence and frozen-component audit — 2026-10-08

The user requested a consolidated examination of all K1 modules and a new
evidence-ranked route, explicitly on local CPU without training. Parent owns
scientific interpretation. Historical server evidence may be read from pinned
local snapshots; this does not transfer custody or query/advance server jobs.
Use this dedicated branch from desktop8e1fb4da. No accelerator, optimizer,
gradient update, new cache/geometry, remote submission, remote inference or download.

## Coverage and alternatives

Cover atom/bond encoding, RWSE, persistent edge updates, local messages, FFN,
BN, global slots, mean readout/head, Dropout/consistency, pretraining/teacher,
EMA/weight averaging, widths/new paths, exposure, execution cost and fusion.
Historical endpoints must retain rows/runtime/recipe/exposure/selection and
ownership differences. Missing causal controls are explicit, not invented.
The separate fixed late-weight averaging question is an independently planned
local diagnostic in pcqm_k1_late_weight_average, not true stepwise EMA.

The frozen test asks whether complete layerwise dependence and representation
statistics identify a plausible redundant or diluted information path in the
accepted clean-BN epoch49 model. Alternatives: every path is co-adapted and
useful; only normalization/selection differs; measured rank/variance describes
features without demonstrating a capacity limit. Frozen sensitivity cannot
establish how removal/addition would affect retraining or a theoretical ceiling.

## Identity and roles

Exact accepted epoch49 parameters, source/factory and normalization. Apply its
previously accepted native CPU clean BN buffer snapshot once, then freeze every
parameter and buffer. Use already materialized, hash-bound pure2D train16K and
development50K graphs; never decode the original whole500K cache for this test.
Trusted local checkpoint/pickle loading only after prospective publication.

Select4,096 development members uniformly without replacement using NumPy
default_rngseed20261009, sorted into source order. Independently select1,024
train-prefix and1,024 extension members from the retained16K calibration set
using the same RNG sequence, sorted within each stratum. Train descriptive
statistics use20/80 population weighting, not an unweighted50/50 pooled claim.
These subset metrics must never replace the accepted full50K endpoint.

Internal-development labels/metrics were historically selection-used. Retained
16K training graph labels are read, even though no objective is optimized;
only2K training members supply descriptive forward statistics. Development
graphs decode50K labels, while only4K receive new forward/metrics here; reading
retained full50K predictions is recorded separately. Official validation,
test-dev, test-challenge, common and OOD remain untouched.

## One fixed component matrix

All attenuation is fixed0.5 with no fit/sweep/reselection. Fourteen controls:

| Path | Intervention |
|---|---|
| Atom categorical features | halve complete atom-encoder output |
| Bond categorical features | halve complete bond-encoder output |
| RWSE | halve complete learned RWSE-encoder output |
| Persistent edge memory | reset input edge state to captured original bond embedding before learned updates2–9; keep current node endpoints and updater |
| Local messages | halve only message outputs at layers1–3,4–6,7–9 in separate cases; preserve root/skip/bias |
| Local FFN | halve FFN outputs at layers1–3,4–6,7–9 in separate cases; preserve residual/normalization |
| Global slot returns | halve residual return at layer3,6,9 separately and at all three together |

Baseline includes6K stratified train/development; each intervention predicts
only the4K development sample. Retain exact predictions, hook restoration,
unchanged state hashes and first128 identity controls. Use existing inference,
slot intervention and hashing/atomic IO owners; add only missing path hooks.
No extreme all-slot deletion is repeated to claim novelty; prior accepted
layer9/all-zero probes remain in the historical matrix.

Passive baseline observations: parameter distribution, module residual ratios,
per-layer pooled feature variance/effective rank, head feature activity,
slot assignment entropy and effective attended-node count on a bounded first
eight batches. Include atom-count and target-quartile error slices against
aligned contextual GPTrans EMA predictions. Slices are exploratory descriptions,
not a new selector/router/learnability proof. Rank is observed on this finite
sample, not a model-capacity theorem or proof that a bottleneck is saturated.
Do not replace nondifferentiated historical fit metrics with these descriptors.

## Decision and resources

Per-case paired error delta=baseline error minus intervened error. Existing
paired_bootstrap_mean1,000draws/seed20261009. This is a multi-control exploratory
mechanistic audit with no multiplicity correction, training nomination or model
promotion. Treat useful/degraded paths as frozen-state dependencies only. Any
positive attenuation signal needs a separate training comparison before an
architectural decision. Reuse direct retained comparisons where available.

Local CPU Torch2.7.1, deterministic FP32/noTF32, four intra-op threads, worker
wall ceiling600seconds. Measure worker wall/process CPU; GPU/queue not applicable.
Timers include loading, instrumentation, controls and bootstrap, exclude local
implementation/tests/RML/Git. No automatic retry or remote fallback on timeout.
Close a completed diagnostic NO_TRAIN, with evidence coverage and unresolved
discriminators, no training replay claim. Reuse plan/finalize/rebuild/check.
Reviewed reusable diagnostic implementation and evidence route to desktop
without changing the model recommendation.
