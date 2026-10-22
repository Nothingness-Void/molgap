# Structural falsifier protocol

Frozen on2026-10-23 JST under the human's explicit research/submission authority.
One Kaggle3 T4x2 allocation:7 session-quota hours /14 physical T4-hours maximum,
including bootstrap, native qualification, profiling, training and publication.
The existing600-second checkpoint reserve and hard-stop guard remain in force.
This is one release, not permission for an autonomous unbounded search.

## Evidence and hypotheses

The [accepted prepared screen](../modern2d_configured_100k/prepared_pair/results/decision.md)
completed20 passes, with PPGT mean MAE0.2004566171 and ESA MAE0.2036746615 eV.
These are frozen within-family contextual endpoints, not K1/GPTrans scores.
The [reference-byte inventory](reference_assets.json) freezes the actual retained
contracts, selected weights, predictions, traces and output manifests before release.
The [summary](../modern2d_configured_100k/prepared_pair/results/summary.json)
pins custody and independent acceptance. No completed baseline is retrained.

PPGT's shared structural pair tensor is constructed by its input stem and
reused across backbone layers. GPTrans's content/relation coupling motivates
testing conditional *use* of that structure, without copying its whole trunk.
The new `ppgt_content_pair/1` shares a rank16 conditioner across attention
layers: `p_ij' = p_ij * (1 + 0.5*tanh(Wout*tanh(Wleft*h_i+Wright*h_j)))`.
Wout starts at zero; all original parameters and mean readout are retained.
This is neither recurrent pair memory nor a claimed faithful author variant.
The extra7168 weights are bounded; node width/depth/slots do not change.

ESA's normalized attention and PMA/mean cannot infer multiplicity when all
edge tokens are identical. This is a code-derived expressivity limitation,
not evidence that it explains any particular residual subgroup.
`esa_count_readout/1` injects only `log1p(atom_count)` through zero-start FiLM
on the pooled graph representation. The scale multiplier is bounded to
[0.5,1.5]; an independent learned shift is permitted. The512 extra weights
do not introduce RWSE, geometry, a new tokenizer or a second encoder.
Graph count comes from the accepted batch boundaries, never labels.

Relevant primary sources: [PPGT](https://arxiv.org/html/2504.12588v4),
[ESA](https://arxiv.org/html/2402.10793), and the retained local
[GPTrans propagation](../../src/molgap/gptrans.py).
Our conditional-pair and count-FiLM edits are explicit adaptations, not
mechanisms asserted to be present in those papers.

## Frozen execution and data

Independent candidate, RNG, optimizer, checkpoints and one visible T4 per arm.
Both reuse accepted indexed UnifiedV2 train100K/Common-Dev50K bytes and the
same two frozen initial base states as the completed prepared screen. Fresh
addon parameters are deterministic seed42 and zero-return; no warm start.
Physical BS128, FP32/TF32off,20 passes,15620 updates and1999360 presentations
per arm; same train-only target mean/std, normalized Gap L1, gradient clip1,
AdamW and four-stage warmup/cosine schedule. PPGT retains LR1e-3/WD1e-5;
ESA retains LR3e-4/WD1e-5. Selection is minimum live Common-Dev MAE.
Different family recipes preclude a causal PPGT-versus-ESA ranking.

Prepared execution is opt-in; both *new addon recipes* must pass fresh native
repeat/resume and accepted-backend shadow checks before any formal update.
A failed gate stops both arms without fallback. Lightweight profiling captures
four updates after eight warmups, then detaches. Retain complete pass traces,
selected/last states, predictions/source IDs/targets, runtime/source pins,
full allocated cost and original execution reports. No parsing/graph rebuild,
new conformer, protected/Audit/official role, extra seed,500K/full or teacher.

## Interpretation and terminal plan

Use saved predictions to independently recompute MAE and align exactly with
the appropriate frozen within-family reference. Report online train/Dev
curves in consistent eV units, time/exposure axes, parameter count, subgroup
dependence (only if already available), throughput and whole allocation cost.
Do not execute additional graph generation or inference merely for analysis.

A gain of at least0.003 eV is a *predeclared exploratory triage criterion*,
not a universal V5 threshold, causal proof, seed-variance estimate or scale
release. Row resampling describes fixed-row uncertainty only. Smaller gains
remain below this question's triage gate; negative changes close that edit.
Even a material positive only nominates a separately reviewed follow-up.

The configured Modern2D owner supports CONTEXT_ONLY initial screens, not a
qualified strict-reference release. The original reference trajectories were
not Replay-Ready. Freeze their real retained bytes, do not rewrite their plans,
manufacture reference evidence or call this screen Replay-Ready. New attempt
`v1` and actual physical version remain separately bound. Prospective records,
observed roles/costs, canonical traces, independent artifacts and honest terminal
decision are required; any unsupported canonical closure stays BLOCKED with
its exact reason. Scientific claims must not be enlarged to satisfy plumbing.

After exact submission, rebind only the existing Luna B/30-minute heartbeat.
QUEUED/RUNNING is silent; actionable failure or completion emits one durable
event to the existing A. No new monitor/chat or automatic successor.

