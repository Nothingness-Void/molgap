# GPTrans after EMA: evidence-ranked research questions

Review date: 2026-10-01. Scope: repository evidence, primary methods and author
source only. This review did not execute training, inference, role access,
remote probes or submission. Proposals below are not release protocols.

## Finding: repair the interpretation before adding capacity

The [accepted follow-up](../pcqm_gptrans_input_ema_100k/gpu/results/decision.md)
established an EMA intervention gain, not stronger live representations.
All 60 live train/dev metrics, learning rates, optimizer steps and presentations
were identical between G1 and G1 EMA999; selected MAE improved by 0.006652 eV.
The same-core path-mean addition did not establish an additive gain.
Exact results and uncertainty remain in that experiment's records.

For constant decay beta, the EMA half-life is log(0.5)/log(beta) updates.
At beta=0.9999 it is approximately 6,931 updates; at beta=0.999, approximately
693. A larger dataset has more updates per epoch, so an unchanged decay has
a shorter epoch-equivalent horizon. The 100K gain therefore cannot establish
a 500K or full-role gain. Initial-average retention and lag were not separated
by this experiment. No historical selection policy should be rewritten.

## Q1: parameter-group regularization without an architecture change

**Priority: first qualified training question, not a demonstrated gain.**

The [accepted parity diagnostic](../pcqm_gptrans_parity_cpu/decision.md)
identified an exact recipe difference. The server runner's
`pcqm_gptrans_v4._make_training_state` passes all model parameters to AdamW
with weight decay 0.05. The [author optimizer](https://github.com/czczup/GPTrans/blob/main/optimizer.py)
exempts one-dimensional parameters and biases. Local normalization weights,
biases and per-channel LayerScale parameters are therefore decayed too.
This is verified implementation behavior, not evidence that their learned
values collapse or that removing decay will improve MAE.

The bounded question would compare G1 EMA999 with one candidate exempting
only one-dimensional parameters and biases. Matrix and embedding decay,
LR, clipping, target transform, exposures, data, architecture, BS128 and FP32
would remain fixed. Embeddings must not silently be exempted as well.
No layerwise-LR policy or raw-target loss should be added in the same arm.

Use the accepted EMA999 arm as the immutable reference if prospective
reference reuse and runtime qualification pass; do not retrain a control by
default. Declare `optimizer_comparison`, exact parameter names/shapes/groups
and their digest. Record live/EMA curves, parameter-group weight/update norms,
pre-clip gradient norm and clip frequency. The diagnostics must have a frozen
sampling/cost policy. A gain confined to EMA selection would not establish
better live optimization; a train-only fit gain would not establish generalization.

This adds no inference parameters and requires no new graph cache. Wall time
has not been profiled for this proposed grouping; prior runtimes are only
context, not a promised completion time. No compute was released.

## Q2: distinguish bond arrangements, not another uniform path mean

**Priority: CPU feasibility before any training; architecture hypothesis.**

The [author input code](https://github.com/czczup/GPTrans/blob/main/models/gptrans.py)
applies hop-specific transformations before pooling path edges. The tested
server G2 instead pools categorical histograms through existing bond tables.
It is order-invariant: equal distance and identical category counts produce
the same contribution. For example, single/single/double and
single/double/single cannot be distinguished by that mean, even though their
endpoint-relative arrangements differ. This proves an encoding limitation,
not that it caused the negative MAE.

The accepted sidecar stores ordered path CSR categories, while the model
adapter consumes counts. Thus the missing sequence is not automatically a
missing database. A possible intervention is a compact hop-conditioned pair
initializer, retaining the existing propagation core and direct-Gap head.
It is neither another SPD-only selector nor the closed path-mean experiment.

Before release, establish:

- Covariance under atom relabeling and the intended ordered-pair behavior
  under reversing endpoints. Directionality itself is not an error; dependence
  on incidental atom IDs is.
- Qualification of equal-length shortest-path ties. Deterministic BFS alone
  does not prove graph-isomorphism invariance. An invariant path-set encoder,
  or a declared unique-shortest-path-only scope, needs its own accepted policy.
- Information separation on synthetic path permutations, exact sidecar
  bindings, padding/unreachable behavior, actual coverage and cache needs.
- Bound on optimizer-inclusive cost and memory. Neither a large path tensor
  nor extra transformer layers should be hidden in a supposedly small addon.

The public [GPTrans paper](https://arxiv.org/html/2305.11424v3) supports the
node/pair propagation mechanism through PCQM ablations, but does not isolate
our proposed order encoding. Its full recipe and efficiency measurements are
not evidence of T4/FP32 small-data portability. This question is not GPU-ready.

## Reserve and rejected shortcuts

[GRPE](https://arxiv.org/html/2201.12787) uses relation-aware attention and
graph-encoded values, with a component ablation on ZINC. However, GPTrans
already has pair-to-node propagation: it is incorrect to claim it has no
edge-value information. Transplanting another full-depth value stream would
need evidence of a distinct bottleneck, not just GRPE's published result.
The [K1 sparse-triplet](../pcqm_k1_sparse_triplet_100k/decision.md) and
[GPS++ local adapters](../pcqm_k1_gpspp_local_100k/decision.md) remain closed.

Clean-main-input local/context supervision remains a conditional reserve,
not a replacement name for failed histogram, masked-atom or motif recipes.
The [joint reconstruction decision](../pcqm_k1_joint_atom_reconstruction_100k/decision.md)
owns the corruption-versus-clean-input comparison. Train-only source-task,
redundancy and shortcut checks would be prerequisites. No pretraining release
or new supervision targets were created by this review.

## Decision boundary

Q1 has the clearest verified implementation gap and lowest inference cost.
Q2 is the more genuinely architectural question, but feasibility can reject it
before consuming accelerator time. Both should compare against a qualified,
EMA-corrected reference, never the weaker old EMA scalar.

Future material gates must be frozen for their purpose; this review does not
reuse 0.003 eV as a universal V5 threshold or edit any existing gate. A saved
cross-cohort endpoint can test portability without further training only when
the exact evaluation role has separately been authorized and qualified.
All eventual runs require prospective/trace/evidence/cost/role/terminal records
and verified candidate/reference Replay admission. No seeds, scale-up,
desktop job adoption or automatic successor was authorized here.
