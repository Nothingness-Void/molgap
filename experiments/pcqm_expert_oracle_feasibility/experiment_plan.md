# Conditional specialist experiment plan

## Evidence-led ordering

1. Await independent terminal acceptance of the desktop pretraining pair in
   `codex/exp/pretraining-family-pair` (K1-v4; GPTrans Noisy Nodes + Pair Update
   Norm; ten reconstruction passes, then their existing 40/60-pass contracts).
   A release/preflight or first-epoch report is not a scientific endpoint.
2. Compare each accepted pretrained arm against its own frozen unpretrained
   reference. Record optimizer steps, presentations, raw/EMA semantics,
   paired MAE/uncertainty, exact role use and actual native cost.
3. Fuse only predictions with matching row/target/role identities. If the two
   pretraining arms' contracts differ, their blend is a prediction-combination
   observation, not a matched causal comparison between architectures. Freeze
   a simple global weight and compare against the corresponding unpretrained
   fusion and each component. Do not assume pretraining preserves diversity.
4. Recompute Oracle and realizable gate headroom on those accepted artifacts
   as a separately planned question. Historical 500K scores are context only;
   they cannot be substituted for missing pretrained predictions or references.

## Candidate A: shared-encoder specialist readouts

Hypothesis: pretrained node representations contain distinct relation signals
which a single readout mixes poorly. Preserve the accepted encoder and its
global context. Add two small expert readouts with distinct learnable atom
attention/projection, and a convex soft gate. Initially freeze the encoder;
provide per-expert regression supervision to avoid one head receiving no useful
gradient. This borrows the diversity mechanism from MoCE; it is a new Gap
regression adaptation, not a reproduction of its multi-task classification.

Cheapest discriminator: qualified retained embeddings on an allowed internal
role, comparing two distinct readouts against equal-parameter ordinary heads
and a fixed mean of the same heads. Scalar prediction files cannot answer this
representation question. Existing accepted loader/factory and inference hooks
must support the exact checkpoint; otherwise record the capability gap before
adding a small owning adapter. No generic CLI model-construction probe stands
in for actual frozen-encoder inference.

## Candidate B: topology-conditioned specialist heads

Hypothesis: an input-available topology embedding separates stable residual
regions better than the already failed 22 coarse graph summaries. Use the same
qualified frozen pretrained encoder, two bounded expert heads and soft cluster
assignment with scaffold-aware training constraints. Retain a global fallback;
do not hard-split the entire data or remove cross-region context. This is the
TopExpert idea adapted to regression. No target Gap enters inference routing.

Cheapest discriminator: train-only clustering on qualified embeddings, then
held-out gain/regret and assignment stability. Compare with the same number
of ordinary heads and the same fixed blend, not just with one weak component.
Missing scaffold/embedding assets remain missing; no CPU graph construction
or checkpoint inference is authorized by this plan alone.

## When a two-arm GPU proposal becomes justified

The present prediction-only gate failed its frozen nomination rule, so neither
candidate is released now. A later independently qualified representation
diagnostic must show a >=1 meV improvement versus its strongest compatible
frozen blend, lower paired CI >0, >=4/5 positive folds, and no predeclared
slice harm. This is a proposal criterion; the future owning contract must
freeze its materiality, statistical and budget gates before training. Do not
reuse repeatedly consumed development labels as independent validation.

If both mechanisms qualify, pair A and B with the same frozen backbone,
two-head parameter budget, head-training exposure, precision, optimizer,
loss, role and selection semantics. Reuse accepted strict references;
baseline retraining is not a third arm. New desktop branch/worktree from the
then-current verified desktop tip; per-arm prospective Spec, release checks,
Kaggle account binding, resumable state, trace, independent acceptance and
two truthful replay-ready RML entries are required under the owning contract.

## Feasibility and costs

Prediction-only Oracle/gate analysis is available and was executed locally.
Retained node embeddings, pretraining terminal predictions, paired matched
reference qualification and independent diagnostic-role authorization are not
established by this study. Existing RML/prediction/Oracle helpers are callable;
an exact two-specialist trainer is not verified and must not be advertised as
ready. Native T4 training hours, inference latency and geometry preprocessing
cost for these proposed variants are unknown, not zero. Head-only training may
save encoder backpropagation, but a numerical saving needs actual measurement.

Late gates consuming both models' predictions cannot skip either encoder.
Conditional compute would require a gate using only cheap input features or
the base model representation before extra experts run. Sparse weights alone
do not prove sparse execution. MI-MoE and multimodal/text specialists are lower
priority until geometry/multimodal inputs show independently useful gain.

## Stop and attribution

Failure of realized held-out gain closes the specific diagnostic NO_TRAIN.
If Oracle is large but the gate fails, attribute the failure to unlearned
winner/weight information under the tested features, not to insufficient encoder
training, a harmful chemical mechanism, or absence of all specialization.
If a future head trial trains, retain train/fixed-cohort development traces,
step/exposure axes, gate gradients/utilization, expert error correlation,
paired calibration/slices and native cost to distinguish redundancy, starvation,
overfit or missing context. Do not claim chemically meaningful specialization
solely from different attention maps.
