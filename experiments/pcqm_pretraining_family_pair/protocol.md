# Two-family local pretraining plan

## Question and interpretation

With a fixed short local-reconstruction stage and a longer target-adaptation
stage, which of the selected K1 and GPTrans recipes is worth further investment?
The user explicitly selected two pretrained families. There is no fresh scratch
control in this two-arm allocation. The difference between the arms measures
whole-recipe performance, not the causal effect of pretraining or architecture.
V5 reference reuse is the default: bind each arm to the accepted RML record
for its own unpretrained variant, then validate the reference bundle, artifact
hashes, data/rows/features/targets, seed, precision, optimizer, schedule,
selection/EMA and runtime qualification. Pretraining is the declared intervention;
matching downstream exposure does not match total compute. Cross-job reuse is
allowed when V5 qualification passes; same-job scratch retraining is not a
requirement. Missing qualification is PENDING, not automatic baseline retraining.
No third/fourth training arm is planned.

## Family selection

- Arm A: frozen K1-v4, 3,658,817 inference parameters. Retain persistent local
  EdgeState and the original single-slot exchanges. Original PairToken won the
  100K screen (0.138330 versus 0.141374), but its accepted 500K comparison was
  worse than its own K1 reference (0.104304 versus 0.103605). It is not the
  scale-supported choice. The later combined simplification was only a
  sub-threshold 100K direction, not a replacement incumbent.
- Arm B: GPTrans-T + Noisy Nodes + Pair Update Norm. Its matched desktop 500K
  result was 0.104812 versus 0.106868 for its core reference; the 2.056 meV
  improvement failed the 3 meV gate. Select it as the numerical candidate
  requested by the user, not as an adopted production model. Its historical
  training parameter count is 5,277,400; distinguish denoising-head parameters
  from inference parameters. Do not compare the two quoted 500K cohorts as
  if they formed one common benchmark.

This selection excludes geometry, pending DSAR/DSMR, and new structural stacks.
It changes neither Track A production nor the accepted full EdgeState reference.

## Frozen proposed exposure

Use the same authorized 100K training membership for both arms, seed42,
physical BS128, FP32/no TF32, full-batch drop_last and deterministic order.
Each arm has 10 local pretraining passes. K1 then uses exactly the historical
40-pass downstream recipe: 7,810 + 31,240 = 39,050 updates, 4,998,400 sample
presentations. GPTrans joint uses exactly its 60-pass downstream recipe:
7,810 + 46,860 = 54,670 updates, 6,997,760 presentations. These are planned,
not measured, counters. The previously proposed extra 20 K1 passes are cancelled.
Additional pretraining exposure is 25% of K1 downstream exposure and 16.7% of
GPTrans downstream exposure; native wall-time ratios remain unmeasured.

Preserve each accepted reference optimizer, scheduler, loss, selection and
EMA semantics verbatim. K1 receives clean normalized Gap L1; GPTrans retains
Noisy Nodes alpha0.1/corruption0.15 plus Pair Update Norm. Reuse their accepted
reference records; do not repeat scratch training.
Both arms start from the same seed's own family initialization, not
from previously Gap-trained checkpoints. Reset optimizer/scheduler at the
pretrain-to-Gap transition; reinitialize the unused scalar Gap head from a
separate frozen seed. Reset the downstream RNG/order stream at that boundary.
No model-wide reinitialization, EMA contamination, or import of pretrained
weights from a different family is allowed.

Pretraining: reuse local atom categorical reconstruction, real-bond categorical
reconstruction and atom-attached functional-group labels. Mask rate0.15;
loss = atom CE + bond CE + 0.5 * group BCE, retaining the original reduction
semantics. No graph-level aggregate descriptor replacement. GPTrans uses its
256-channel non-token nodes and 32-channel pair states gathered ONLY at real
bond endpoints; exclude graph token, padding and virtual/nonbonded pairs.
K1 uses final 192-channel node and 64-channel real-edge states. These dimensions
must be verified by the frozen factories, not guessed by the launcher.
Disable GPTrans's separate Noisy Nodes corruption during pretraining to avoid
double corruption; retain Pair Update Norm. Remove pretraining heads/hooks
before the downstream stage and inference. Existing local reconstruction code
zeroes masked categorical inputs; preserve and document that semantics rather
than claiming a dedicated learned MASK token.

The first ten pretraining passes remain fixed. This plan tests extended Gap
adaptation, not whether longer pretraining itself is useful. No schedule sweep,
early-stop heuristic, second seed or automatic 500K advancement is included.

## Roles and endpoint evidence

Reference bindings found in canonical V5 records:
- K1: pcqm-k1-v4-100k-reference-s42, strict_reference_accepted and
  reusable_same_contract_only, 40 Gap passes. Retained reference bundle and
  aligned predictions must be retrieved/verified before release.
- GPTrans joint: pcqm-gptrans-noisy-pair-norm-100k-s42,
  positive_under_contract at 100K, 60 Gap passes. Its later negative 500K
  promotion outcome does not erase the accepted 100K baseline measurements.

Use their frozen internal-development membership, disclosing prior role use.
The user's instruction to reuse these V5 references replaces the earlier draft's
blanket fresh-audit prerequisite. This is a new two-family pretraining question,
not another search over the old EdgeState 10/30 allocation. No extra fresh audit,
protected role or repeated baseline is added merely for bookkeeping. A later
promotion audit is a separate decision. Do not relabel the reused role as fresh.

Retain development predictions at Gap20/30/40 (plus60 for GPTrans) and each reference-matched
selected checkpoint. K1 uses its reference weight semantics; GPTrans uses its
own reference EMA semantics. Never silently replace EMA with live/raw selection.
Capture additional raw metrics separately for interpretation. Compare early/tail
relative gains only where reference trace exposure and metric semantics match.
Fixed-train-cohort metrics are
needed for fitting attribution; online training loss alone is insufficient.
Row bootstrap does not measure training-seed variability.

## Cost and acceptance

Proposed platform: one T4x2 job, one isolated arm per GPU, independent outputs.
Before release, use the platform skill to verify actual account, availability,
allocation, mounts, source package and durable output. No account/run is assumed.
Engineering budget proposal: 8 T4-hours per arm / 16 allocated T4-hours total,
with separately recorded CPU cache cost. This is a cap, not a runtime forecast.
Use measured warmup/profile plus evaluation/save overhead to decide whether
each arm's frozen exposure fits. If not, STOP_FOR_COST or freeze a new plan before launch; do not
truncate scientific exposure or silently shrink the batch. Durable checkpoint
stages must fit the verified platform limit and retain processed step cursor,
sampler and mask RNG, model/head/optimizer/scheduler, source/config/cache pins,
and immutable remote artifacts.

Mechanical acceptance and scientific interpretation are separate. Candidate
nomination requires >=3 meV same-family gain, positive paired uncertainty and
qualified reference/role identity. Without an equal-budget scratch comparator,
even a positive nomination is not a compute-efficiency win. Two-family ranking
on the reused internal development role can still be reported with its exact one-seed scope.
Any scientific failure gets attribution before another module. Rebuild RML
only after accepted milestones; preserve the original negative decisions.
