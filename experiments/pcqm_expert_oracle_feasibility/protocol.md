# Existing-expert Oracle feasibility contract, 2026-09-30

## Question

Can inference-available prediction disagreement identify useful specialization
beyond the accepted K1/GPTrans global blend? This differs from the closed coarse
structure router: no graph summary, new encoder, checkpoint inference, or remote
job is used. Prior decisions remain unchanged. Pretraining results remain pending
and are not substituted with historical scores.

## Permitted inputs and computation

Read SHA-pinned accepted V4 500K/50K predictions for original GPTrans-T, K1,
EdgeState, GPTrans distance and K1 distance+angle. All must contain exactly
ordered source_idx 500000..549999, identical targets, finite predictions.
Original models were selected on this already-consumed internal-development
role. Geometry endpoints are mechanically accepted; their strict training replay
qualification is unresolved. Row alignment permits a prediction diagnostic,
not a causal geometry claim. No official validation/test role is accessed.

Reuse the owning matched-V4 loader, global L1 pair-weight and row-bootstrap
helpers; `multi2d.targetwise_oracle` and `hierarchical_oracle_analysis` own
Oracle computations. A thin caller adds only the diagnostic comparisons.

Freeze these reports before execution:

- Individual endpoints, fixed original/geometry 50:50 blends.
- Discrete label-informed Oracle for original pair and the five retained arms;
  compare these with their actual fixed-blend comparators. Also include the
  original fixed blend as a fallback action in the five-arm Oracle.
- Pair convex interpolation Oracle and 5/10/20/50/100% conditional-call budgets,
  with K1 as base and original GPTrans as extra expert. Encoder-pass figures are
  conceptual workload counts, not measured latency or native hardware cost.
- Original pair error-sign opposition, winner shares and cancellation benefit.
- Five outer folds source_idx % 5. On four folds fit the existing global L1
  weight. Separately fit a fixed prediction-only hard classifier and a soft
  convex gate. Inputs: K1/GPTrans predictions, signed/absolute disagreement,
  mean prediction. Neither true Gap nor errors are inference features.
- Hard gate: StandardScaler + LogisticRegression(C=1, max_iter=500). Soft gate:
  HistGradientBoostingRegressor(loss=absolute_error, max_iter=100,
  max_leaf_nodes=7, max_depth=3, min_samples_leaf=200, l2_regularization=1,
  learning_rate=0.1, random_state=42), fitted to clipped Oracle alpha with
  absolute prediction-disagreement sample weights. Zero-disagreement rows
  have no gate effect. No hyperparameter search or fold-based choice of method.
- Report pooled held-out MAE, every fold effect, regret versus Oracle,
  winner AUC, and bootstrap CIs for hard/soft gates versus cross-fitted global
  weight. The existing 10,000-row-bootstrap helper is reused unchanged.

The models themselves are not OOF-trained. Cross-fitting isolates only the
gate's row labels; prior development model selection and repeated analyses
remain. These are exploratory feasibility results, never independent promotion.
No scaffold/embedding routing is executed: its inputs are not retained here.

## Decision and stop rule

Oracle headroom alone never releases training. A prediction-only gate is
worth a separate independent-role proposal only if its pooled improvement over
cross-fitted global blend is >=1 meV, paired CI lower bound >0, and >=4/5 folds
improve. The 1 meV criterion is a newly declared diagnostic nomination rule,
not a training-promotion threshold or a variance estimate. Otherwise close this
diagnostic NO_TRAIN and retain pretraining + fixed fusion as the next comparator.
Large Oracle headroom with failed learnability means unidentifiable routing,
not absence of complementary errors or proof against all future specialists.

## Resource and provenance

One local CPU saved-prediction diagnostic; no accelerator allocation, model
training/inference, remote query, resubmission, or monitoring. Estimate <=15
minutes wall time; record measured wall and process CPU separately. Native
training/inference cost of a future MoE is unknown. Hash the source/input
bindings, preserve the prospective record before reading labels, and use the
existing RML planner/finalizer/rebuild/check. No training replay-ready claim.
