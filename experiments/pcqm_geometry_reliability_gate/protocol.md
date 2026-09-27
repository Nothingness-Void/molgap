# Frozen existing-prediction geometry fusion screen

## Question and authority

Can a single global convex weight on the accepted 2D EdgeState and
distance-plus-angle EdgeState predictions improve over the geometry arm on
the same 50,000 development rows? The accepted source is
`experiments/pcqm_distance_angle_500k/v5_evidence.json`; the previous
component diagnostic is `experiments/pcqm_geometry_component_attribution/`.
This is a new prediction-combination question. The original training protocol
excluded fusion, so its training decision is unchanged.

## Frozen local screen

Read only the two SHA-pinned `direct_gap_development.pt` payloads. Check the
accepted pair, exact hashes, ordered `source_idx` 500000..549999, identical
finite targets and finite predictions, and recomputed component MAEs. Use a
fixed weight grid 0, 0.05, ..., 1 on the geometry prediction. Report fixed
weights 0.5, 0.75 and 1.0. For five folds defined by `source_idx % 5`, choose
the lowest-MAE grid weight on the other four folds; break ties toward weight 1.
Evaluate each held-out fold once and concatenate its predictions. Report the
pooled MAE, fold gains against the geometry arm, and weight stability.

Describe [2,4) and [8,infinity) eV target-Gap tails only after the frozen
OOF predictions exist. True target Gap never selects a weight or routes an
inference row. The models were already selected on this development role, so
OOF weight fitting is exploratory nomination evidence, not independent
validation, causal geometry attribution, V5 replay, or promotion evidence.

## Decision boundary

The screen supports a follow-up geometry-reliability model question only if
pooled OOF MAE improves by at least 0.001 eV over the accepted geometry arm,
at least four of five folds improve, and neither reported tail has higher OOF
MAE than the geometry arm. Otherwise close `NO_TRAIN` without accelerator
release. Passing only authorizes a separate frozen proposal, strict V4/V5
reference and role qualification, preflight, and cost review. It does not
authorize a Kaggle push by itself. No baseline retraining fills a missing
reference field.

The only permitted action here is one hash-gated local CPU screen on already
consumed internal-development predictions. Official validation, test-dev,
challenge-test, graph caches, checkpoints, model inference, and remote
schedulers remain untouched. Record actual local wall time, unknown CPU time,
and non-applicable accelerator and queue time.
