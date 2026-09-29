# GPTrans-T 100K V5 audit reference — 2026-09-30

The user-authorized, observation-only seed-42 rerun completed six accepted
10-epoch Kaggle2 T4 segments. The final 60-epoch result passed both the V5
streaming artifact/trajectory acceptance and the existing prediction-content
acceptance. It produced 60 live-train, live-development, and EMA-development
observations, 46,860 optimizer steps, and 5,998,080 sample presentations on
the immutable 100K/50K fixed dataset. Source archive, graph manifest, initial
state, FP32/BS128 runtime certificate, model, predictions, checkpoint, and
protected-role checks passed. Compact hashes and measured cost are in
[`results/final_acceptance.json`](results/final_acceptance.json).

The selected EMA model's internal-development Gap MAE was `0.1560144881 eV`
at zero-based epoch 59. The contemporaneous live-model development MAE was
`0.1491744071 eV`; the two metrics use different weights and are not competing
model-selection claims. The historical P100 V4 endpoint was `0.1566272043 eV`.
Their `0.0006127162 eV` difference across platform/runtime is not an
architecture improvement or an estimate of stochastic variation. The
scientific gain here is the missing live-versus-EMA trajectory, not a new model.

All six segments used one visible T4 within a two-T4 allocation. Native cost
counts both allocated devices: `5.0155572684` T4-device-hours, below the
prospective 20-hour snapshot ceiling. The wrong-slug first submission failed
its source gate before training and remains a separate infrastructure event.

This run does not release G1/G2, more seeds, 500K, protected roles, or a new
architecture comparison. The canonical trace and prediction payload are
accepted, but a strict RML terminal/reference-bundle finalization has not yet
been performed; no replay-ready or strict causal comparison is claimed from
that missing packaging step.
