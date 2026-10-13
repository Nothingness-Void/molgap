# Attempt v1 disposition - 2026-10-13 JST

Kaggle1 kernel138365645/version1 ended ERROR. Both mean2 and single_ema
failed during preflight, before formal training. Each has observed zero epochs,
optimizer steps and sample presentations. This attempt closes as NO_TRAIN;
the scientific500K comparison is INCONCLUSIVE, not a negative model result.

Both worker logs report `Missing artifact:
pcqm4mv2-ogb-fixed-500k-scnet-v1/manifest.json`. The shared bootstrap passes
`/kaggle/input` to a loader requiring that exact relative path. Remote metadata
contains the intended graph dataset, but the retained output has no mount
inventory: the actual graph location or absence is unknown. This establishes
an input-resolution failure, not a corrupted graph, EMA defect or dependency
conflict as the cause. See [acceptance](terminal_acceptance/acceptance.md) and
[attribution](terminal_acceptance/attribution.md).

The retained entry observation measures219.285577661 wall seconds and
438.571155322 assigned T4 device seconds (0.121825321 device hours). Queue,
allocation before entry and release/billing tails are unknown. No selected
model, resume checkpoint, predictions, training trace or qualified runtime
certificate exists for either arm. Neither arm qualifies for strict replay.

The frozen prospective records and source are preserved. Failure closure
does not resolve the research question or authorize a retry, continuation,
training, adoption or official-role access. The owner branch retains custody
pending a separate user-directed next action and terminal Git routing.
