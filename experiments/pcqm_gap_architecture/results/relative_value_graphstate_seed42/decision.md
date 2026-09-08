# Sparse relative-value GraphState seed-42 decision

## Decision

The seed-42 candidate was mechanically accepted and scientifically rejected on
2026-09-08. The exact relation-value mechanism is closed without seed, width,
path-limit, optimizer, or schedule retries. It does not authorize confirmation,
full-data training, official-role access, or a successor.

## Evidence

Under the frozen paired PCQM-100K contract, the fresh GraphState9 control reached
`0.12969738245010376 eV` validation Gap MAE and the sparse relative-value
candidate reached `0.1310027837753296 eV`. Candidate minus control was
`+0.0013053970178589225 eV`; its paired bootstrap 95% interval was
`[-0.0002641468359797727, 0.0029769058513920755] eV`.

The candidate used `3,734,977` parameters versus `3,665,809` for the control.
Mean throughput fell from `575.0127991178333` to `509.4930911688668` graphs/s,
and the candidate/control epoch-time ratio was `1.1284804381634`. Both models
completed 40 epochs and selected epoch 39. Memory remained within the declared
reserve, so resource failure does not explain the accuracy result.

The no-model acceptance is retained at
`platforms/_records/kaggle/training/pcqm_gap100k_relative_value_graphstate_seed42_v1/acceptance.json`.
It verified source/cache identity, per-row metric arithmetic, both T4 workers,
checkpoints, payload hashes, and sealed-role flags.

## Attribution

The earlier exact-shortest-path result was a weak, residual-subgroup signal.
Replacing uniform aggregation with learned query/key/value relation mixing did
not turn it into a stable global gain. The added value path increased capacity
and compute but also gave the encoder another route to amplify noisy or
weakly supported 2/3-hop relations. The final paired regression, confidence
interval spanning zero, and throughput loss jointly reject this mechanism;
they do not reject GraphState9 or all relational encodings.

No official PCQM validation/test-dev role or molecular-research server was
accessed.
