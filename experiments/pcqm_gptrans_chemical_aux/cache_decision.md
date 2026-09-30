# Train-label cache disposition, 2026-09-30

Disposition: **NO_TRAIN — input preparation blocked**. No Kaggle dataset or
kernel was published and no GPU was allocated in this submission attempt.
This is not evidence against chemical supervision's Gap accuracy.

The accepted V4 archive, fixed graph bytes, official train membership, CSV
index order and exact 100K train-only SMILES export were verified. The
unchanged pinned chemical encoder rejects source index **51128**, SMILES
`O[Si]123O[Si]3(O1)(O2)O`, as `invalid_smiles`: RDKit reports a silicon atom
with valence five. See `cache_observation/confirmed_blocker.json` and
`train_export_binding.json` for encoder and input identities.

One failed row is sufficient to falsify complete-cache acceptance under
`cache_protocol.md`. The full descriptor build was stopped after independently
rechecking this row through the same encoder. No accepted cache or labels.npz
was produced; the failure inventory is incomplete. Do not infer that only one
row fails, or claim the remaining labels are valid.

The first export attempt was blocked before label processing by multiline JSON
in a JSONL file. Commit `8ad7dbe9` corrected the narrow export adapter and four
focused regression checks passed. The second attempt reached real label
construction and exposed the molecular parsing blocker above. Neither attempt
is a trained arm or a scientific comparison.

`cache_observation/process_before_cancel.json` retains the measured Windows
process CPU/wall interval before cancellation. It is a lower bound, excluding
the cancellation tail, the first failed export, the separate confirmation
probe and orchestration. Aggregate task cost is unknown. Accelerator and
queue use are not applicable. Only the frozen train SMILES/chemical labels
were consumed; no development selection or protected evaluation occurred.

The cheapest unresolved discriminator is an explicit, reproducible policy for
non-RDKit-parseable official training SMILES, followed by CPU coverage checking.
Dropping rows, imputing labels, changing the descriptor encoder or masking
auxiliary supervision would change this contract and require a separately
recorded prospective decision before execution. GPU training remains blocked.

The chemical research question remains untested. Retain its experiment branch
for resolving input policy; this diagnostic does not adopt a model or archive
the whole scientific route. The original fixture feasibility record remains
a separate, limited engineering observation.
