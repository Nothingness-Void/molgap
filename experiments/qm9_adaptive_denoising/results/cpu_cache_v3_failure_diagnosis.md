# CPU cache preparation v3 failure diagnosis

Kaggle2 kernel `kaseichou/molgap-qm9-adaptive-denoising-cache-prep`, version 3,
completed all 17 train/validation shards and then failed the frozen cache gate.
The terminal evidence is retained under ignored platform storage and bound by
the adjacent idempotent handoff marker.

## Mechanical result

- graphs: 33,000 (30,000 train; 3,000 validation)
- valid ETKDG geometries: 31,765
- failed geometries: 1,235
- valid fraction: 0.9625757575757575
- terminal requirement: at least 0.99
- terminal log SHA-256:
  `1c66d04a82fa9bb3fe9909db30d5cbc3bba1f1fd616a55c7f7130ab0e38a960d`
- no final manifest, aggregate SHA, acceptance, model run, or GPU submission

All failures reached the existing `etkdgv3_random_coords` fallback and returned
RDKit embedding status `-1`. Read-only characterization of the selected
train/validation failure IDs found that 100% contain rings, 99.9% contain at
least two rings, and 99.4% retain stereochemical atom tags. The failures are
therefore structurally concentrated rather than missing at random.

## Disposition

Lowering the 99% gate would systematically mask geometry for the most rigid
and stereochemically constrained molecules, weakening the experiment. The gate
remains unchanged. Before any retry, the cache method must explicitly freeze a
deterministic ETKDG-only difficult-ring fallback and retain the same molecules,
roles, graph identities, scalar distance/angle inputs, and failure masks.
