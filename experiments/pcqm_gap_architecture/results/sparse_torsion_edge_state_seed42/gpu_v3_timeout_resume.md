# Sparse Torsion GPU Version 3 Timeout and Resume

Record date: 2026-09-01

## Terminal state

Kaggle1 kernel `nothingnessvoid/molgap-pcqm-sparse-torsion-s42`, version 3,
ended at its planned 23,400-second search budget. The fresh distance-plus-angle
comparator completed all 40 epochs. The sparse torsion candidate completed
epochs 0 through 38 and wrote its atomic checkpoint before the budget guard
stopped the process. The candidate therefore lacks only epoch 39 and final
validation/artifact assembly; version 3 is not itself a complete scientific
comparison.

The partial trace had its best candidate validation value at epoch 36. This is
diagnostic only and does not determine the experiment because final validation,
payload, metrics, and selection artifacts were not produced.

## Resume evidence

The version-3 output was downloaded intact. The resume bundle contains the
complete comparator artifacts and the candidate's best model, epoch-38 atomic
checkpoint, and trace. Its tracked manifest is
`resume_dataset_manifest.json`; the manifest SHA-256 is
`9d0f4ccc5f315dd5c7f5fe9305bb6cd36f1bd88659bffeea96711525678c77f9`.

The candidate checkpoint records model, optimizer, scheduler, train data-order
generator, CPU RNG, CUDA RNG, best epoch/value, stale-epoch count, and the full
39-epoch trace. The runner verifies every resume artifact by path, byte count,
and SHA-256 before atomically hydrating output state. Normal checkpoint logic
then skips the completed comparator and resumes the candidate at epoch 39.

## Contract

The resume changes no data role, graph or geometry cache, architecture,
parameter count, seed, precision, batch, optimizer, learning rate, weight
decay, epoch limit, patience, target, or advancement gate. Official PCQM
validation and test-dev remain unread. The resume exists only to complete the
already-authorized seed-42 comparison and its evidence chain.
