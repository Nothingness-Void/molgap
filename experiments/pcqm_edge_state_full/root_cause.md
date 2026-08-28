# Strict Official EdgeState Root-Cause Decision

Decision date: 2026-08-26

## Finding

The `0.189450 eV` official-validation result is primarily an input-contract
failure, not evidence that more PCQM training rows hurt the architecture. The
official graph builder reduced each atom to element, explicit degree, formal
charge, and aromaticity, and each bond to bond type. It omitted radical
electrons, total hydrogens, hybridization, chirality, ring membership, bond
stereo, and conjugation.

This loss is not hypothetical. Under the accepted v1 builder,
`[CH2]CCC[CH2]` and closed-shell `CCCCC` produce identical node features,
edges, bond features, and RWSE. The model therefore cannot learn different
predictions for them. Radical-containing molecules are `7,213/73,545`
official-validation rows (`9.81%`) but contribute `28.19%` of total absolute
error: their MAE is `0.544589 eV`, versus `0.150832 eV` for non-radicals.

The second cause is a supervision transfer mismatch. EdgeState passed its
three-seed PubChemQC-100K gate while jointly predicting HOMO, LUMO, and Gap.
The strict official archive exposes only Gap, so the full run was Gap-only.
The earlier matched screen had already measured a `0.011795 eV` mean Gap
regression when the same Structural GPS family lost HOMO/LUMO supervision.

Training horizon is a smaller contributor. Epoch 9 was the best epoch while
the learning rate had already reached `1e-6`; a short warm restart could still
improve the checkpoint, but it cannot resolve input collisions. Global
calibration is excluded: signed bias is only `+0.007275 eV`, and an affine fit
on the same validation rows worsens MAE to `0.196682 eV`.

The tail is highly concentrated. The worst `10%` of rows account for `55.11%`
of total absolute error. Gap below `2 eV` has `2.884251 eV` MAE, and the worst
individual rows are radical chains and rings. The 33 topology-only fallback
rows do not occur in official validation and cannot explain the result.

Exact statistics and the collision contract are in
`results/root_cause_analysis.json`. The accepted training curve is in
`results/training_metrics.json`.

## Repair Decision

The current checkpoint is not repairable by calibration or ordinary
fine-tuning. Changing the missing chemical fields changes the model input and
requires new graph shards and training.

Before another full run, execute one official-train-only matched screen:

1. Keep the current EdgeState GPS9-192 architecture and Gap objective fixed.
2. Compare the v1 continuous features against OGB's complete nine-field atom
   and three-field bond categorical contract using embedding-based atom and
   bond encoders.
3. Use a deterministic train/development subset drawn only from official
   training rows, with radicals and Gap bins represented in both roles.
4. Require at least `0.010 eV` overall development improvement, at least
   `0.10 eV` radical-subset improvement, and no more than `0.002 eV`
   non-radical regression, with the same direction in three seeds.
5. Only after that gate, screen a 16-20 epoch warmup/cosine schedule on the
   same internal split. Read official validation once after architecture and
   schedule freeze.

A full rich-feature run should retain atomic checkpoints and the existing
train/test separation. It must not reuse the v1 checkpoint as architecture
evidence, because the first-layer contract differs.

## Operational Note

Frozen test inference job `1348205.ccpbs1` failed before producing NPZ files
because IMS Torch multiprocessing again lost tensor file descriptors. This is
an infrastructure failure separate from the accuracy result. Its retrieved
log is `results/test_inference_failure.log`. The negative checkpoint does not
justify another A100 inference attempt.
