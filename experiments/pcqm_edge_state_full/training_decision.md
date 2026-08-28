# Strict Official PCQM4Mv2 Training Decision

Decision date: 2026-08-26

## Result

The random-initialized EdgeState Structural GPS completed ten epochs over all
`3,378,606` official training rows and selected checkpoints only with the
`73,545` official validation rows. The selected epoch was epoch 9 with Gap MAE
`0.189450 eV`. Total training time was `21,611.4 s` (`6.00 h`), within the
declared ten-hour projection gate.

The training loss continued to decrease, and validation improved again at the
final epoch, but the achieved error was not competitive with the project's
existing PCQM specialist evidence. The standalone model was therefore not
promoted and did not alter the production registry. A frozen-model test
inference job was authorized only to produce protocol-complete OGB submission
files and measure the official four-hour inference constraint.

## Evidence

- Exact metrics and epoch log: `results/training_metrics.json`
- Remote completion contract: `results/training_completion_manifest.json`
- Aligned official-valid predictions: `results/valid_predictions.pt`
- Accepted graph identity: `results/graph_acceptance.json`

No external data, pretrained weights, official test labels, or production
registry changes were used.

Post-run residual attribution identified an information-losing atom/bond
feature contract and the Gap-only supervision transfer as the dominant causes.
The evidence and repair disposition are recorded in `root_cause.md`.
