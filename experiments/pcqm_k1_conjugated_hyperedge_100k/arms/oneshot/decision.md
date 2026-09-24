# One-shot conjugated hyperedge — terminal decision (2026-09-24)

The fixed Kaggle2 PCQM-100K/50K seed-42 screen completed 40 epochs with
3,679,617 parameters. No-inference saved-artifact acceptance passed, including
the frozen source/archive/sidecar identities, 50,000 aligned development
predictions, runtime contract, checkpoint and every completion-manifest hash.
Official validation, test-dev and test-challenge were untouched.

The selected development Gap MAE was **0.1415904909 eV** (epoch 37), versus
immutable K1-v4 at **0.1413736343 eV**. The candidate was worse by
0.0002168566 eV. The paired 95% interval for candidate-minus-reference
absolute error was [-0.0006307792, 0.0010891963] eV and crossed zero.
The prospectively frozen 0.003 eV material gate failed. The result is
`NEGATIVE_UNDER_CONTRACT`; this exact one-shot arm received no seed, 500K,
full-scale, or protected-role release.

Saved-prediction subgroup analysis in
[`../../results/saved_error_attribution.json`](../../results/saved_error_attribution.json)
is exploratory on the same repeatedly used development role, not independent
confirmation. Small gains within conjugated-component rows did not offset a
0.003755 eV regression on 4,904 no-component rows.

The authoritative method is [`../../protocol.md`](../../protocol.md); the
accepted joint result is retained in the platform record and each arm has a
separate RML terminal trajectory. No inference or additional training was
performed during acceptance.
