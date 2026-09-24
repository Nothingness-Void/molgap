# Persistent conjugated hyperedge — terminal decision (2026-09-24)

The fixed Kaggle2 PCQM-100K/50K seed-42 screen completed 40 epochs with
3,679,617 parameters. No-inference saved-artifact acceptance passed, including
the frozen source/archive/sidecar identities, 50,000 aligned development
predictions, runtime contract, checkpoint and every completion-manifest hash.
Official validation, test-dev and test-challenge were untouched.

The selected development Gap MAE was **0.1401050985 eV** (epoch 39), versus
immutable K1-v4 at **0.1413736343 eV**: a gain of **0.0012685359 eV**.
The paired 95% interval for candidate-minus-reference absolute error was
[-0.0021709513, -0.0003703046] eV, favorable but still below the
prospectively frozen **0.003 eV** material gate. The result is
`POSITIVE_BELOW_GATE`, not a promoted architecture. No seed, 500K, full-scale,
or protected-role release followed.

The saved-prediction attribution in
[`../../results/saved_error_attribution.json`](../../results/saved_error_attribution.json)
is exploratory on the same development role. It showed +0.004920 eV on 4,904
rows without a conjugated component, +0.001745 eV on 37,533 rows with one,
but -0.003464 eV on 7,563 rows with multiple components. Because the new
branch has no active component on the first group, its improvement there
cannot be credited to direct hyperedge message passing; altered shared-weight
optimization is a possible explanation. Regression in the multi-component
group weakens the mechanism-specific scale hypothesis. The previously observed
PairToken 100K-to-500K reversal is an independent caution, not proof of this
arm's 500K behavior.

The authoritative method is [`../../protocol.md`](../../protocol.md); the
accepted joint result is retained in the platform record and each arm has a
separate RML terminal trajectory. No inference or additional training was
performed during acceptance.
