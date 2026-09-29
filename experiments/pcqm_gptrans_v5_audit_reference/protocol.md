# GPTrans V5 audit reference — prospective decision, 2026-09-30

The user explicitly authorized one 100K GPTrans reference rerun to repair the
historical V4 trace's missing live-development observations. This does not
authorize a changed scientific recipe, G1/G2 candidate, extra seed, 500K
training, or protected-role evaluation.

One seed-42 reference follows the immutable V4 contract in `contract.json`.
Each 10-epoch segment must pass a fresh T4 preflight, match the source, cache,
initial state and runtime certificate, and begin only from a SHA-verified
previous segment. A changed runtime certificate stops continuation; it is not
silently considered comparable. The entire T4 allocation counts in native
cost even if only one GPU is visible to the model. Training stops if the
preflight's estimated training time exceeds six hours or memory reserve is
below 15%; the 20 allocated-device-hour snapshot is an additional planning
guard, not a claim about the user's preferred spend.

An accepted result requires 60 observed epochs, 46,860 optimizer steps,
5,998,080 sample presentations, live online train, live development, and EMA
development MAE at every epoch; same fixed role and source order, finite
predictions, SHA-verifiable selected EMA checkpoint and runtime certificate,
atomic recovery checkpoint and bounded segment outputs, and measured whole
allocation cost. Only then can a V5 reference bundle and future causal
comparison be considered. A queue-complete job alone is not acceptance.
