# GPTrans Feature Denoising 100K Protocol

## Question

The accepted Noisy Nodes plus Pair Update Norm candidate improves the fixed
PCQM-100K V4 reference, but its auxiliary task reconstructs only atomic number.
This experiment asks whether preserving more categorical chemical identity in
the final node/pair states gives a material additional gain without changing
the validated GPTrans propagation topology.

## Frozen arms

- `full_atom`: reconstruct all nine OGB categorical atom features from final
  node states.
- `full_atom_bond`: use the same atom objective and additionally reconstruct all
  three OGB categorical bond features from final directed-edge pair states.

Both arms retain Gaussian node noise (`std=0.15`), Pair Update Norm, direct Gap
prediction, and total auxiliary weight `0.10`. The bond arm averages atom and
bond cross entropy before applying that fixed weight. It does not increase the
overall auxiliary-loss coefficient.

## Comparison

The strict reused comparator is
`pcqm-gptrans-noisy-pair-norm-100k-s42` at `0.14724504947662354 eV`.
Its accepted checkpoint is hash-bound and evaluated once on the same fixed
50K internal development rows so all candidates have row-aligned predictions.
The reference is not retrained.

An arm is materially positive only when:

1. development Gap MAE improves by at least `0.003 eV`;
2. the paired bootstrap interval is favorable under the terminal acceptance;
3. all V4/V5 identity, role, runtime, artifact, and finite-value checks pass.

No official validation, test-dev, or test-challenge role is authorized.

## Execution and stop rule

One Kaggle1 T4x2 kernel runs one isolated arm per T4. Each arm uses its own
prospective trajectory, run identity, output directory, checkpoint, raw trace,
cost event, role events, terminal package, and RML finalization. Both arms run
the fixed 60-epoch V4 endpoint. No successor or 500K action is pre-authorized.

If neither arm clears the material gate, close expanded feature denoising. If
only `full_atom` clears it, reject bond reconstruction. If `full_atom_bond`
materially exceeds both the strict comparator and `full_atom`, retain it as the
single candidate for a separate scale-transfer decision.

## Replay-ready requirement

Terminal acceptance must retain each arm's raw trace and predictions, generate
one canonical trace per trajectory, prune only after an artifact inventory,
write a retention receipt, and run the RML terminal pipeline independently for
both arms. Submission alone is not replay-ready.
