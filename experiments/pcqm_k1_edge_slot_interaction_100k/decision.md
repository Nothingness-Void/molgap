# Release and terminal decision — 2026-09-15

Round1 closed edge-normalization placement as a standalone promotion but found
a small favorable effect only when historical real-bond state was normalized
before each update. This operates in local relation recurrence. The no-slot
attention and uniform-return signals operate in K1's sparse global exchange.
Their different paths justify one pairwise additivity test under the remaining
three-round authority.

Running the two pairs separately avoids interpreting a triple stack without
knowing which global simplification interacts with edge recurrence. The frozen
[protocol](protocol.md) sets an explicit additivity condition before any triple
composition. No outcome is claimed before terminal acceptance.

## Terminal result

Both candidates completed 40 epochs and passed the corrected saved-artifact
acceptance without model inference. The first acceptance attempt falsely
compared the no-slot model's mode-specific shared-state hash against the full
K1 state hash. The validator now uses the already accepted no-slot hash for
that arm and the full K1 hash for the uniform-return arm; remote preflight had
already verified shared tensors before training. Source, checkpoint, RNG,
recovery chunks, payload order, finite tensors and artifact hashes also passed.

The edge-context plus no-slot arm reached `0.1416543126 eV`, a regression of
`0.0002806783 eV` against frozen K1. The edge-context plus uniform-return arm
reached `0.1395999342 eV`, a favorable `0.0017737001 eV` change with paired
bootstrap interval `[-0.00269830, -0.00088432] eV`, but remained below the
`0.003 eV` material gate.

The favorable pair did not preserve both parent effects. Its gain was slightly
smaller than uniform return alone (`0.0018262863 eV`) and was
`0.0009428859 eV` below the isolated-gain sum (`0.0027165860 eV`), outside the
protocol's `0.0005 eV` additivity tolerance. The no-slot pair also failed both
additivity conditions. Triple composition is therefore prohibited and the
edge/slot stacking question is closed. Exact evidence is in
[acceptance](results/acceptance.json) and [summary](results/summary.json).
