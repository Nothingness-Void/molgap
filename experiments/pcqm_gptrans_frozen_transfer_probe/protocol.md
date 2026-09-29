# GPTrans frozen 100K-to-500K development probe

## Question

Would a frozen-weight prediction check on the 500K internal-development cohort
have warned that the 100K GPTrans `pair_update_norm` gain did not meet the
material 500K retraining gate?

## Frozen inputs and scope

- Candidate: accepted 100K seed-42 `pair_update_norm` EMA best checkpoint,
  SHA256 `e70c922d5ddd347551b1d378fc37e2105130a3f6d2bf4153344e8698dbfd82fa`.
- Reference: accepted 100K seed-42 GPTrans-T reference EMA best checkpoint from
  the centered-logits paired job, SHA256
  `a06f1f163dc6fd7ebae45eec6bfcff75cf05da4ae077344a93926b7d498d0565`.
  This is a different accepted run from the historical frozen V4 reference used
  for the original `pair_update_norm` 100K decision. It has the same 100K graph
  manifest, architecture, initial state, seed, target normalization, and
  60-epoch exposure, but cross-job execution drift remains a limitation.
- Input: fixed OGB PCQM4Mv2 500K internal-development graph shard,
  `source_idx` 500000:550000, SHA256
  `1a37dc5d458396b590044d978526822e6d1cb35df223de913a837dc7277d64c1`.
- No training, optimizer step, official validation, test-dev, or challenge read.
- Native GPU inference cost was not measured in advance. The two local passes
  were expected to fit the desktop GPU; no training budget was released.

Run both frozen checkpoints on every available row in this shard. Compare
source-aligned absolute errors and a row-bootstrap interval for the reference
minus candidate gain. The existing 0.003 eV materiality floor is a descriptive
benchmark here, not a validated frozen-transfer cutoff.

If the frozen 500K-cohort gain is below 0.003 eV, this one known negative
500K retraining case would have received a risk flag. If it remains above
0.003 eV, the proposed transfer check would have missed this case. Neither
outcome estimates a false-stop rate, training-seed variance, or the effect of
retraining on 500K.
