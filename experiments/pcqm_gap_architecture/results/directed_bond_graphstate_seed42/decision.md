# Directed-bond GraphState seed-42 decision

## Result

Kaggle1 version 1 completed both 40-epoch T4 workers and passed the frozen
no-model acceptance. The zero-start non-backtracking candidate reached
`0.1299586743 eV`, versus `0.1303543001 eV` for its fresh GraphState9 control:
a paired reduction of `0.0003956258 eV` (0.3035%). It was better in eight of
the final ten epoch-aligned validation measurements, so the sign is not caused
only by selecting one isolated checkpoint.

The added directed edge path used 3,741,265 parameters versus 3,665,809
(+75,456; +2.06%). Throughput was 454.06 versus 504.88 graphs/s, a ratio of
0.8993. Both best checkpoints occurred at epoch 39. All shared initial tensors
matched, the added return was zero initialized and received finite nonzero
gradients, and official validation/test-dev remained unread.

## Interpretation

Directional predecessor-to-outgoing-bond flow contains some useful signal
beyond the symmetric wedge feedback. However, its paired gain is below the
project's `0.001 eV` material-gain rule and is accompanied by a meaningful
runtime penalty. The late-epoch deltas also fluctuate in magnitude, which is
not enough evidence to spend two confirmation seeds.

## Disposition

The mechanism is retained as weak positive architecture evidence, not as a
replacement for the three-seed GraphState9 handoff. Seeds 43/44, full-data
training, official-role evaluation, production changes and molecular-research
server use were not authorized. The separately isolated SignNet-LapPE screen
is unaffected.
