# GPTrans CPU parity diagnostic — technical result

Decision date: 2026-09-29. Kaggle2 `kaseichou/molgap-gptrans-cpu-parity-v1`,
version 1, executed as a private CPU-only kernel with no dataset mount, graph
cache, checkpoint, training, or evaluation role. The script SHA-256 was
`85daf621261e385e77f48d08021d58f4ff7281bbbe7e7d71241acd6a4f293983`.
Kaggle reported `COMPLETE`; the retrieved `summary.json` SHA-256 was
`e818a5e629d4ef831e3fb5421904c703ba8a6f84de546b8676354b260d770a35`.
The separate local output verifier checked that script identity, all pinned
upstream/local source hashes, role declarations, runtime mode, and all ten
per-test records agreed. It also compared the Kaggle-pulled latest source to
the submitted script after newline normalization; all 490 lines matched.
Pulled metadata reported kernel ID number `136402028`, GPU disabled and no
dataset sources. Verification passed. Raw output remains in the ignored local
`platforms/_records/kaggle/gptrans_cpu_parity_v1/v1/` tree and at the
[Kaggle kernel](https://www.kaggle.com/code/kaseichou/molgap-gptrans-cpu-parity-v1).

Observed on tiny synthetic graphs and pinned source, with no MAE inference:

- With identical weights and inputs, the author and MolGap node/pair
  propagation operators matched exactly in the tested forward pass
  (`node_max_abs_diff=0`, `pair_max_abs_diff=0`). This does not validate the
  whole 12-layer model.
- On a connected three-node chain, author Cython and local shortest-path
  distances matched. The author's multihop edge input changed when a bond
  along the two-hop path changed; the local initial pair key did not. The
  author's compiled path routine also returned a zero edge-input code for a
  route whose intermediate atom index was zero. That boundary behavior is
  visible in the verified output and must not be copied uncritically.
- For disconnected pairs, the author Cython distance code was 510 and the
  tested local cap-20 code was 21. The author collator shifted a distance-two
  code to 3. Its `GraphEdgeFeature` reset incoming attention bias: replacing
  that bias with all `-inf` changed the tested output by exactly zero. The
  collator's distance mask is therefore not evidence of an effective mask at
  that point in the author model.
- Source and analytic initialization checks found an expected pre-LayerNorm
  degree-embedding variance fraction of 0.968 locally versus 0.182 for the
  author configuration, under the stated independent zero-mean assumptions.
  These are initialization expectations, not measured prediction quality.
- In the tested Linear+LayerNorm toy network, the author optimizer exempted
  bias and one-dimensional parameters from weight decay; the frozen MolGap
  full runner applied 0.05 decay to every parameter. The author training code
  fed raw labels to L1 loss; MolGap standardized Gap using train-set mean and
  sample standard deviation before L1 loss. The label named `mse` in the
  author configuration dispatched to L1 in the pinned `main.py`.

This diagnostic established concrete non-architecture parity gaps; it did not
show which gap explains MolGap's full-scale MAE or whether changing any one
would help. No new training, database substitution, protected-role access, or
architecture promotion was released by this result.
