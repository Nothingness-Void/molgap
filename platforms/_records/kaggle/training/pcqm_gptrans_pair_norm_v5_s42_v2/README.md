# Kaggle3 GPTrans pair-normalization execution

- Kernel: `nvoid912/molgap-gptrans-pair-norm-v5-s42`
- Accepted version: 2
- Terminal state: `COMPLETE`
- Accelerator: two isolated Tesla T4 workers
- Source commit: `5f8d27e52d73dca782fad19cde63896e590e8245`
- Source archive SHA256:
  `8b034350a7188cccc3c9b112625d874af910bc8034ea6591f1607f8e85d47ccf`

Version 1 reached the two-T4 allocation check but failed before training
because the accepted graph cache referenced the absent
`molgap.pcqm_wedge.WedgeData` pickle class. Version 2 restored that compatibility
module without changing data, initialization, model variants, or the training
contract. Both version-2 workers completed all 60 epochs.

Large checkpoints and prediction tensors remain local and retrievable from the
kernel. Tracked JSON files preserve the runtime, trace, completion, and artifact
hash evidence. Scientific acceptance is owned by
`experiments/pcqm_gptrans_pair_norm_100k/acceptance.json`.
