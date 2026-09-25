# Local paired 100K scale-transfer control

This is a diagnostic, not a new model-promotion screen. The question is whether
the historical 100K joint Noisy Nodes + Pair Update Norm advantage persists in
a same-machine, same-source, same-role control and whether its frozen weights
transfer to the already-used 500K development cohort. The accepted historical
reference and joint candidate remain unchanged and are not replaced.

## Frozen inputs and arms

- One official-train-derived `ogb-train-100k` cache, manifest SHA256
  `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
  Train source indices `0:100000`; internal development `100000:150000`.
  `validate_fixed_assets(..., verify_content=True)` checks all three shards.
- Frozen seed-42 initial state SHA256
  `9205fc0f0f97f1cc1cea84ab4bd24206274a00c7d84d26366497feee1710c20c`.
- Reference: unmodified GPTrans-T 12x256/pair32. Candidate: the existing
  `make_noisy_nodes_pair_norm_model` with noise standard deviation `0.15` and
  denoising-loss weight `0.1`. The reference core initialization is identical;
  the candidate's auxiliary head is newly initialized under seed 42.
- Both use the V4 60-epoch, physical-batch-128, FP32/no-TF32 deterministic
  optimizer and schedule: 781 steps/epoch, 46,860 total steps, 5,998,080
  presentations. AdamW LR `1e-3`, WD `0.05`, four-epoch warmup, cosine decay to
  `1e-6`, clipping `1.0`, EMA `0.9999`, normalized Gap L1. Candidate alone adds
  the specified denoising loss. EMA development MAE selects the checkpoint.
- Both run sequentially on the local RTX 5060 from one SHA-pinned source bundle;
  no reference substitution, learning-rate retune, seed change, or test access.
  Windows graph loading uses zero worker processes without changing batches.

## Decision and cost

The predeclared primary result is `MAE(reference) - MAE(candidate)` on the
same 50K internal-development rows at each arm's V4-selected EMA checkpoint.
Report per-row paired error changes and a row-bootstrap interval as a
single-seed uncertainty description, not a training-seed confidence interval.
Compare early and late same-epoch curves; do not declare causality from online
training loss alone. The historical `0.003 eV` materiality floor is a
descriptive reference here, not an automatic scale-up gate.

After both checkpoints are frozen, a separate train-free check may predict
the accepted 500K internal-development rows `500000:550000` with both weights.
That cohort has already been used in prior model selection, so this is a
diagnostic transfer check only. It cannot establish an independent promotion
claim. No official validation, test-dev, or challenge role may be read.

Planning ceiling: `2.5` local RTX 5060 wall hours per arm, `5.0` hours total.
These are estimates from the short preflight, not measured actual costs. If an
arm cannot complete within its ceiling, preserve its atomic last checkpoint,
record `STOP_FOR_COST`, and do not claim a terminal paired endpoint. Native
wall and GPU allocation time must be reported separately where measured;
unknown device utilization is not inferred from wall time.

The existing `pcqm_gptrans_v4` and `noisy_nodes` APIs own training. The local
entry point only selects the arm, sets Windows loader workers to zero, and
passes pinned paths. Keep large checkpoints outside Git; hash and record them
at terminal acceptance. RML trajectories and role/cost records remain beside
this experiment.
