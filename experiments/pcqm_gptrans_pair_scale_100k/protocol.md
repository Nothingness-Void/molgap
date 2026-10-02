# Pair-update scale and unresolved decay attribution

Frozen on 2026-10-02. This is one authorized Kaggle3 dual-arm screen, not a
restart of the completed path arm or a rescue of the stopped 500K PreNorm run.

## Evidence funnel

- [EMA attribution](../pcqm_gptrans_input_ema_100k/gpu/results/decision.md):
  changing EMA0.9999 to 0.999 improved selected MAE by 0.006652 eV with identical
  live optimization. The accepted EMA0.999 endpoint is the immutable comparator.
- [Recipe/path closure](../pcqm_gptrans_recipe_paths_100k/gpu/results/decision.md):
  endpoint path contrast worsened MAE by 0.001387 eV. Grouped decay failed in
  inventory serialization, not in optimization; its scientific outcome is unknown.
- [Relation-flow](../pcqm_gptrans_relation_flow/decision.md) and
  [500K stop](../pcqm_gptrans_prenorm_500k_v5/decision.md): full pair channel
  normalization was locally positive but did not preserve its advantage on the
  observed 500K curve. Do not repeat that intervention or claim scale transfer.
- [Memory readback](../pcqm_gptrans_memory_readback/decision.md): more direct
  accumulated pair content improved fitting but not development MAE.

Code inspection, not an activation measurement: the twelve-block pair stream
adds raw, unnormalized updates while the node stream has pre-normalization.
GPTrans's node-to-edge update itself contains raw attention logits plus their
softmax; see [GPTrans equations 6/9](https://arxiv.org/html/2305.11424v3).
[DeepNet](https://arxiv.org/pdf/2203.00555) motivates residual-scale control, but
its derived normalization/initialization is NOT implemented or reproduced here.
The hypothesis that pair accumulation is excessive remains unproved. This
screen tests it, rather than assuming every failed relation module was redundant.

## Independent arms

1. `degree_group_decay_ema999`: recover the unanswered optimizer question with
   repaired shared inventory digest. Biases and 1D tensors have zero decay;
   matrices/embeddings retain 0.05. Start fresh from the accepted degree-scaled
   initial state; the failed best-only checkpoint lacks optimizer/RNG progress.
   Use a distinct recovery trajectory, preserving the failed attempt's cost.
2. `degree_pair_depth_scale_ema999`: change ONLY persistent pair recurrence to
   `P[l+1] = P[l] + delta[l]/sqrt(12)`. Preserve initial chemical/SPD embeddings,
   all tensor keys, parameter count, node update, readout and optimizer. No
   LayerNorm on pair channels, additional memory readback, new paths or geometry.
   This reduces update amplitude without deleting pair mean/magnitude at every
   attention input. The factor is frozen from depth, not swept after results.

Neither arm stacks with the other. Both retain 5,246,817 parameters. Pair-scale
diagnostics record input/update/output RMS per layer on the first already
scheduled train batch each epoch, excluding padded query/key pairs. They cannot
prove baseline activation growth because baseline activation traces are absent.
Optimizer diagnostics use the existing gradient/clip and parameter/moment logs.

## Fixed release and stop conditions

Accepted cross-platform PCQM assets: 100,000 train, 50,000 internal development;
no new cache/split. Seed42, FP32, TF32 disabled, physical BS128/device, 60 epochs,
46,860 optimizer steps, 5,998,080 sample presentations. Same initialization,
target transform, dropout, clipping, lr0.001, warmup/cosine schedule and EMA0.999
as the accepted comparator. No baseline retraining or official/protected roles.

One T4x2 kernel, one isolated independent worker/GPU and atomic checkpoint/chunks
per arm. Estimated 3-4 wall hours / 6-8 allocated device-hours; maximum 6/12,
counting idle allocation. No automatic successor, seed confirmation or scale run.

Use the existing native trainer, release gate, scheduler, saved-output acceptance,
per-arm terminal transaction and replay compiler. Require real reference pointers,
frozen prospective identities, source/initialization hashes and runtime gate
before submission. A complete accepted candidate/reference replay pair is a
terminal requirement, never a prelaunch claim. Failed/partial attempts retain
honest non-replay status, not fabricated observations.

For this bounded screen, material advancement requires gain >=0.003 eV and
paired-bootstrap95% upper candidate-minus-reference bound <0; this is a policy
choice, not a V5 universal constant. Preserve smaller gains separately. Inspect
late live/EMA curves and diagnostics; one seed or a development-set gain cannot
establish 500K/full transfer. Close only the tested hypothesis if negative.
