# Frozen protocol

Authorized 2026-10-23 JST: one Kaggle1 T4x2 Notebook, fixed 100K, two arms,
40 passes each, maximum 14400 Python-allocation seconds including setup and
180 seconds cleanup. At most 8 allocated T4-device hours; platform quota
debits, queue and measured native cost are different quantities. No automatic
retry, continuation, monitor, protected roles, 500K/full or adoption.

## Evidence review and new question

[K1 audit](../pcqm_k1_complete_local_audit/analysis_zh.md) does not establish
slot saturation or a unique cause of scale regression. Historical
[dropout consistency](../pcqm_k1_dropout_consistency/terminal_acceptance/decision.md)
gained 1.529 meV, below its 3 meV gate, and required two forwards.
[Structural-bias attribution](../pcqm_structural_bias_norm_100k/terminal_acceptance/attribution.md)
shows direct LN substitution hurt; do not repeat BN replacement or infer
overfitting from online train versus clean development errors. Independent RMS
and weight-decay questions remain on their owners. RML and owning-code search
found no K1 graph-wise message DropPath endpoint.

[Stochastic depth](https://arxiv.org/abs/1603.09382) motivates random path
regularization, not a guarantee for this adaptation. Here only the existing
POST_AGGREGATION seam is masked: each local layer and molecule draws an
independent Bernoulli keep mask with probability .95 and scales surviving
aggregates by 1/.95. All nodes/channels of that molecule share its mask.
Root/self messages, outer residuals, BN, FFN, edge updates and slot mixers
remain active. This is not whole-block skipping or edge/node dropout and
claims no compute saving. Evaluation is an exact no-RNG bypass.

Question: can this single-forward, parameter-free structural regularizer
improve held-out selection-role error without changing the K1 backbone?
Alternatives: it only disrupts useful chemistry; BN estimates react adversely;
existing dropout already regularizes enough; apparent gain is seed variation.
The cheapest quality falsifier is one complete same-job paired 100K screen.
Small synthetic tests establish wiring only, not the scientific hypothesis.

## Matched arms and endpoint

`baseline` reuses the exact retained post-BN recipe and transported initialization.
`message_droppath` differs only by the registered probability=.05 addon on
all nine local aggregate branches. Same parameter count, initial tensor keys
and bytes; no added learned parameters or changed loss. No independent tuning.

Both use seed42, physical BS128/drop-last, FP32/no-TF32, AdamW LR4e-4/WD1e-5,
40-pass cosine/min1e-6, no warmup, normalized L1/clip1, EMA.999 retained but
primary best-development-Live, 31240 updates and 3998720 presentations.
Use the existing accelerated CPU layout/aggregation and checkpoint path.
Both retain the same 16-update profiler window. Required native repeat/resume
proofs must pass for both arms before formal training.

Primary gate: complete candidate-minus-reference Gap MAE <= -.003 eV and
paired 2000-resample row-bootstrap 95% upper bound below zero. Row uncertainty
does not measure seed variation. At the resource cap retain STOP_FOR_COST;
partial exposure cannot establish a complete negative or successful endpoint.
Report throughput and allocation separately; neither is a substitute quality gate.

## Roles, custody and implementation

Only the accepted official-train-derived 100K train and next50K development
roles. Development has been reused for selection and is not independent test.
No official-valid, test-dev/challenge, sealed20K, PubChemQC or geometry.
Retain best/last state, LR/steps/samples/checkpoint-bound curve, aligned finite
predictions, native qualification, immutable source/config and allocation ledger.
Use existing terminal attribution and RML closure, not handwritten acceptance.

`baseline_data_declaration.json` is a projection of the historical K1 post_bn
Spec retaining its one arm and data identity, not a new accepted reference.
The explicitly authorized control is trained in this same physical job.
Budget-field plumbing is reused byte-for-byte from reviewed GPTrans owner
source `32b96f619404160c9c685ba9bd8d1d79d77921f2`; no unrelated GPTrans
experiment/model is integrated here. Frozen remote packages remain unchanged.
