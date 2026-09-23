# Same-platform GPTrans-T / centered-logits 100K protocol

## Question

Does centering valid node-to-pair logits improve GPTrans-T Gap generalization
against an otherwise identical GPTrans-T control trained alongside it on the
same Kaggle1 T4x2 allocation? The earlier frozen GPTrans-T Kaggle2 result is
retained as historical context. The canceled Kaggle3 candidate has no accepted
metric.

## Frozen comparison

- Two independent seed-42 arms: unmodified GPTrans-T `reference` on GPU 0 and
  `centered_logits` on GPU 1. Each sees exactly one T4. Both start from the same
  frozen initial state and use the same committed V4 training source.
- Accepted fixed Kaggle1 graph dataset:
  `nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1`, latest accepted version 4;
  manifest SHA256 `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
- The 100,000 training and disjoint 50,000 development rows are drawn only
  from official training. Official validation, test-dev, and challenge remain
  sealed. The target is direct PCQM4Mv2 B3LYP Kohn-Sham Gap in eV.
- Per arm: deterministic FP32/no TF32, physical BS128, drop-last, no
  accumulation, 60 epochs, 46,860 optimizer steps, 5,998,080 sample
  presentations, normalized L1, AdamW LR 1e-3 and weight decay 0.05, four-epoch
  warmup plus cosine to 1e-6, clip 1, EMA 0.9999, and best-development EMA
  selection. The centering addon changes only the node-to-pair message.

## Release and decision gates

Before the one remote submission, verify the committed source, two-arm spec,
accepted dataset and initial-state hashes, both arms' real-shard CPU model
smokes, exact optimizer/schedule/source values, and isolated GPU dispatch. The
Kaggle wrapper requires exactly two T4 devices and runs each arm's GPU runtime
preflight in parallel. Both must pass repeatability, finite optimizer steps,
memory reserve, and at most six estimated training hours per arm before either
arm trains. Training runs concurrently with independent checkpoints and traces.

Terminal acceptance verifies 60-epoch exposure, selected and resumable
checkpoints, aligned finite 50K development predictions, role use, source/data
identity, and separate measured T4 device costs. Compare candidate minus
same-job reference by paired row bootstrap. A candidate shortlist requires at
least 0.003 eV lower MAE and a paired 95% upper bound below zero. Row bootstrap
does not estimate seed variance. A failed or canceled arm yields no causal
promotion claim. No 500K, full-scale, official-role, or second-seed release is
implied by this screen.

Both arms require separate canonical V5 envelopes, trajectory decisions,
cost/role events, normalized 60-epoch traces, and trace manifests. The two new
RML replay entries are each bound prospectively to the already accepted
Kaggle2 GPTrans-T 100K evidence ID, because their same-job comparator does not
have an accepted evidence ID at launch. The scientific comparison still uses
the two new aligned predictions. Delivery of the paired experiment requires
both new entries in `research_memory/derived/replay_pool.json` with
`capability: complete` and no exclusion reasons. One complete arm is a partial
delivery, not dual replay-ready acceptance.
