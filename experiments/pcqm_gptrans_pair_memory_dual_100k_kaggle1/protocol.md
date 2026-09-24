# GPTrans pair-memory dual-candidate 100K protocol

## Question and interventions

The accepted GPTrans flow ablation showed that removing pair-to-node readback
worsened 100K development MAE by 2.657 meV. The following existing model
variants have not been screened under the accepted fixed 100K contract:

- `memory_value`: route readback weights on the current relation update, but
  use accumulated pair state plus update as values.
- `memory_message`: route both readback weights and values on that accumulated
  pair state.

Both retain the GPTrans-T core and alter only pair-to-node readback. They are
separate scientific questions, not a baseline/candidate split. Compare each
with `pcqm-gptrans-t-100k-v4-reference` using its accepted V5 evidence and
aligned development rows. The two candidate arms share a Kaggle1 T4x2 kernel
but retain separate processes, checkpoints, traces, predictions, costs, role
events, decisions and RML trajectories. An exploratory comparison between
candidates does not replace either baseline comparison.

## Frozen training and gate

Use the accepted Kaggle1 fixed PCQM4Mv2 100K graph dataset with 100,000
official-train-derived training rows and disjoint 50,000 internal-development
rows; manifest SHA256 is
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
Both arms use frozen initial state SHA256
`9205fc0f0f97f1cc1cea84ab4bd24206274a00c7d84d26366497feee1710c20c`,
seed 42, GPTrans V4 12x256/pair32, normalized Gap L1, AdamW LR 1e-3 with
weight decay 0.05, four-epoch warmup then cosine to 1e-6, physical batch 128
per T4, drop-last, FP32 with TF32 disabled, EMA 0.9999 and best-development
EMA selection. Each arm must complete 60 epochs, 46,860 optimizer steps and
5,998,080 sample presentations. Official validation and test roles stay sealed.

For each candidate, shortlist only if its MAE beats the frozen GPTrans-T V4
reference by at least 0.003 eV and the paired row-bootstrap 95% upper bound
for candidate minus reference is below zero. The row bootstrap does not
measure training-seed variation. Failure closes that candidate under this
contract; success only nominates a separately frozen scale decision.

## Release and acceptance

Before one remote push, verify exact source/package/data/initial-state hashes,
the frozen numeric recipe, syntax/import and real-shard loader preflight. The
remote T4x2 runner must certify both candidate workers independently before
training, including repeatable finite optimizer steps, memory reserve and an
estimated run duration no greater than six hours.

Use the existing ExperimentSpec v2 `plan-prospective`, package, preflight,
launch receipt, GPTrans acceptance and RML terminal commands. Each terminal
arm needs its own runtime certificate, selected/resumable model, 60-epoch
canonical trace, aligned finite 50K development predictions, exact role-use
events, measured native T4 cost, V5 envelope, strict comparison-readiness
record and trace manifest bound to the accepted GPTrans-T reference. Require
two distinct `capability: complete` replay-pool entries with no exclusion
reasons before declaring dual replay-ready. Queue or launch state is not
terminal evidence.
