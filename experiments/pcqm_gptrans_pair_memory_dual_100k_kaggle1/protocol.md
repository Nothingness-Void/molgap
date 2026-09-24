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
two new scientific directions, not a GPTrans-T baseline rerun. In the same-run
replay binding, `memory_value` is the relative reference and `memory_message`
is the candidate. Compare both development endpoints contextually with the
accepted V5 GPTrans-T V4 reference; those cross-run values alone do not form
a new strict causal claim. The new arms share a Kaggle1 T4x2 kernel but retain
separate processes, checkpoints, traces, predictions, costs, role events,
decisions and RML trajectories.

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

The primary frozen paired gate is `memory_message` versus same-run
`memory_value` on aligned 50K development rows: shortlist the former only if
its MAE is lower by at least 0.003 eV and the paired row-bootstrap 95% upper
bound is below zero. The row bootstrap does not measure training-seed
variation. Failure closes this mechanism comparison under the contract;
success only nominates a separately frozen scale decision. Absolute scores
against the GPTrans-T V4 reference remain contextual until a separately
qualified cross-run strict reference bundle exists.

## Release and acceptance

Before one remote push, verify exact source/package/data/initial-state hashes,
the frozen numeric recipe, syntax/import and real-shard loader preflight. The
remote T4x2 runner must certify both candidate workers independently before
training, including repeatable finite optimizer steps, memory reserve and an
estimated run duration no greater than six hours.

Use the existing ExperimentSpec v2 `same_run_replay`, `plan-prospective`,
package, preflight, launch receipt, GPTrans acceptance and RML terminal
commands. Each arm needs its own runtime certificate, selected/resumable
model, 60-epoch canonical trace, aligned finite 50K development predictions,
exact role-use events, measured native T4 cost and V5 envelope. Bind the
reference trace to its own accepted evidence; bind the candidate trace and
strict comparison-readiness record to that same evidence after accepting the
reference. Require two distinct `capability: complete` replay-pool entries
with no exclusion reasons before declaring dual replay-ready. Queue or launch
state is not terminal evidence.
