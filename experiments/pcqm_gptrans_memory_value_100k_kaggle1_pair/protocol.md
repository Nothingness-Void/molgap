# GPTrans persistent-pair readback 100K protocol

## Question and frozen intervention

The accepted GPTrans propagation-flow ablation showed that removing direct
pair-to-node readback worsened 100K development MAE by 2.657 meV. The model's
existing readback weights and values use only the freshly generated relation
update. This screen asks whether keeping the persistent pair state in the
readback values improves generalization. The sole candidate intervention is
`memory_value` in `src/molgap/gptrans_memory.py`: weights still route by the
current update, while values become the sum of the old pair state and update.
The block's residual pair update remains unchanged. This mechanism has no
accepted PCQM screen in the desktop RML index.

Run two independent seed-42 arms in one Kaggle1 T4x2 kernel: unmodified
GPTrans-T `reference` on GPU 0 and `memory_value` on GPU 1. Use the same frozen
initial state, source archive, accepted fixed 100K graph dataset, 100,000
official-train-derived training rows and disjoint 50,000 internal-development
rows. Keep separate processes, checkpoints, traces, predictions, costs and RML
trajectories. Official validation and all test roles remain sealed.

## Training and decision gate

Use the GPTrans V4 12x256/pair32 architecture, normalized Gap L1, AdamW
LR 1e-3 with weight decay 0.05, four-epoch warmup then cosine to 1e-6,
physical batch 128 per T4, drop-last, FP32 with TF32 disabled, EMA 0.9999,
best-development EMA selection and seed 42. Each arm must complete 60 epochs,
46,860 optimizer steps and 5,998,080 sample presentations. The frozen graph
manifest is `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.

The primary comparison is candidate minus same-job reference on aligned 50K
development rows. Shortlist only if candidate MAE is lower by at least
0.003 eV and the paired row-bootstrap 95% upper bound is below zero. A row
bootstrap does not measure seed variance. Failure closes this mechanism under
this contract; success nominates it for a separately frozen scale decision.
Neither result automatically releases another seed, 500K, full training,
official evaluation or production promotion.

## Release and evidence

Before one remote push, verify frozen source and package, accepted dataset and
initial-state hashes, per-arm local loader/syntax checks, exact recipe, and
isolated T4x2 dispatch. Each remote worker must pass the existing GPU runtime
preflight, including repeatable finite optimizer steps, memory reserve and an
estimated training duration no greater than six hours, before either trains.

Use the shared ExperimentSpec v2 `same_run_replay` binding and
`plan-prospective` for the pair. A terminal result requires separate V5
envelopes, role/cost events, selected and resumable checkpoints, 60-epoch
canonical traces, aligned finite 50K predictions, and matching same-run source,
dataset and platform identities. Terminal closure must accept the control
first and bind both replay manifests to its accepted evidence ID. Only two
distinct `capability: complete` replay-pool entries with no exclusion reasons
establish dual replay-ready delivery. Before terminal acceptance, RML contains
only ACTIVE prospective trajectories; a launch receipt is not replay evidence.
