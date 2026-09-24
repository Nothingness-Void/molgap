# GPTrans-T FP32 versus FP16 mixed precision, fixed 100K

## Question

Does CUDA FP16 autocast materially shorten GPTrans-T training on Kaggle1 T4
without materially worsening development MAE? Kaggle1 T4 has compute
capability 7.5 and lacks native BF16, so BF16 is not an arm. The FP32 arm is
retrained only because this is a matched training-precision comparison.

## Frozen pair

One Kaggle1 T4x2 job runs one isolated process per GPU. Both use the accepted
PCQM4Mv2 fixed graph dataset (manifest SHA256
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`),
train rows 0–99,999, disjoint internal-development rows 100,000–149,999,
GPTrans-T 12x256/pair32, frozen random seed-42 initialization, normalized Gap
L1, AdamW, physical batch 128, drop-last, four-epoch warmup/cosine schedule,
gradient clipping 1, EMA 0.9999 and best-development EMA selection. Each arm
must complete 60 epochs, 46,860 actual optimizer steps and 5,998,080 sample
presentations. TF32 remains disabled and deterministic algorithms enabled.

The only intervention is precision: the control uses FP32 throughout; the
candidate uses CUDA FP16 autocast for forward/evaluation and dynamic gradient
scaling (initial scale 1024, growth interval 1,000,000), retaining FP32 model
weights, loss accumulation, optimizer and EMA. A skipped or non-finite step
fails acceptance; it is not counted as completed exposure. The checkpoint must
retain scaler state for exact continuation.

## Measurements and decision

Measure synchronized optimizer-loop seconds separately from development
evaluation and checkpoint time for every epoch. Report total native T4 seconds,
median steady-state graphs/second (epochs 5–59), peak GPU memory, and speedup
FP32-seconds / FP16-seconds. Report each arm's best and epoch-59 development MAE
on the same 50,000 source-aligned rows, plus paired absolute-error difference
and a row-bootstrap interval. That interval does not measure training-seed
variation.

For a screening-efficiency nomination, require both complete 60-epoch arms,
zero skipped optimizer steps, at least 20% faster FP16 optimizer-loop time,
and FP16 best MAE no more than 0.001 eV worse than FP32. A nomination does not
change full-scale or production precision without a separate replication and
release contract. If FP16 preflight cannot sustain the frozen recipe, close the
experiment as a capability/precision failure without inventing MAE.

Official validation, test-dev and challenge roles remain sealed. Retain each
arm's runtime certificate, trace, selected model, aligned development
predictions, resumable checkpoint, source/data hashes, exact role use and
measured native cost. Use the existing CLI/RML and Kaggle adapters for planning,
packaging, submission and terminal evidence.
