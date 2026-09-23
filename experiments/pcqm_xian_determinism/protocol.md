# Xi'an reduction determinism audit protocol

## Question

Can the Xi'an Card2 runtime reproduce the reduction operations used by the
PCQM graph models closely enough to permit scientific model ranking?

The motivating observations are separate:

- two nominally matched Kaggle scratch controls differed by
  `0.0026921320 eV` after 40 epochs;
- two identical Xi'an three-epoch runs differed by `0.0214302112 eV` at their
  best checkpoints and by `0.1172343036 eV` at epoch 3.

The Xi'an result also failed exact repeated-forward identity. Fixed random
seeds therefore do not establish repeatability on that runtime.

## Frozen audit

- Runtime: the accepted Xi'an Card2 Torch 1.10 / DTK 22.10 environment and the
  locally rebuilt `torch_scatter 2.0.9` 256-thread wheel.
- Device: one `Z100SM` from `xahdtest`.
- Inputs: synthetic fixed CPU-generated FP32 tensors only. No PCQM labels,
  official validation, or test-dev role is read.
- Modes: one default-runtime process and two independent processes using
  `torch.use_deterministic_algorithms(True)`, with cuDNN benchmark and TF32
  disabled.
- Operators: native `index_add_`, `torch_scatter` add/mean, and sorted
  `segment_csr` sum/mean.
- Evidence: 12 independent forward/backward replays per operator, SHA-256 of
  every output and input gradient, maximum absolute differences, runtime
  identity, and atomic JSON outputs.

## Gates

1. A reduction path is a candidate only if every strict-mode replay is
   bitwise identical in both forward output and backward gradient, within and
   across the two independent processes.
2. Strict-mode rejection is evidence that the vendor runtime cannot promise
   determinism for that operation; it is not treated as a passing run.
3. Operator acceptance alone never authorizes training. The selected path must
   next pass repeated full-model forward/backward replay on the immutable
   100K cache.
4. Only after model replay passes may two identical short training runs be
   submitted. Their initial state, first-batch loss, epoch metrics, and
   checkpoint hashes must be compared before the Xi'an ranking policy changes.

The short training replay intentionally retains the historical
seed-42/three-epoch/batch-48 benchmark contract so its result can be compared
directly with the two failed-repeatability jobs. It is a diagnostic exception
to the batch-128 architecture-screen policy and cannot rank a model.

No full training, official-role evaluation, production change, or threshold
reduction is authorized by this audit.
