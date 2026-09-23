# Prospective protocol: K1 FP32/TF32 execution comparison

## Question and scope

On a single IMS NVIDIA A100, do FP32-stored K1 operations with TF32 *matmul*
compute improve training throughput without materially changing the fixed
PCQM-100K internal-development Gap MAE? This is a newly declared precision
intervention. It cannot be pooled into the existing strict-FP32 V4 reference
or used to modify an already running desktop-owned full job.

## Frozen paired conditions

- Two independent, **sequential** arms on the *same* visible Ampere-or-newer
  GPU: `fp32` then `tf32_matmul`. A single A100 is necessary because T4x2
  cannot execute TF32. The arm sequence does not share trained weights,
  optimizer state, checkpoint, or best-epoch selection.
- Architecture: unchanged K1-v4, 3,658,817 parameters. Both arms have identical
  seed-42 initial model-state SHA-256. No model/module/data changes.
- Input: only the accepted cross-platform fixed 100K/50K PCQM cache, manifest
  SHA-256 `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
  OGB atom9/bond3/RWSE16; the cached geometry fields are removed before batch.
- Train: the first 100,000 accepted official-training rows; internal
  development: the following 50,000 official-training rows. Never read
  official validation, test-dev, test-challenge, or any shadow role.
- Direct Gap, normalized L1 loss using the training-role mean and sample std.
  Physical BS128, `drop_last`, deterministic epoch row order, 781 steps and
  99,968 presentations per epoch, 40 epochs total (31,240 steps; 3,998,720
  presentations) **per arm**.
- Same AdamW `lr=4e-4`, weight decay `1e-5`, gradient clip 1.0, epoch-wise
  cosine to `1e-6`; one visible GPU, no accumulation, same PyTorch environment.
- Both store parameters/targets/predictions in FP32. The sole intervention is
  `torch.backends.cuda.matmul.allow_tf32` false versus true (equivalent
  `highest` versus `high` in this runtime). cuDNN TF32 remains false in both.
  Deterministic-algorithms mode stays on. Hardware arithmetic must be shown to
  change by a real GEMM probe before any model training.

## Measurement and decisions

The primary cost metric is synchronized training-loop seconds and resulting
graphs/s over all 40 epochs; report validation seconds and total allocation
separately. Report both best internal-development MAEs and epochs, their signed
difference, initial-state identity, two-repeat optimizer calibration, complete
loss/metric finiteness, peak memory, and exact artifact hashes. This is a
single-seed numerical/runtime comparison, not proof of accuracy equivalence.
Only a prospective, separately reviewed precision contract could authorize
any future use of TF32; **no automatic full-training change follows**.
V5 classifies this as a noncausal runtime/precision diagnostic. Its two arms
have different `precision_identity`, so the current RML strict screening replay
pool must not merge them or claim a replay-ready causal model win. Both arms
still receive prospective identity, canonical traces, role history, native-cost
records, terminal artifacts, and a no-inference paired endpoint acceptance.

Each arm atomically writes an epoch checkpoint, trace, best model, aligned
50K prediction payload, and every-10-epoch recovery tar. Terminal acceptance
must recompute MAE and verify aligned source indices/targets without model
inference. A wall-time or infrastructure interruption requires explicit resume
after scheduler diagnosis and checkpoint identity validation. Never overwrite
accepted prior outputs.
