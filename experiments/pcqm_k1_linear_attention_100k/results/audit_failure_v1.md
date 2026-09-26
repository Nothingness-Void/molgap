# Training completed; audit environment failure (2026-09-26)

Kaggle `nothingnessvoid/molgap-k1-linear-attention-s42` version 1 ended ERROR.
The training worker exited zero after all 40 epochs. The independent audit
sibling failed at its first CuBLAS operation, before publishing an inference
chunk. No fixed500K audit result was produced.

The training child set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, but child environment
changes cannot reach the parent or the later sibling audit. The audit enabled
deterministic algorithms without this CuBLAS setting. Dependency-resolver
warnings were not the terminal cause. The repair sets the environment in the
launcher, before importing Torch; deterministic checks remain enabled.

## Verified training, not whole-job acceptance

No-inference saved-artifact acceptance passed. The candidate MAE was
0.1417725831270218 eV versus K1 0.1413736343383789 eV. Candidate minus K1 was
+0.0003989487886428833 eV, with paired row-bootstrap 95% interval
[-0.0005090527391433715, 0.0013640565963983534] eV. The experiment-local
0.003 eV promotion gate failed. This did not establish significant inferiority
or justify a universal claim against linear attention.

The actual accelerator was Tesla T4. Parameter count was 3,582,209, best epoch
39 (zero-based), observed optimizer steps 31,240 and presentations 3,998,720.
Reported mean throughput was 811.938 graphs/s and peak reserved memory 634 MiB.
No extra seed, retraining, 500K training, full run or successor was released.

Retained original root:
`platforms/_records/kaggle/training/pcqm_k1_linear_attention_s42_v1/`.

- `training_acceptance.json` SHA256:
  `a40e13a2c7dce882173a4b6e432039dd8486db8a6a90329b436278f223c242d4`
- `logs/kernel.log` SHA256:
  `a08c269ffa3680a28d599c9e016dbc01f442a40df3fd1d36d36a958e15edb1d9`
- Original source/model/payload bindings: [recovery specification](../audit_recovery.json).
- Native canonical trace, all atomic checkpoints and recovery archives were
  retained at the original root; no failed outputs were overwritten.

## Bounded infrastructure recovery

The preplanned NO_TRAIN audit alone was authorized to recover. A separate
private input package binds the accepted best checkpoint and saved predictions;
it does not contain an optimizer or authorize training. The original scientific
source remains unchanged. The repaired launcher is separately versioned.
The original audit prospective plan remains immutable; the failure and recovery
are separate operational attempts, not a new architecture trial or a new seed.

The audit reproduces original internal dev predictions before opening the fixed
500K internal dev role. BS128, FP32, transform, selected weights and chunks
remain identical. The latter role is reused diagnostic data, not sealed
confirmation. Protected official roles remain untouched.

At this decision boundary, only training was accepted. Audit acceptance,
terminal RML ingestion and new replay entry were pending. No PASS was fabricated
to conceal the failed audit. Local repair tests: 58 passed; no local training or
model inference.
