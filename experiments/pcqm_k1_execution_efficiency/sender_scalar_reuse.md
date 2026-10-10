# Single loss-scalar read

2026-10-10; synthetic infrastructure acceptance, not a scientific trajectory.
The frozen sender `_step` reads the same detached finite loss twice, once for
`train_loss_sum` and once for its return. A [minimal patch](sender_scalar_reuse.patch)
reuses one Python scalar without changing guards, counter order or the owner step.
No shared helper or duplicate trainer is needed for this three-line change.

Source owner: `colab_k1_screen.py`, SHA256
`ac13918013015c27b1f46020b0eabf2b4f7ad0eb18bb091a9e6033b94948dc34`.
The [immutable snapshot](checkpoint_io_attempt_001/source_snapshot/colab_k1_screen.py)
is not modified. The live-source question stays with its own branch/contract.

[Tests](../../tests/test_k1_sender_scalar_reuse.py) isolate only `_step` and
reverse-check the surgical AST transformation. Thirteen CPU synthetic cases verify
two-to-one conversion count, returned scalar/counters/events, EMA order, mocked
CUDA-sync branch and rejection/cleanup for nonfinite/malformed output, owner
failure, loss NaN, geometry, three dtype failures and deadline expiry. Relevant
regression run:73 passed, one upstream PyG deprecation warning.

`git apply --check --ignore-space-change` passed against the hash-pinned original
owner checkout. This was read-only; the patch has not been applied to any trainer,
submitted source or active job. CRLF context requires the stated check option.
No molecular data, real model/optimizer, gradients or CUDA work was performed.

These tests qualify host metadata/control-flow parity only. Native scalar-read
cost and full-model numerical/speed qualification remain unknown. Do not describe
this as fewer forwards, a2x speedup, a completed GPU optimization or training
replay evidence. Apply/re-freeze/requalify only with the next reviewed owner package.
