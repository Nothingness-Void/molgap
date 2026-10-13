# Validation Scope - 2026-10-13

Tester: Luna Max; parent wrote implementation and interpreted retained outputs.
Checkout: `D:/w/family-checkpoint-efficiency`, branch
`codex/exp/family-checkpoint-efficiency`, production source frozen at`d63544db`.
Prospective scope committed at`8f3695e4` before native timing.

The combined suite covers checkpoint finite errors/mutation/byte identity,
reproducibility, common core/kernel/session, exact mid-pass continuation,
full lifecycle, artifact workflow and GPTrans EMA hotpath.

```powershell
$env:PYTHONPATH='D:/w/family-checkpoint-efficiency/src'
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HIP_VISIBLE_DEVICES='-1'
$env:ROCR_VISIBLE_DEVICES='-1'
& 'D:/文档/molgap/.venv/Scripts/python.exe' -m pytest -q `
  tests/test_checkpoint_finite_efficiency.py tests/test_k1_execution_reproducibility.py `
  tests/test_family_training_core.py tests/test_family_training_kernel.py `
  tests/test_family_training_session.py tests/test_family_full_lifecycle.py `
  tests/test_experiment_family_workflow.py tests/test_gptrans_execution_hotpath.py
```

Explicitly isolated rerun: **250 passed,5 native-only skipped**,0 failures,
174.56s. The earlier empty-mask run reported255 passed/0 skipped in173.97s;
it is not CPU-isolated evidence. On this Windows shell an empty environment
assignment does not establish accelerator isolation. Use explicit`-1` instead.
The first run's GPU allocation cost is unknown, not zero.

Compileall and Git whitespace checks passed in the first test receipt. No
implementation change followed it. Native synthetic result completion is
retained in`results/completion.json`:24 cases, byte-identical serialization,
all checkpoint tensors/RNG unchanged. Native unit-test receipt is separate.
That native selection passed**5 tests,18 deselected** in0.38s on RTX5060.
The native worker used Torch2.7.1+cu128, not the default Torch2.11 training pin.

The first metadata finalization stopped because its cost event pointed directly
to the raw timing JSON, which does not contain canonical cost events. The closure
adapter now points to`acceptance.json`, retaining the raw measurement through
that receipt. Only closure metadata code changed after native execution; frozen
benchmark/config/source commit and measured outputs are not rewritten or rerun.
Preparation now rejects an existing prospective config before writing anything;
closure reuses a published finalization before regenerating terminal timestamps.
These lifecycle-only safeguards do not change the measured runtime checker.
Luna verified blocked preparation (exit1), idempotent closure
(`ALREADY_FINALIZED`, exit0), unchanged config/terminal/result SHA256, compileall,
Git whitespace and`RML check --frozen` (all exit0). No native rerun occurred.

No molecular cache, protected role, trained checkpoint, model forward/backward,
optimizer update, platform submission or scientific training was involved in
the checkpoint benchmark. These tests are not T4/A100/DCU/default-runtime
qualification, a MAE/non-inferiority test or training replay readiness.
