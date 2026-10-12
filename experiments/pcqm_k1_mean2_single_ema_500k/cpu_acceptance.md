# CPU preparation acceptance - 2026-10-13

Independent tester: gpt-6-luna/max, test/inspect only; no files edited.
BaseHEAD bd25a038; tests targeted the working tree before source publication.
Python: `D:/文档/molgap/.venv/Scripts/python.exe`, worktree src on PYTHONPATH;
CUDA/HIP/ROCR devices hidden. Exact commands:

```text
python -m pytest -q tests/test_kaggle_bootstrap.py tests/test_experiment_workflow.py
python -m pytest -q tests/test_family_training_data.py tests/test_k1_screen_lifecycle.py tests/test_family_training_kernel.py tests/test_family_training_runner.py
git diff --check
```

Results:120 passed and109 passed, zero failures, one PyG deprecation warning
per suite. Whitespace check passed. Tested configurable9h/>4h budgets, legacy
fallback/conflict rejection, cleanup reserve, finite positive pair budgets,
existing mirror format, pure2D geometry stripping and protected-role rejection.
No training hyperparameter change, real-shard qualification, native accelerator
qualification, speed/accuracy acceptance or training replay follows.
Question-specific preparation still requires its own sealed-source release
check before submission; this receipt is not that check.
