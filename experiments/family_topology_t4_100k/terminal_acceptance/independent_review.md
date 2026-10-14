# Independent CPU Artifact Acceptance

Binding: frozen Spec `9db73758d7f12974219bcf8d5eafd25789a7f2a675660eeddb53b87c5a6068bc`; source `35ddecdc0ccf7e9ca664588cc836bd2ecb7ec6f3`; package `ed6bb15dc3ac8a668e299b50ee5fc8898dc638b4301de7b6193b56bc2d563e5b`; retained Kaggle receipt `nothingnessvoid/molgap-k1-gptrans-topology-100k-s42` version 1.

Both arms were checked with `inspect_session_output` using the output of `read_recipe(training_contract.json)` and standard `tensor_digest` (`target_identity=None`). Both native checks used `_verify_preflight(..., profiling_required=True)`. CUDA/HIP/ROCR were disabled for this review; only retained tensors were loaded on CPU with `weights_only=True`.

- `k1_mean2`: mechanical `MECHANICALLY_VERIFIED` (blockers: []); native `NATIVE_PREFLIGHT_VERIFIED` (blockers: []); exposure `{'optimizer_steps': 31240, 'sample_presentations': 3998720, 'completed_passes': 40, 'next_batch': 0, 'status': 'complete'}`; selected step `28897` / `live`.
- `gptrans_local`: mechanical `MECHANICALLY_VERIFIED` (blockers: []); native `NATIVE_PREFLIGHT_VERIFIED` (blockers: []); exposure `{'optimizer_steps': 46860, 'sample_presentations': 5998080, 'completed_passes': 60, 'next_batch': 0, 'status': 'complete'}`; selected step `40612` / `ema`.

Mechanical acceptance is not scientific acceptance, promotion, full runtime qualification, or a performance claim. This review did not rerun saved-prediction comparison, inspect formal profiling timings, contact the remote service, or execute a model.
