# Independent CPU Artifact Acceptance

Binding: frozen Spec `8f4129600181bda7160bb6b23079fa71072817278208d5cc9ea01386f27fc532`; source `1f627236c1f7780e35411e741352504092e145c8`; package `e71d12b00a30765d5003b42255058d6fcc7668f35352306f07a5577e0d717ba3`; retained Kaggle receipt `kaseichou/molgap-k1-norm-ablation-100k-s42` version 1.

Both arms were checked with `inspect_session_output` using the output of `read_recipe(training_contract.json)` and `TargetIdentityBinding.from_acceptance_plan`. Both native preflight checks passed via `_verify_preflight(..., profiling_required=False)`. CUDA/HIP/ROCR were disabled for this review; only retained tensors were loaded on CPU with `weights_only=True`.

- `k1_single_bn`: mechanical `MECHANICALLY_VERIFIED` (blockers: []); native `NATIVE_PREFLIGHT_VERIFIED` (blockers: []); exposure `{'optimizer_steps': 31240, 'sample_presentations': 3998720, 'completed_passes': 40, 'next_batch': 0, 'status': 'complete'}`; selected step `31240` / `live`; internal-development Gap MAE `0.14129446 eV`.
- `k1_ada_rmsnorm`: mechanical `MECHANICALLY_VERIFIED` (blockers: []); native `NATIVE_PREFLIGHT_VERIFIED` (blockers: []); exposure `{'optimizer_steps': 31240, 'sample_presentations': 3998720, 'completed_passes': 40, 'next_batch': 0, 'status': 'complete'}`; selected step `30459` / `live`; internal-development Gap MAE `0.14518416 eV`.

Mechanical acceptance is not scientific acceptance, promotion, full runtime qualification, or a performance claim. This review independently verified tensor identities, checksums, deterministic repeat/resume native proofs, and exact exposure.
