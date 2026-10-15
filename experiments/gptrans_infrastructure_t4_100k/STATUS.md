# GPTrans-T execution status

Exact private Kaggle3 kernel138811813/version1 completed and passed saved-output
and native-proof acceptance on 2026-10-15. The
[terminal decision](results/decision_2026-10-15.md) owns speed, numerical drift,
profiler observations and cost. Both arms are RML-closed as excluded execution
diagnostics; no automatic successor, scientific promotion or Replay is released.

| Binding | Value |
|---|---|
| Actual job | [nvoid912/molgap-gptrans-infrastructure-dual-100k-v1](https://www.kaggle.com/code/nvoid912/molgap-gptrans-infrastructure-dual-100k-v1), kernel138811813/version1 |
| Frozen executable source | `3bde7ee7aeff1868df8a9f503d81f31633db94ab` |
| Spec identity | `f532dc44196c4aa5c103698af1e2246c65a911e783c3e9f9bb70c0fc26e08769` |
| Source dataset | `nvoid912/molgap-gptrans-infrastructure-source-v1`, version1, private |
| Allocation | Explicit `NvidiaTeslaT4`; two observed T4s, one independent worker per device; complete cost in terminal evidence |
| Arms | Original `gptrans_t/1` reference and optimized `gptrans_t/2` reference; no addon |
| Exposure | Fixed100K, internal50K development, seed42, batch128,60passes/46860updates per arm |
| Observation | Both skip120 training updates and capture16 actual updates with CUDA profiler |
| Budget | 7wall hours/14allocated T4 hours including setup;60s stop reserve |

The [release binding](submission/release_binding.json),
[API source verification](submission/source_verification.json),
[11-file upload verification](submission/source_upload_verification.json),
[quota observation](submission/quota_before_submit.json),
[platform response](submission/platform_response.json),
[launch receipt](submission/launch_receipt.json) and
[scheduler observation](submission/scheduler_observation.json) retain distinct
authorities. Remote source bytes and mounts matched the frozen local package;
the returned ID/version and private T4 source were verified through the API.
The [terminal scheduler/source observation](results/terminal_scheduler_observation.json)
and [exact-version retrieval inventory](results/artifact_retrieval.json) retain
completion, unchanged remote bytes/mounts and hash-pinned output transport.

Prelaunch CPU/static checks were retained separately from native qualification.
Both native preflight workers passed before either formal worker started.
Acceptance rechecked the original CUDA profile/plain full-state proof and the
shared CUDA repeat/resume, profile/plain and fill-on reference proofs.

Exact-version retention/report and42selected pinned files were retained through
the [bounded transport](../../docs/operations/FAMILY_EXECUTION_PROFILE.md).
Original scientific output is under `experiment/original/normalized`; its raw
profile/proofs/checkpoint chunks remain under `experiment/original`. Shared output
is under `experiment/optimized`. Both matching owning inspectors and aligned
selected-EMA comparisons passed. Historical bond-local results were not used.

Ignored raw records: `platforms/_records/kaggle/gptrans_infrastructure_v1/`.
