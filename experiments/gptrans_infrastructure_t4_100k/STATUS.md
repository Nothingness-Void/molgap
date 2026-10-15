# GPTrans-T execution status

One private Kaggle3 submission was accepted on 2026-10-15. Exact
kernel138811813/version1 was observed `RUNNING` at 07:45:20 UTC; native runtime
qualification and formal training completion remain unverified. This is the
user-authorized [execution diagnostic](protocol.md), with no automatic successor.

| Binding | Value |
|---|---|
| Actual job | [nvoid912/molgap-gptrans-infrastructure-dual-100k-v1](https://www.kaggle.com/code/nvoid912/molgap-gptrans-infrastructure-dual-100k-v1), kernel138811813/version1 |
| Frozen executable source | `3bde7ee7aeff1868df8a9f503d81f31633db94ab` |
| Spec identity | `f532dc44196c4aa5c103698af1e2246c65a911e783c3e9f9bb70c0fc26e08769` |
| Source dataset | `nvoid912/molgap-gptrans-infrastructure-source-v1`, version1, private |
| Allocation | Explicit `NvidiaTeslaT4` request; two independent workers/devices declared; observed full allocation awaits native evidence |
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
The bounded [startup observation](submission/startup_observation.json) retained
116 log events including Torch2.7.1+cu126/PyG2.6.1 installation. The log stream
ended with a transport exception; that is not a job-failure observation.

Local checks passed:165 regression tests with one host symlink-permission skip,
85 final focused tests, RML validation and897-document pointer audit. These are
CPU/static checks, not native qualification or a speed result. Both native
preflight workers must pass before either formal worker begins.

After completion, retrieve exact-version execution retention/report, then the
report's hashed `profile_artifacts` via the
[bounded transport](../../docs/operations/FAMILY_EXECUTION_PROFILE.md).
Kaggle scientific output is under `experiment/original/normalized`; its raw
profile/proofs/checkpoint chunks remain under `experiment/original`. Shared output
is under `experiment/optimized`. Use the matching owning inspectors and aligned
selected-EMA predictions; do not substitute the historical bond-local model.

Ignored raw records: `platforms/_records/kaggle/gptrans_infrastructure_v1/`.
