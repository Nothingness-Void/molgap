# Execution status

2026-10-09 account migration: Kaggle3 accepted the same frozen pair as
`nvoid912/molgap-k1-consistency-500k-pair-s42-v1`, ID137710959/version1.
The authoritative [scheduler observation](submission_kaggle3_v1/scheduler_observation_v1.json)
is QUEUED; actual allocation, optimizer qualification and resumed execution are
not observed. API source bytes, private T4 request and dataset membership match
the release-bound payload. See [response](submission_kaggle3_v1/platform_response.json),
[verification](submission_kaggle3_v1/source_verification.json) and
[migration authorization](continuation_authorization_kaggle3.md).

Resume both arms from46/60,179676steps using the accepted version2
model/optimizer/RNG states. Original Spec and recipes remain unchanged.
Executable source commit `2cf7c56f84dacf89fd69c97c24cde0d284d81709`,
archive `b2b7539d7fbd0fbff6e674ed632ab90be00f86e8484eb6f2e8cb3ddafe06a3fd`,
package `a04a203a40a1207e454e413db525f0862e536f257474ae089d03a821a6e08e0f`.
The30000-second window fits the remaining17.158148T4-hour training allowance;
new native costs and terminal raw/clean-BN endpoints remain pending. Both RML
trajectories stay ACTIVE, not training replay-ready. No local trigger or monitor.

The previous Kaggle1 observations below remain historical evidence.

2026-10-09 reconciliation: version2 is COMPLETE with accepted bounded stages
at46/60epochs in both arms, not terminal training. See [stage acceptance](submission_v2/terminal_inspection_20261009/stage_acceptance.md)
for checkpoints, exact source, metrics and cumulative34.841852T4-hour training
allocation. Fourteen epochs and the predeclared clean-BN endpoint comparison
remain pending. No continuation was submitted by this acceptance.

The retained submission-time observation below is historical, not current scheduler state.

Kaggle1 version2 continuation was accepted and authoritative snapshot is RUNNING.
Exact kernel `nothingnessvoid/molgap-k1-consistency-500k-pair-s42-v1`, ID137483676,
version2; script-version ID remains unknown. Exact API source bytes, T4 request
and the three published mounts agree with the release-bound payload. Dataset
membership matches; Kaggle reordered those mounts. The initial log is empty;
actual native allocation, optimizer preflight and resumed training are not yet
observed, and are not inferred from RUNNING.

Both arms resume at epoch index23 (23/60 complete,89,838steps,
11,499,264presentations) using verified model/optimizer/RNG/cursor state.
Original scientific Spec and recipes are unchanged. Each bounded invocation
allows at most32,400seconds, so another reconciled stage may be needed to60epochs.
No default monitor, server takeover, full-data training or protected role use.

| Identity | Value |
|---|---|
| Spec | `65fc71b0a622a8661df33b061915e0630dfb57abf5e01546cfc1803489f23c12` |
| V2 executable source commit | `529c138838eabe57b908b6039cd63b3e57c48595` |
| V2 source archive SHA256 | `c3a94757a9fb77d8096e9f30a5af0fdcd7d494d5dfe1a041ea7f02471a257c74` |
| V2 package identity | `dcd226df9827ec3b30cc90ad565dc85fadfab9e5ba1a13fba913eed3753af6a9` |

Private source: `nothingnessvoid/molgap-k1-consistency-500k-source-s42-v2`.
Private recovery: `nothingnessvoid/molgap-k1-consistency-500k-resume-s42-stage23-v1`.
Graph: `nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1`.
Original producer stage manifests and prospective launch bytes are unchanged;
selected/resume retention omits only unneeded numbered epoch predictions.
The shared retention repair passed32focused tests in one local run. Shared release
checks passed before publication and were repeated by the existing submitter.
No training artifacts were retrieved after this healthy submission.

Version1 [bounded acceptance](submission_v1/terminal_inspection/stage_acceptance.md)
retains selected MAE0.109245628119 (coefficient0) and0.110386952758(coefficient0.1).
Its measured training allocation17.389569095T4hours leaves34.610430905 under52hours;
version2 incremental costs remain unknown until an actual ledger is retained.
Both independent RML trajectories remain ACTIVE and neither is replay-ready.

Read [continuation authorization](continuation_authorization.md),
[actual response](submission_v2/platform_response.json),
[source verification](submission_v2/source_verification.json),
[continuation binding](submission_v2/continuation_binding.json) and
[return procedure](REMOTE_HANDOFF.md) before further action.
