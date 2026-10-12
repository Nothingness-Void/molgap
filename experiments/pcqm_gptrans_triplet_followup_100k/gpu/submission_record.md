# Physical submission — 2026-10-13

One Kaggle2 T4 dual-arm task was submitted through the owning accelerator
adapter with the actual release report. The returned identity was
`kaseichou/molgap-gptrans-triplet-followup-dual-s42`, kernel138367521/version1.
Remote metadata and the pulled entry script matched that identity and the
frozen local script. The observed API state was RUNNING. This observation did
not prove formal training/preflight completion; startup JSON was not yet
retrievable through the exact-output API and no bulk fallback was used.

Private source dataset creation produced version1; a pre-training version2
added the separately frozen deferred post100K audit action, without changing
the training source/initialization bytes. Dataset state was ready and its
listing contained that action. The source publication record binds all local
uploaded file hashes. Candidate checkpoint hashes remain unavailable until
actual terminal acceptance; the audit action is conditional, not an executable
NO_TRAIN release or a terminal RML claim.

The local `prepare-release` operation reused existing staging and source gates;
its successful phase took55.20seconds, excluding scientific addon coding,
reference qualification, tests, commits and uploads. It reported
LOCAL_RELEASE_INPUTS_VERIFIED and two prospective RML plans. The first local
attempt correctly rejected untracked contract/configuration files before any
compute release. Their classified commit preceded successful preparation.

Checks:44 focused synthetic/static regressions passed. Both initialized model
constructors loaded exactly with5,880,961and5,885,185parameters; no molecular
forward/backward, training or inference ran locally. Repository RML validation
passed. Existing user edits were not staged into this task's commits.

Existing Luna B and the existing30-minute heartbeat were rebound, not recreated.
The one manual monitor-wiring probe returned RUNNING/SILENT with no event.
Completion/actionable fault is handed to the existing A through an idempotent
event; A-model overrides are omitted. No automatic training successor,
baseline rerun, extra seed, desktop custody or protected-role access was released.
Independent terminal acceptance and actual Replay pool admission remain pending.
