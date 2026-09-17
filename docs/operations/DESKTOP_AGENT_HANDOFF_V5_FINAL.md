# MolGap desktop agent handoff v5 — FINAL CONTRACT

Date: 2026-09-17 (Asia/Tokyo)
Document ID: MOLGAP-DESKTOP-V5-FINAL
Owner branch: `molgap-desktop`

This document supersedes all prior desktop handoff contracts, including V3 and V4. V5 is the stable final operating contract unless the user explicitly authorizes a future contract version.

## D0. Read this first

Desktop is intentionally independent from the always-on server.

Desktop assumptions:
- The desktop may be powered off.
- There is no default desktop heartbeat loop.
- There is no default desktop Conversation B.
- Desktop jobs are not handed to the server for monitoring when desktop shuts down.
- Desktop/server conversations are not assumed to communicate.
- If a desktop-owned remote job finishes while the desktop is off, the resulting silent time is ACCEPTED BY DESIGN.
- When the desktop returns, its agent reconciles the real remote state and continues from durable artifacts.
- The server must not monitor or advance desktop-owned work unless the user explicitly makes a new, exceptional delegation for that exact task.

Desktop normally owns:
- full-scale training;
- official evaluation;
- final submission;
- selected desktop-owned 500K work when useful.

Server normally owns its own independent 100K/500K screening loop.

Do not build cross-machine live orchestration.


## D0A. Final-contract rule

Treat V5 as the default architecture for all future desktop work.

Do not add a desktop heartbeat, server fallback, cross-machine conversation bridge, shared live control database, or automatic server takeover merely because desktop can be offline.

Future work may improve implementation, full-run contracts, reference packaging, profiling, recovery, and evaluation discipline without changing this topology.

V5 desktop invariants:
1. Desktop is independent from the server A/B loop.
2. Desktop may be powered off and the resulting silent time is accepted.
3. Desktop-owned remote jobs are reconciled when desktop returns.
4. Desktop normally owns full-scale training, official evaluation, final submission, and selected desktop-owned 500K work.
5. Desktop never duplicates an active server-owned experiment merely because it is online.
6. Full-scale admission requires a qualified evidence package, not a marginal 100K result.
7. Track B progress never silently changes Track A production.
8. Shared code is integrated through reviewed commits, not wholesale branch replacement.

Changing these invariants requires explicit user authorization of a future contract version.

## D1. Why the accepted silent time is not a bug

The server exists to eliminate unattended waiting for server-owned research because it stays powered on.

The desktop has a different trade-off:
- it can be shut down;
- its conversations are not kept continuously active;
- the user accepts that a remote desktop-owned task may finish and wait several hours before analysis resumes.

Do not "fix" this by:
- creating a desktop heartbeat;
- making server poll desktop jobs;
- making server A adopt a desktop scientific chain;
- adding an ownership lease between machines;
- creating a shared live control DB;
- adding cross-machine thread messaging.

The desktop requirement is DURABILITY, not continuous supervision.

A desktop-owned remote run must be able to survive the desktop being offline through:
- immutable submitted source/config;
- remote checkpoints;
- durable output artifacts;
- progress/terminal records where the platform supports them;
- sufficient provenance to recover later.

## D2. Desktop branch responsibility

Keep:
- `master`: stable delivery.
- `molgap-server`: independent server research integration.
- `molgap-desktop`: desktop full/evaluation/submission integration plus desktop-owned 500K work.
- `archive`: inactive/rejected/superseded exact histories.

Desktop does not need server permission for normal desktop-owned work already inside its own recorded authority.

Server does not need desktop permission for its normal bounded server campaign.

The two branches exchange reviewed code/evidence later through explicit Git integration, not live conversation messages.

Do not wholesale merge `molgap-server` to receive one helper.
Prefer small reviewed commits/cherry-picks with provenance.

## D3. Files to inspect

Read, in order:
1. applicable `AGENTS.md`;
2. `BRANCHES.md`;
3. desktop `CURRENT_STATE.md`;
4. relevant `ROADMAP.md`;
5. `TRACKS.md` if needed;
6. `ARCHITECTURE.md`;
7. owning experiment protocol/decision/status;
8. `platforms/REMOTE_HANDOFF.md` before remote access;
9. only touched code.

Before editing:
- inspect branch/HEAD/upstream;
- inspect tracked/untracked state;
- preserve unrelated edits;
- no blind reset/clean/stash/force-push;
- no server AGENTS copy;
- no whole-file AGENTS rewrite.

## D4. Required desktop AGENTS correction

Desktop `AGENTS.md` is not the same as server `AGENTS.md`.

Add a short desktop-only clarification, in substance:

```text
## Desktop offline behavior

Desktop work is operationally independent from the server A/B research loop.
Desktop has no default heartbeat monitor. If desktop is powered off, no server
agent automatically monitors, adopts, repairs, or advances desktop-owned jobs.
Silent time after a remote desktop-owned job finishes is accepted by design.

Therefore desktop-owned remote work must be durable and recoverable: freeze
source/configuration before submission, keep checkpoints and remote artifacts,
and reconcile scheduler/artifact state when desktop returns.

Desktop normally owns full-scale training, official evaluation, final
submission, and any explicitly desktop-owned 500K work. Server independently
owns its own bounded 100K/500K loop. Cross-machine live handoff is not part of
the default architecture.

Details:
docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md
```

Do not add a server-style 30-minute monitor section to desktop.

## D5. Desktop shutdown procedure

Before planned shutdown, do NOT transfer the job to server.

Instead:

1. Ensure the submitted remote task is bound to an immutable source/config identity.
2. Ensure accepted input/cache identities are recorded.
3. Ensure output/checkpoint location is durable on the remote platform.
4. Ensure resume semantics are known and checkpointing is enabled where required.
5. Commit/push required source/protocol changes if that is part of the experiment's normal provenance.
6. Record the remote job/kernel identity locally or in the proper desktop experiment record.
7. Do not require server acknowledgement.
8. Shut down.

No desktop heartbeat is required.

When desktop returns:

1. inspect current branch/worktree first;
2. query the authoritative remote scheduler/platform;
3. inspect terminal logs/progress;
4. retrieve required artifacts;
5. perform mechanical acceptance;
6. perform scientific comparison only when reference/provenance is complete;
7. resume/repair/close/advance according to the desktop experiment's existing authority.

Do not blindly resubmit because the local conversation is stale.

If the remote job finished hours earlier, that delay is expected and not recorded as a server failure.

## D6. Desktop-owned 500K work

Desktop may independently perform some 500K work while online when:
- the task is explicitly desktop-owned;
- source/reference/protocol are clear;
- it does not duplicate an active server experiment;
- budget is available.

There is still no server fallback.

If desktop shuts down during a remote 500K:
- the remote job may continue;
- server ignores it;
- desktop resumes analysis later.

If desktop shuts down during a local GPU task:
- local computation stops with the machine;
- rely on a valid checkpoint/resume mechanism;
- do not claim the server can continue it.

## D7. Reference evidence desktop should expose for reuse

Desktop owns important accepted evidence that may later be useful to server or future desktop work.

Publish compact indexes, not live operational handoffs.

For each reusable reference, index:
- benchmark/config/source identity;
- complete scientific contract;
- prediction artifact/hash/locator where allowed;
- source_idx/target provenance;
- training membership;
- role-use history;
- model/checkpoint/recovery identity;
- runtime certificate and scope;
- learning trace and actually retained prefixes;
- native costs where known;
- acceptance/decision authority.

This is asynchronous evidence sharing.

Server may consume such committed evidence later.
Desktop does not need to message a live server conversation.

Do not retrain a baseline simply because another machine cannot immediately see its artifact. First locate accepted existing evidence.

## D8. Full-scale admission

A new full run should be considered only from complete qualified evidence.

Check:
- frozen candidate identity;
- applicable 100K/500K qualification;
- correct immutable reference;
- aligned artifacts and paired comparison;
- explicit statistical limitations;
- cost on target hardware;
- role-use history;
- exact recovery/schedule plan;
- duplicate full/convergence evidence;
- actual full/evaluation authority.

A server-generated `READY_FOR_DESKTOP` package is simply a durable Git/evidence artifact.
Desktop reads it when convenient after it is online.
No live server notification is required.

If no complete package exists, do not invent a full candidate from stale server summaries.

## D9. Keep three questions separate

For every desktop decision distinguish:

1. mechanism effect under a matched scientific contract;
2. target-budget delivery value;
3. convergence/training-strategy value.

Do not use a different-horizon full result as a pure architecture causal comparison.

Do not let a Track B Gap result modify Track A HOMO/LUMO/production without its separate gate.

## D10. Future 100K/500K/full efficiency

Desktop should help design future target-scale policy from existing evidence, not simply add more stages.

Historical warning:
- small-data positives may fail at scale;
- different data roles/horizons/EMA settings confound transfer;
- a 500K bridge can consume more optimizer steps than a historical full run.

Future low-cost ladders, if adopted, should:
- be prospective;
- be backtested;
- keep candidate/reference at matched prefixes;
- use one declared LR/optimizer trajectory;
- resume the same run across cumulative budget rungs;
- avoid resetting schedule at each rung.

Do not retrofit these rules into already running contracts.

## D11. No desktop A/B implementation

The implementation agent must NOT:
- create a desktop monitor B because the server has one;
- create a desktop heartbeat automation;
- create cross-machine event schemas for live desktop handoff;
- add server fallback monitoring;
- implement desktop/server `owner_epoch` fencing for ordinary work;
- build a shared live SQLite DB;
- make desktop shutdown depend on server confirmation.

Desktop can use ordinary local conversations while it is online, but persistent A/B monitoring is not part of the required architecture.

## D12. Shared code integration

Server may initially implement reusable:
- strict acceptance helpers;
- runner profiling helpers;
- experiment ledger helpers;
- evidence schemas.

Desktop may import reviewed commits later.

Import rules:
- review exact diff;
- preserve origin;
- import only dependencies needed;
- test desktop adapter;
- do not duplicate helper implementations;
- do not bulk merge unrelated server history.

Desktop-specific docs remain desktop-specific.

## D13. Desktop tests

Use local/synthetic tests for infrastructure changes.

Required behavior:

- desktop AGENTS patch does not add server-style monitor logic;
- shutdown procedure requires durable remote artifacts, not server handoff;
- stale desktop conversation after shutdown verifies exact remote job identity before resubmission;
- server-owned 500K is not taken over or duplicated by desktop;
- desktop-owned 500K remains desktop-owned while desktop is off;
- missing reference payload => comparison pending, not baseline rerun;
- incompatible v3 result => not strict v4 comparator;
- consumed role => not called untouched; untouched protected role still requires authorization;
- incomplete D8 evidence => no automatic full;
- full runner resume preserves model/optimizer/scheduler/RNG/cursor;
- Track B positive => no automatic Track A registry change;
- imported common helper failure => no local bypass.

## D14. Implementation order for a lower-capability agent

Phase D0:
- inspect/reconcile only;
- no code changes.

Phase D1:
- install V5 docs;
- narrow desktop `AGENTS.md` and `BRANCHES.md` clarifications.

Phase D2:
- publish reusable reference indexes;
- no baseline rerun.

Phase D3:
- import reviewed shared acceptance/runtime patches when available;
- test desktop adapter.

Phase D4:
- improve full-scale contract/recovery qualification.

Phase D5:
- design future target-budget ladder from backtested evidence when useful.

No new training is authorized merely by this implementation document.

## D15. Final report

Report:

1. branch/worktree facts;
2. files/commits changed;
3. desktop-owned active/pending jobs as observed when desktop is online;
4. durability/recovery readiness;
5. reference evidence exported;
6. shared commits imported;
7. next permitted desktop action or pause;
8. explicit confirmation that no server monitoring/takeover or desktop heartbeat was added.


## D16. Permanent desktop screening/full lifecycle under V5

Desktop should normally receive one of two inputs:

```text
A. desktop-owned 500K question
or
B. server READY_FOR_DESKTOP evidence package
```

Then:

```text
reconcile actual current state
-> verify evidence/reference/roles/cost
-> decide whether target-scale question is still scientifically useful
-> if not useful: close without full training
-> if useful and authorized: freeze exact full contract
-> run/recover full training
-> evaluate only under permitted role policy
-> record delivery decision
-> promote to master only as a separate explicit action
```

Do not treat "server positive" as "desktop must run full."

## D17. V5 change control

Implementation discomfort is not a reason to redesign the project.

If a V5 invariant appears wrong:
- preserve the current workflow;
- collect evidence;
- write a separate `V5_CHANGE_PROPOSAL`;
- do not activate it without explicit user authorization.

No desktop agent may self-declare V6.

