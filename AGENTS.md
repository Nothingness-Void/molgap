# AGENTS — Reading Protocol

MolGap predicts HOMO/LUMO/Gap (eV) for organic molecules from PubChemQC
B3LYP/6-31G*. This file is routing plus hard constraints; one fact lives in one place.

## Read in this order
1. This file.
2. **`CURRENT_STATE.md`**: the only source of what is true now (recommended model,
   blocker, next actions). It wins any conflict.
3. **`ROADMAP.md`**: priorities and backlog; read the relevant section only.
4. **`TRACKS.md`** only for tasks labeled A, B or C.
5. **`ARCHITECTURE.md`** (which file owns a change) and **`NAMING.md`**.
6. The one `README.md` of the tree you work in (`production/`, `experiments/`,
   `platforms/`), then the decision record it links.

`production/history/` is frozen; never infer live state from it. The durable V5
contracts are `docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md` and
`docs/operations/SERVER_AGENT_HANDOFF_V5_FINAL.md`; they supersede older handoff rules.
Protected roles (official_validation, test_dev, test_challenge) are never used,
evidence is never fabricated, and frozen evidence is never rewritten.

## How to change code
- Make only the change the request needs: no unrequested refactors, features,
  abstractions, dependencies, workflows, or tests for speculative or unlikely edge
  cases. If a necessary change would grow scope or complexity, state the concrete
  failure, the evidence it occurs and the maintenance cost, then get user approval.
- Read the neighbouring code first and match its naming, structure and comment
  density. Comments say why, not what. Reuse what exists: reusable logic lives in
  `src/molgap/`, stage and experiment `scripts/` are thin CLI wrappers, and model
  classes are never redefined in scripts.
- Fix the root cause once. State the cause in one sentence before editing. Do not add
  a guard, retry, special case or message-only change for a symptom whose cause is
  unknown; temporary diagnostics to find the cause are allowed.
- If the same check fails the same way after your fix, do not patch it again: write
  down the cause and the assumption that was wrong, then change approach or report to
  the user.
- Each fact (hash, pin, identity) has one owning check. Later stages call that owner or
  read its recorded result; never reimplement the comparison, and never remove an
  owning check without approval.
- Derive, do not copy: identities, refs and retry records come from recorded evidence
  by code, not from hand-edited copies of a predecessor.
- Never revert changes you did not make. Use no destructive git command (reset --hard,
  forced checkout or push, clean) unless asked.

## Tests and verification
- Verify the affected behaviour locally before delivering, as
  `docs/operations/INFRASTRUCTURE_VERIFICATION.md` directs: it owns check selection,
  permitted claims and test growth. A passing-test count never establishes submission
  readiness. Never run an implicit full suite.
- A fix adds no test by default: reproduce it through the nearest existing workflow test
  and extend that with one case if it missed the behaviour. A new test file needs a
  stated reason and a raised ceiling (below).
- Assert behaviour, not wording. Match an exact error message only when the message is
  the contract.
- When a test fails: read the failure, decide whether your change caused it, fix the
  cause. Never weaken or delete an assertion to pass, and never add tests to justify a
  patch. Report unrelated failures; do not fix them.
- Test ceiling: `TEST_SUITE_LIMITS` in `src/molgap/documentation_check.py` caps the
  count and total lines of `tests/test_*.py`; any change that adds test lines runs
  `tests/test_documentation_check.py`. Raise the cap only in a commit that states why.
- The coordinator does not run tests. It assigns test workers (at most five at once,
  independent module scopes, separate scratch and report paths, within available
  memory). A worker runs its assigned suites itself, never spawns or relays to another
  worker, is reused for a related retry, and is closed after reporting. Report a
  recurring environment blocker instead of multiplying workers. This grants no remote
  compute or experiment authority.

## Hard constraints
- **Python**: always `.venv\Scripts\python.exe`; system Python lacks torch/pyg.
- **Train-inference consistency**: both use the same conformer method (ETKDG). Never
  mix PM6 training coordinates with ETKDG inference.
- **Targets**: `homo`/`lumo`/`gap` (eV, B3LYP Kohn-Sham), not experimental values.
- **Shared experiment CLI**: `molgap.experiment_cli` is a local identity, packaging,
  diagnostic, receipt and terminal entry point only. Never add a platform submitter;
  use the owning platform adapter and workload skill.
- **Experiment plumbing**: read `.agents/skills/molgap-experiment-reuse/SKILL.md`
  before writing a runner, diagnostic, packaging or RML adapter. Do not recreate an
  existing wheel.
- **Completed experiments** are not re-run; cite the experiment's decision record
  under `experiments/`.
- **Remote durability**: every cloud job checkpoints atomically and produces
  independently retrievable output chunks; never rely on a transient worker
  filesystem or one long task as the only copy of results.
- **Remote resource separation**: high-memory CPU jobs do parsing, ETKDG, graph
  construction and acceptance. GPU/DCU jobs start only after an immutable graph cache
  passes acceptance and are reserved for encoder training, embedding extraction and
  fusion.
- **Kaggle accelerators**: request `NvidiaTeslaT4` only (P100 is unavailable). With two
  or more independent candidates that can be isolated, use T4x2 candidate parallelism,
  each worker with its own model, RNG, optimizer, checkpoint directory and one visible
  GPU. One justified candidate may still receive two devices: isolate one visible
  device before importing CUDA, record the full allocation and cost, and never run an
  unapproved second experiment on the idle device. Record why a job needs a single
  accelerator.
- **Seed budget**: architecture discovery defaults to one paired seed-42 screen. A win
  never automatically triggers seeds 43/44; multi-seed confirmation is reserved for
  the final shortlist after a material gain and an explicit compute decision. Protocols
  may restrict but not broaden this rule.
- **Screen comparability**: new model screens follow `experiments/SCREENING_POLICY.md`
  (physical batch exactly 128 per independent model/device). A baseline is trained and
  frozen once per benchmark contract. Cross-platform candidates compare with that
  reference when the scientific-contract fingerprint is identical and each
  platform/runtime has one reusable accepted calibration certificate; platform
  identity is provenance, not a forced match.
- **Fixed PCQM data identity**: every PCQM 100K/500K screen on Kaggle, SCNet or IMS
  consumes the accepted cross-platform fixed dataset assets. Never rebuild graphs per
  platform or substitute a platform-local split/cache for a comparable claim. Exact
  identities are in the acceptance records linked by `platforms/README.md`.
- **IMS access boundary**: before any molecular-research-server command, read and
  obey `platforms/REMOTE_HANDOFF.md`; it restricts all path access, including
  read-only probes.

## Remote monitor handoff
The always-on server runs one local two-conversation loop for server-owned chains
only. Conversation A owns bounded 100K/500K interpretation, release validation and
authorized submission. Conversation B uses the configured monitor runtime and a fixed
30-minute heartbeat.
- B reads only the compact local binding, last observation and unresolved
  event/outbox (never the repository on each tick), and checks only the exact
  server-owned bound job. Healthy `QUEUED`/`RUNNING` is silent. A confirmed completion
  or actionable execution fault is persisted atomically as one idempotent event for the
  existing A.
- Uncertain API/status is `UNKNOWN`, not failure. B never selects candidates, edits
  code, contracts or scientific fields, retries, cancels or submits jobs, opens
  protected roles, or creates another A.
- A claims events idempotently, separates infrastructure from scientific outcomes,
  chooses zero or one justified covered action, then rebinds B or pauses or closes the
  chain.
- A submission timeout is `SUBMIT_UNKNOWN` and needs reconciliation before any retry;
  duplicate ticks never create duplicate successors.
- Desktop is independent: the server never monitors, adopts, continues or wakes
  desktop-owned jobs, and there is no default cross-machine live control plane.

## Branch governance
- **`master`**: minimal stable delivery; only validated, reviewable production content.
- **`molgap-server`**: server-side integration (protocols, Kaggle screens, runners,
  acceptance logic, compact evidence, decisions).
- **`molgap-desktop`**: independent desktop integration (full training, official
  evaluation, final submission). Never overwrite another machine's branch; the two
  integration branches exchange only reviewed, explicit commits.
- **`archive`**: rejected or inactive implementations and provenance.

Work on the owning branch; no long-lived branch per experiment. Use a short-lived
`work/<topic>` only for parallel, isolated or risky work, merge accepted work into
`molgap-server`, preserve rejected compact evidence in `archive`, verify remote
reachability, then delete it. Before branching or integrating, fetch, inspect the
worktree and preserve unrelated changes. Classify commits: `docs(...)` protocols and
decisions, `feat(...)` new capability, `fix(...)` repair that never disguises a
scientific-contract change, `ops(...)` packaging, submission, retrieval or handoff.
Large caches, checkpoints, models, logs and predictions stay in ignored local or
platform storage; git receives compact metrics, hashes, manifests, acceptance records
and decisions. A server experiment passes its confirmation gate before desktop full
training, and only a validated delivery candidate is promoted to `master`.

## Conventions
- Docs in English. One file answers one question. Do not write a fact twice; the
  second place is a link.
- Entry documents hold routing, hard constraints and live deltas only. Methods,
  metrics and logs go to conditionally read records. Decision records use dated
  historical language and never say `current`, `default`, `running` or `next`.
- Directories are named for a role or question, never a calendar phase (`NAMING.md`).
- Shared dependency environments: select checkout code with `platforms/local/run.py`;
  never repoint a shared editable installation or change a running historical
  environment. Editable installs are for a dedicated development venv only.

## Doc map
How it was done: the experiment's `decision.md` under `experiments/`. What ships:
`production/README.md`. Remote runs: `platforms/README.md`. Install, inference and
experiment infrastructure: `README.md`, `docs/operations/EXPERIMENT_QUICKSTART.md`,
`docs/operations/EXPERIMENT_ADDON_GUIDE.md`, `docs/operations/EXPERIMENT_CLI.md`.
