# Reviewed shared training integration

## Provenance and scope

The server review used desktop tip
`bd25a038559abe3beccbc1256d3c766b3fb8859e`, the shared-core implementation
`d1b23b0a6ab6d7580f82ec88bdcd16081a188426`, and its reviewed prerequisite
helpers. This was an additive integration, not a merge or replacement of the
desktop branch. Desktop acceptance is retained separately in
[FAMILY_TRAINING_TEST_HANDOFF.md](FAMILY_TRAINING_TEST_HANDOFF.md).
The [shared core](FAMILY_TRAINING_CORE.md) owns APIs and module responsibilities.

The reviewed [logging/execution update](DESKTOP_LOGGING_EXECUTION_SERVER_REVIEW.md)
ports desktop `853ac0ed` / `48c3bdde` while retaining server-specific bounded
campaign modes, frozen jobs, profiling and source/role/continuation gates.

The later reviewed reliability integration from desktop `a01cda9f` is recorded
separately in [the server recovery review](SHARED_RECOVERY_SERVER_REVIEW.md).
It preserves the server's bounded same-run pair, trace and transport hooks;
desktop speed records do not qualify server/native performance.

Eight training owners cover typed recipes, selected fixed data, persistent
loaders, optimizer/loop, family kernels, resumable sessions, registered runners
and mechanical outputs. New interfaces are `neural_atom_k1/3` and
`gptrans_t/2`; interface versions are not scientific contracts.

## Server compatibility decisions

- Retain server registries/addons, graph-screen hooks, strict launch validation,
  retention/allocation/resume adapters, and prospective/reference/evidence gates.
- New interfaces use the shared accelerated kernels. Historical execution
  comparisons retain their original/optimized backend choices; the desktop
  accelerated-only policy was not applied retroactively.
- Retain legacy K1 scale/full and GPTrans screen loops. K1's screen loss delegates
  the same update equation to the common optimizer primitive. Historical jobs
  consume their matching immutable payloads; no source pins were rewritten.
- Import the desktop GPTrans full owner as library code only. Its presence gives
  server no full-training, official evaluation or desktop-job custody authority.
- Keep the server EMA evaluator signature through a shared-kernel adaptation;
  the outer reused-weight view supplies the selected weights once.
- New shared initialization release checks construct a CPU model in the isolated
  packaged-source subprocess and check transported tensor keys/shape/dtype.
  SHA-valid but incompatible tensors are rejected; no forward occurs there.
- Keep explicit optional target-encoding binding and one predictions inspector
  for legacy and shared-session outputs.
- Preserve final-only outputs without fake development predictions. Mechanical
  success leaves scientific acceptance and replay readiness NOT_EVALUATED.
- No experiment, RML trajectory, canonical/derived evidence or monitor was
  adopted from desktop.

## Capability boundaries

| Stage | Covered behavior | Limit |
|---|---|---|
| Variant | Static model/addon hooks; shared recipe/session/loader/EMA/IO | Unsupported combinations need reviewed registration, not arbitrary JSON imports |
| Local release | Packaged import closure, recipe SHA, initialization transport/schema, staged bindings | Not a real-shard forward/backward smoke or a native qualification |
| CPU diagnostics | Existing owning diagnostic modes where registered | Do not assume every new family supports real-shard model smoke |
| Native preflight | Shared runner repeat plus serialized optimizer-inclusive resume proof | Requires separately authorized accelerator and accepted training data |
| Submission | Existing owning platform adapter/skill and actual receipt | Preparation is not a submitter; timeout requires reconciliation |
| Output | Mechanical checkpoint/trace/prediction/counter/role/cost inspection | Cannot reconstruct unrecorded evidence or infer causal validity |
| RML | Existing terminal translation and owning finalization/replay gates | Missing canonical reference/runtime/role/cost evidence remains blocked |

A CPU smoke reduces shape/import/backward/serialization mistakes but cannot
promise a fixed 15–30 seconds or 100% GPU success. Device kernels, memory,
dependency/runtime differences, mounts, permissions and remote persistence still
need native checks. No smoke authorizes a new scientific recipe.

`terminal --execute` consumes a complete supported terminal descriptor; it is
not a universal raw-output analyzer. Family inspection/acceptance constructs the
applicable inputs first. SHA-bound evidence is tamper-evident, not literally
immutable storage. A measured worker wall time is not automatically a complete
platform allocation bill. Replay-Ready is granted only by its actual validators.

## Verification

CPU-only synthetic graphs/tensors were used; no retained molecular assets,
protected roles, remote platforms or production inference were consumed.

- Shared core/data/recipe/loaders/kernel/session/runner: **194 passed**.
- Luna compatibility suite (CLI, Spec, packaging, family output, execution
  backend/hotpath/tuning): **402 passed, 1 skipped**.
- Release/workflow/resume/GPTrans/preflight suite: **157 passed, 2 skipped**;
  two additional new isolation-fixture cases initially failed because the fixture
  did not stub the host-import-origin check. Only the fixture was corrected;
  production isolation was not relaxed. The corrected new integration tests
  passed **5/5**; the final combined integration/family-output rerun passed
  **66/66**.
- Prior intermediate failures exposed missing prerequisite helpers (frozen-state
  reader, scalar metric reduction, EMA default and full-schedule source helper);
  these were resolved before the passing shared-core run.
- Missing upstream standalone execution/source-inventory test filenames were
  omitted explicitly; owning server suites were used instead.
- Counts describe separate invocations and must not be summed when suites overlap.

No real molecular training, production inference, remote submission, GPU/DCU
qualification, speed/quality claim or automatic Replay-Ready claim was made.
