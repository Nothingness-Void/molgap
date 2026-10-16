# Research workflow reform — cumulative migration record

Date: 2026-10-16. Implementation baseline: server
`1179054750fd8ff500548ad6b3ee3eece94bb786`.
Input: `MOLGAP_REFORM_PLAN.md`, SHA256
`e617a4f32da311942d48a8ca1b11a26ca125c351509410aaba72e321d85a174d`.

## Acceptance scope

Status: **INFRA_CPU_ACCEPTED for the delivered server subset**;
the entire A–G reform is **NOT_COMPLETE**.

The bounded server implementation is CPU-qualified, not a completed deployment
of every reform batch. No remote POST, credential inspection, GPU allocation,
real-data run, graph rebuild, official/protected-role read or scientific ranking
was performed. V5, historical scientific decisions and canonical RML are unchanged.
The [quickstart](EXPERIMENT_QUICKSTART.md) owns the normal workflow; this is its
one cumulative implementation/evidence record, not another operating protocol.

| Batch | Delivered | Qualification / remainder |
|---|---|---|
| A | Default pytest collects `tests`; CLI fixtures load on demand; selected checkout runs under `-I`, including owning platform scripts | CPU accepted; shared `.pth` and running environments were not changed |
| B | Machine-local external data/artifact/scratch roots; stable run/attempt paths; streaming verified no-clobber copy | New path accepted; one old archive copied and recovered by hash; no bulk historical migration/GC |
| C | One catalog and read-only projections; shared K1/3 and GPTrans/2 scientific inputs generate recipes, Spec, workflow, source and prospective bindings | Bounded CPU scenarios accepted; arbitrary recipe/new-family compilation is not implemented |
| D | Fixed runtime distribution, bulk Git, atomic complete archive cache; V2 explicitly distinguishes generated JSON from committed source | CPU package accepted with V1 compatibility; no native qualification |
| E | Existing Kaggle `--experiment` entry; confirmed pre-publication failure recovery; same original submission journal; read-only run-ID locator | CPU/fake-API accepted; authenticated latency and real receipts not measured |
| F | One normal guide, single portable source inventory and explicit commits | Server patch supplied; Desktop is NOT_APPLIED and requires its own review/tests |
| G | Public model/recipe/session/checkpoint paths preserved | Deferred until cross-owner integration; no mass source moves or compatibility constants deleted |

The remaining C/F/G work is real work, not declared complete by the CPU status.
Reference-input V1 remains compatible. Scientific-input V1 compiles bounded
optimizer/catalog-addon changes through existing scientific builders and gates;
it does not make arbitrary recipe/model changes executable or admissible.

## Reuse and replaced defaults

Reused `common.spec`, canonical fingerprints and IO, `FamilyTrainingCore`,
Recipe/Session/Logger/Profiler, existing fixed-data and cross-scale validators,
prospective/RML planning, source-package V1, launch/acceptance/recovery owners,
Kaggle preparation and submission journal. No replacement trainer, optimizer,
sampler, EMA implementation, RML database or platform submitter was introduced.

- Removed eager six-CLI imports from root conftest; dependent tests request their
  existing loader fixture explicitly.
- Removed automatic repository document-length assertions. Optional legacy
  budget CLI behavior remains callable, not an ordinary acceptance gate.
- Removed manually duplicated ordinary addon/family execution tables. The
  catalog supplies both parsing metadata and executable descriptors. Legacy
  `TrainingAdapter`/`TrainingAddon` APIs remain read-only projections.
- New editable preparations derive tracked runtime Python automatically; old
  selected-source/AST APIs remain for compatibility and development diagnosis.
- Commit verification now uses one tree read and one batch object read. The
  source archive cache never reuses another Spec/package/release/receipt.
- Removed the addon guide's duplicate handoff sequence; detailed documents are
  API/compatibility references to one normal quickstart.

## Local storage and retained originals

The user rejected C-drive artifact storage. The machine-local profile is
`E:/MolGap/machine-profile.json`, with roots `E:/MolGap/data`,
`E:/MolGap/artifacts`, and `E:/MolGap/scratch`. The interpreter remains the
existing project venv on C; no environment relocation was attempted. Data-root
configuration does not rewrite an accepted dataset manifest or relocate a cache.

One existing source archive was copied, stream-hashed and rechecked at:

```text
E:/MolGap/artifacts/source-packages/5bb2057ddf9f9691780f8572bfb57c50f8975dec8a7e5a733988ea169e80f57c/source.tar.gz
```

It contains 9,757 bytes and SHA256
`5bb2057ddf9f9691780f8572bfb57c50f8975dec8a7e5a733988ea169e80f57c`.
The source under the old `platforms/_records/local/modern2d_review_hardening/archives`
location was retained. No original, worktree, ignored payload or branch was deleted.
Historical manifests were not rewritten. This is a physical locator supplement,
not a new source/scientific identity.

Only registered worktree metadata was inventoried; ignored/unique-payload
censuses of all old worktrees remain unperformed. Five worktrees were registered
at observation, including the primary server, archive, two detached checkouts and
the reused reform worktree. No cleanup eligibility is inferred from that count.

## Actual CPU verification

Runtime: project Python 3.11.15, Torch 2.7.1+cpu, PyG 2.6.1; CUDA hidden and CPU
threads bounded. Raw test reports are retained under
`E:/MolGap/artifacts/reform`; pytest scratch directories are on E and retained.
Counts below overlap and must not be added as independent experiments.

| Raw JUnit name | Observed result | Scope |
|---|---|---|
| `catalog-package.xml` | 15 passed, 19.52s | Catalog equivalence, source safety/cache and A/B fixtures |
| `package-legacy-recheck.xml` | 128 passed, 1 skipped, 74.62s | Existing packaging/common/recipe compatibility; native symlink privilege unavailable |
| `entry-journal.xml` | 84 passed, 36.31s | New inputs, original fake-API journal, family output inspection |
| `distribution.xml` | 1 passed, 33.42s | Actual 426-file fixed distribution in an isolated child; existing core/state/session tests run inside it |
| `core-regression.xml` | 268 passed, 18.85s | Core, Recipe, Session, state, common API, catalog, storage, registry and input compatibility |
| `lifecycle-regression.xml` | 305 passed, 1 failed, 1 skipped, 134.24s | One stale CLI command-set assertion; no package/journal production failure |
| `gates-cli-recheck.xml` | 233 passed, 54.20s | Corrected CLI guard, new workflow, Modern2D CPU restrictions and cross-scale no-auto-decision rules |
| `final-front-door.xml` | 107 passed, 27.43s | Final run-ID locator, local CLI, checkout/platform launcher, portable inventory and fake-API entry integration |

The original pre-storage-choice runtime report was also copied to E as
`initial-runtime.xml`, with its original retained. The isolated child's raw
JUnit was copied to `isolated_cpu_state.xml`, and its observed timings/module
path to `distribution_observation.json`; each copy used the same streaming
no-clobber helper. All reports preserve failed attempts rather than replacing
them with later green results.

The isolated child's copied report contains **116 passed**. Report byte pins:

| Report | SHA256 |
|---|---|
| `catalog-package.xml` | `7100a3b59cb1bbbf625701d4ea9613be2419075d350c2e9edb99a9bb05da1a65` |
| `core-regression.xml` | `597bd76c9141a1408d2dd31a983a2fe9170b307b1dad3c9c519fa27bed882bc3` |
| `distribution.xml` | `942300e0b25d92e02b1f8d0466826297d6fc3c4ec5fdf6be8a8353e4cf183f03` |
| `entry-journal.xml` | `c060f31eaaa4e1df3cadeee4acc09593b32181e365eb40d9e1a28c825ab05a0a` |
| `final-front-door.xml` | `ec065defffe2de4880e4da24791ca3bcb88137b80bd956447fab53a7b6986edc` |
| `gates-cli-recheck.xml` | `f81406a33bf741a81adce1bddf2744e19bf35db241f4843cdc275e84ac2cf6c5` |
| `initial-runtime.xml` | `d8852c0adcc8c07d02111e6ff5009ff4bb5535053b9ef059695b8ce506bd542c` |
| `isolated_cpu_state.xml` | `97ba92585f1b043faffafbae058b46ead522ac6c0b1b201cbcaffdf5ecb0e1da` |
| `lifecycle-regression.xml` | `be9df2f9cce609c79736d867ed5252e128a34cba56b68c3ef185ed4faf2f5a42` |
| `package-legacy-recheck.xml` | `3553a77d3d9b177948f67a9fde0a471c58213d1a65f7607ce0ece121e226231d` |
| `package-legacy.xml` | `33bdaed7843859ce7b0978b813470a8cac622710eaa944e872d4d30c5098043b` |

The earlier `package-legacy.xml` retained 73 setup errors because the fresh E
scratch parent did not yet exist (56 tests passed). Creating that explicit parent
and rerunning produced `package-legacy-recheck.xml`; raw failure evidence was
not deleted. The command-set regression was updated to enumerate the new local
commands, without relaxing the no-network/no-dynamic-dispatch checks.

The isolated distribution test verified every imported MolGap module came from
the extracted E-drive source, despite a deliberately incorrect inherited
`PYTHONPATH`. It exercised actual random K1 then GPTrans single-batch addon
forward/backward/state roundtrip via the existing sandbox. It copied the existing
CPU core/session/state test drivers, not a training implementation. Continuous
and interrupted/resumed optimizer, scheduler, RNG, cursor, EMA and selection
states are checked by those owners; synthetic BN/Clean calibration has its own
rollback/roundtrip checks. Scaler is inapplicable to the retained FP32 CPU path.
Twelve scientific model/training-owner source files matched the pre-reform
baseline byte-for-byte after LF normalization.

## Measured performance, not targets claimed as results

One actual 426-file distribution archive contained 1,301,148 compressed bytes.
At source commit `f4c9afdd6fb89eaa361d8046dcf0c0670684a591`:

| Measurement | Raw seconds |
|---|---:|
| Cold source-package/cache publication | 7.4543983 |
| Warm source-package creation using the same archive | 3.8344914 |
| Independent CPU child, actual family sandboxes and state/session tests | 15.2916196 |

The cold/warm packages shared archive bytes but had different Spec/package
identities. Another 80-file fixture verified repository checking starts exactly
three Git processes, independent of source-file count. Tampering hidden by
`assume-unchanged`, cache corruption and scoped credential names are rejected.
The cache key includes builder version, exact source commit and all source
payload hashes. A changed source/record commit invalidates it conservatively.

These are single raw samples, not p50/p95 estimates. Full local preparation
against a live authorized declaration, upload/source-ready time, actual receipt
latency and queue time were not measured. The 90-second/5-minute targets are
therefore NOT_ESTABLISHED. No seed-43/44 experiment was created to test caching.

## Portable commits, rollback and owner boundaries

Each batch has a separate local commit on `codex/reform`, based on the server
baseline above:

- `35892fab`: A, lazy collection and isolated checkout launcher.
- `d4a7a51c`: B, explicit machine roots and verified streaming copy.
- `744a128a`: C, single catalog projection.
- `fe9baca0`: D, fixed inventory, bulk Git and archive cache.
- `8fd264b2`: E, reference freeze and original platform entry.
- `027164cf`, `0670dd42`: child scratch and platform-script source selection.
- `f4c9afdd`: actual isolated distribution verification.
- `a72ca779`: original preparation/journal fixtures through the new entry.
- `84f776cf`: read-only run-ID locator.

The [portable inventory](REFORM_SHARED_FILES.json) reuses the existing
`molgap-source-files-v1` format and pins 49 shared files after review hardening. The original
[historical inventory](SHARED_TRAINING_CORE_FILES.json) remains unchanged at
`f6ec2ad02ea9105c75217bd7e9e0cdb70331229d`. Catalog/platform glue must be reviewed
against Desktop's own capabilities rather than copied as machine policy.
Desktop tip `1ebb2e45d78d623374b50fb5df40b7af8e51ce59` was inspected read-only;
its kernel, loaders, recipe and session differ from server, so file-name matching
is not a portability certificate. No Desktop branch was changed.

Rollback is per implementation commit/new-entry selection. Frozen historical
runs keep their original source/dependencies/artifacts; no hot update is applied.
Submission ambiguity always uses the original journal, never a new run ID/slug.
No push or remote branch update is authorized by this reform.

## Remaining work

1. Scientific compilation is bounded to shared K1/3 and GPTrans/2, catalog
   addons, learning rate and weight decay. Other typed recipe/input/family
   changes require their existing owning builders and focused acceptance.
   Frozen screen budget, data, batch, seed, sampler and selection cannot be
   silently changed through this input. No baseline is retrained automatically.
2. Desktop must apply/review the portable mechanisms with its own catalog glue,
   preserve its unique recovery/execution semantics and pass the same inventory
   and targeted compatibility tests. Native/runtime qualification stays separate.
3. G source regrouping and historical constants migration follow that cross-owner
   acceptance; public checkpoint module paths remain intact in this delivery.
4. Old unique/ignored artifact census and any further copy-only migration remain
   separate, bounded work. No deletion or generic TTL/GC service was implemented.
5. Real platform latency/receipt samples and any native test require separately
   applicable authority; this CPU task does not release them.
6. V1 archives contain frozen recipe records as well as code. Changed recipe
   bytes or commits conservatively miss the cache, even if model Python is
   unchanged. Logical Spec-only reuse is verified; seed-only reuse across
   changed recipe records is NOT_ESTABLISHED. A separate runtime/config package
   layer was not introduced. V2 adds explicit generated-record provenance, not
   removal of recipe byte binding or seed-only reuse claims.

## Primary server integration verification

After a read-only remote fetch, the primary `molgap-server` checkout was
fast-forwarded from `1179054750fd8ff500548ad6b3ee3eece94bb786` to
`5985e26b8305ac7c65ccff69d2c24b9e63557742`. All 14 unrelated dirty tracked
files matched their original SHA256 pins before and after integration, including
`tests/test_research_memory.py`. They were not staged or included in the patch.
Unrelated untracked/ignored evidence was retained, not adopted or cleaned.

The final primary suite (`primary-acceptance.xml`) passed **41 tests**, with
zero failures/errors, in 61.098s (JUnit suite timing). Its byte pin is
`2a31c213e309147d2b83072e8653e57510b3149ec4874e9a8a58cd4922236189` under the same E-drive report root. It reran
the actual committed fixed distribution and isolated family/state/session proof,
as well as input/run-ID, cache, catalog, path and platform-journal integration.
The primary distribution sample used 426 source files;
cold/warm packaging took 6.068310/3.710259s and the isolated CPU
child took 15.619401s. These remain raw single samples, not platform latency.

The 14 changed active Markdown documents passed targeted navigation audits:
317 pointers before integration and 318 after, eight anchors, zero issues,
no line-count gate. This does not claim
a clean-checkout audit of every historical/untracked evidence link. Twelve
scientific execution owners, both durable V5 contract texts and the historical
shared inventory had no diff from the implementation baseline.

No push, Desktop integration or worktree removal followed. The local reform
branch/worktree was retained for recovery; remote reachability of the new commits
is not claimed. The existing RML derived/user changes remain the user's own.

## Review hardening of 13eed22e (2026-10-16–17)

Review input SHA256:
`edc25ec00a9a15afc9d48bf57e5f589096f016c46f4ad1c59080ceb6f6176e1f`.
The supplied review was implemented without a new trainer, RML database,
platform submitter or approval system. Its core scenario, not test count alone,
is the acceptance target:

- One committed human learning-rate edit generated a new recipe, Spec,
  workflow, purpose-scoped comparison and prospective inputs using existing
  owners. K1 and GPTrans were tested independently in isolated E-drive fixture
  repositories with actual model initialization schemas and actual package,
  staging, release and RML checks. These were CPU synthetic infrastructure
  fixtures, not scientific experiments or accepted native references.
- An injected staging IO failure preceded RML publication. The same front door
  retained its complete failed report, regenerated staging with identical frozen
  input/Spec identity, published one prospective record and reused its completed
  preparation on the third call. No SDK/remote request occurred. Published,
  unknown, active PREPARING and operation-journal states remain non-retryable
  here; the original platform journal owns remote reconciliation.
- Source caches are validated in temporary directories and published by one
  directory rename. Legacy partial cache objects are retained and rebuilt from
  their bound source; corrupt complete objects still fail closed. V1 packages
  remain unchanged. V2 retains six sidecars while explicitly binding bounded
  generated JSON and the committed human input; untracked Python is forbidden.
- Root editable-install ambiguity was removed. The indexed CPU-only prototype
  hashes once per loaded shard instance across 100 repeated reads, invalidates
  changed file identities and retains its real-data prohibition. This metadata
  cache is not a malicious-filesystem or native loading certificate.
- Permanent old-source/old-capability locks were removed from ordinary tests;
  prior one-time byte proofs remain in this record. Current package hashes,
  state/gradient/recovery behavior and source isolation remain tested.
- Local errors expose finite safe field/reason messages; arbitrary exception
  strings and SDK/HTTP request details are not echoed by the front door.
- Baseline data declarations must match the Spec identity bound by the retained
  reference context. Changed membership cannot borrow an unchanged recipe and
  prediction manifest to manufacture a compatible comparison. This binds the
  historical reference input, not the candidate's platform/run identity.

| Raw JUnit name | Observed result | Scope |
|---|---|---|
| `review-repairs-2.xml` | 85 passed, 1 skipped, 78.014s | Recovery/input/cache and old package compatibility |
| `review-regression.xml` | 141 passed, 242.061s | Existing workflow, acceptance, comparison, catalog, indexed CPU and fake-journal fixtures |
| `review-final.xml` | 106 passed, 1 skipped, 101.765s | Final R1–R6, generated-package safety, V1 compatibility and original journal front door |
| `committed-review.xml` | 169 passed, 146.429s | e4a11e04: actual K1/GPTrans config/recovery and two-arm compilation; committed isolated distribution plus Core/Recipe/Session/state tests |
| `data-context.xml` | 2 passed, 14 deselected, 40.23s | eda23d16: changed baseline membership rejected in both families before publication |

These overlapping suites are not independent scientific experiments. The skip
is the existing Windows link-privilege limitation. Earlier fixture failures were
retained on E. `scientific-final.xml` had 32 passes and one instrumentation
failure: counting started after the constructor had already verified/opened the
shard. Counting before construction passed in `review-final.xml` (one hash for
construction plus 100 reads, then one additional hash on each identity change).
No failed infrastructure test was recorded as a model failure or hidden by xfail.

The committed distribution contained 426 source files (1,303,554 archive bytes).
Its cold/warm package samples were 5.434930/4.127232s; isolated CPU checking took
15.772613s. These are single local samples, not authenticated preparation/upload
latency. The actual compiler scenario used original release/RML checks with
synthetic artifacts; fake-API journal tests retained their explicit qualification
stubs. Neither proves an actual platform execution or scientific improvement.

Portable implementation commits: `e4a11e04` (compiler and R1–R6) and `eda23d16`
(baseline data/context binding). Applying these mechanisms to Desktop still needs
review of its own catalog and runtime semantics, not a whole-branch copy.

Raw review regressions and the final scenario report are stored under the
existing E-drive report root. No historical V4/V5 decision, durable scientific
contract, existing RML record or primary user change was edited. Desktop/native
application and end-to-end live platform latency remain unqualified. Overall
A–G reform completion is still NOT_COMPLETE.
