# Shared recovery reliability review — 2026-10-14

## Provenance and scope

Reviewed desktop repair `a01cda9f6caa12da876701c18b26a4a79e7ad14a`
against server base `ef116936230f78100d450bab06ed78faf6edf240`. All four
reliability gaps also existed in the server shared-family owners. This was a
scoped port, not a desktop merge, scientific intervention or remote release.
The server implementation/test commit is `298affc1`.
The existing [shared training integration](SHARED_TRAINING_SERVER_INTEGRATION.md)
and [core](FAMILY_TRAINING_CORE.md) retain their authority boundaries.

## Repairs and reuse

| Gap | Shared owner and correction |
|---|---|
| Deleted AdamW moments accepted | `family_training_session.py` acknowledges initialized parameter IDs, requires unconditional/base coverage and retains the conditional SSMA exception. |
| Selected checkpoint only checked for finite values | `family_training_outputs.py` also compares selected/resume tensor keys, shapes and dtypes, even after file hashes are rebound. |
| Failed checkpoint leaves selected/trace ahead | The same owner stages outputs, records a context/hash-bound journal, keeps the previous generation and publishes checkpoint last. Recovery validates all entries before any rollback. |
| Interrupted invocation omitted from cumulative cost | `family_training_runner.py` starts an invocation marker before input/GPU work, binds closed cost to checkpoint bytes and keeps interrupted/unbound cumulative cost unknown. |

Reused the existing atomic IO, canonical trace recorder, checkpoint translation,
source validation and output inspector. No new framework, platform submitter,
model, RML schema or acceptance bypass was introduced. The three desktop
regression files were retained; the server has a separate focused hook suite.

The server port preserved both local/cycle modes, live and EMA development
observations, eV training-metric conversion, per-pass wall/device semantics,
checkpoint identities, ten-pass transport hooks, actual-state resume qualification
and the sixty-pass scientific horizon. The bounded trace is staged against the
next checkpoint digest before either file is published. Transport hooks run only
after the journal has committed and cleared.

Metric prefixes transfer only new scalars and preserve sequential Python addition.
The prepared GPTrans backend omits the unused final shortest-path expansion and
rejects an unreviewed distance override. K1's private index/mask copies request
non-blocking transport on the owning stream. These are reviewed execution changes,
not a default precision/optimizer/sampler change or a measured speedup.

## Compatibility and operation limits

- New source packages require their own release/native qualification. No frozen
  source, recipe, old certificate or checkpoint hash was rewritten.
- Checkpoints lacking the new participation field use their historical source;
  no participation history is inferred or invented for them.
- One exclusive producer owns the output directory. While interrupted, retain
  the journal and `session_pending/` along with all visible outputs. Journal
  recovery is not a platform backup, live-reader transaction or power-loss proof.
- Unknown costs remain unknown. Process wall time is not allocated accelerator
  billing. Mechanical inspection is not scientific acceptance or Replay-Ready.
- Existing user/parallel experiment edits and RML-derived changes were excluded.

## Verification

Synthetic CPU fixtures only, using `.venv/Scripts/python.exe` (Torch2.7.1+cpu)
with CUDA/HIP/ROCR visibility set to `-1`. No real molecules, retained models,
protected roles, inference, remote platform or training allocation was consumed.

Luna independently ran:

```powershell
.venv\Scripts\python.exe -m pytest --noconftest tests/test_family_commit_safety.py tests/test_family_recovery_guards.py tests/test_family_speed_equivalence.py tests/test_family_training_session.py tests/test_family_training_runner.py tests/test_family_training_core.py
```

Result: **165 passed, 1 skipped**, 18.32s pytest time. The skip was the explicit
native K1 transport test, not a substituted CPU native qualification. Regressions
cover deleted/relabelled moments, conditional lazy state, rehashed malformed
selected tensors, journal/rename failure before/after commit, corrupt/wrong-context
evidence preserved unchanged, exact continuation, stale/interrupted cost and
operation equivalence. Pin-memory warnings were expected with no accelerator.

The initial server-specific run exposed a fixture using the default `reference`
builder for non-reference modes. Only the fixture was corrected to declare the
actual mode; no production gate was relaxed. Its final **11/11** checks passed,
covering both bounded modes, rollback/commit, exact state continuation, live/EMA
and timing/hash preservation, ten-pass transport ordering and distance-override
rejection.

The disjoint lifecycle/packaging/loader regression invocation passed **355 tests,
3 skipped**, in 268.42s:

```powershell
.venv\Scripts\python.exe -m pytest -q --noconftest --tb=short tests/test_experiment_family_workflow.py tests/test_experiment_resume.py tests/test_experiment_workflow_resume.py tests/test_experiment_package.py tests/test_experiment_preflight.py tests/test_experiment_workflow.py tests/test_family_training_recipe.py tests/test_family_training_loaders.py tests/test_gptrans_execution_backend.py tests/test_gptrans_cycle_pair.py tests/test_gptrans_cycle_inputs.py
```

The existing `tests/test_family_cycle_release.py` passed **14/14**, including
actual synthetic ten-pass transport/restore, in 6.24s. Together with the11 new
server cases and165 Luna cases, the non-overlapping accepted suites covered
**545 passed, 4 skipped, 0 failed**. The three lifecycle skips retain their owning
host-symlink/explicit-real-input guards; no real-input execution was substituted. Scoped
`compileall` and `git diff --check` also passed. No full-corpus RML rebuild,
historical migration check, native benchmark or remote operation was executed.

## Speed evidence disposition

Desktop's later bounded synthetic diagnostic (retained on its branch through
`2340fbe5cecae7650ddbd3f06130ae67f3b2ea3f`) established exact endpoint equality,
not a major speedup. Its K1 compute windows were slightly slower and its GPTrans
compute direction varied; journal protection added session boundary work.
That RTX5060/Torch2.7.1 diagnostic did not qualify T4/A100/DCU, Torch2.11, real
500K epochs, MAE or allocation billing. No such result was adopted as a server
qualification or used to relax durability. Its numerical records remain owned
by `experiments/family_session_execution_speed/` on desktop.
