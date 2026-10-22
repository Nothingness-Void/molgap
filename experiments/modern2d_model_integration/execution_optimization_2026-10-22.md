# Execution optimization verification — 2026-10-22

## Decision and scope

Implementation `909f16a4c3076bc595c11586fde88298376874b7` added an explicit
CPU-prepared PPGT/ESA execution candidate and removed shared indexed-read and
profiler overhead. Packaging correction
`9da8e118a77cfde65d977c4b0a742508940d4759` bound its inventory pin to the
committed optimization-only source, excluding the concurrent worktree addition.
The affected generated-CPU behaviour passed its focused
checks. Native equivalence, warmed accelerator speedup, budget feasibility and
scientific training release were not established. No remote job was submitted.
The [machine verification record](execution_optimization_verification.json)
pins the retained local reports and separates the known snapshot failure.

The [operation guide](../../docs/operations/MODERN2D_EXECUTION_OPTIMIZATION.md)
owns reusable interfaces and qualification requirements. The implementation
reused the existing Kernel, Session, normalized-Gap update, cache, profiler and
atomic checkpoint owners; it introduced no training loop or platform client.
FP32/TF32-off, BS128, model configuration/parameters, sampler order, loss,
optimizer/schedule, evaluation frequency, roles and checkpoint cadence were
not changed. Learned positional/bond/stem projections were never cached.

## Exercised paths

- Both real family implementations ran through the shared CPU preview Session
  on generated topology: four optimizer steps,512 sample presentations, full
  synthetic development evaluation at two pass boundaries, and matched full
  state after atomic disk save/read and mid-pass continuation. PPGT used zero
  dropout/stochastic depth in this fixture so its batched-RRWP candidate, not
  the stochastic fallback, was exercised. The corrected three-case report
  superseded the earlier Session cases, rather than adding three independent
  successes. Selection/checkpoint callbacks retained synthetic state in the
  fixture; a complete scientific output manifest/RML finalizer was not tested
  by this Session check.
- PPGT batched features matched its per-graph feature oracle within declared
  tolerances; candidate outputs, all parameter gradients and three AdamW
  updates matched the existing bucketed reference within the existing strict
  tolerances. The stochastic preset retained graph-order output/RNG behaviour.
  This did not erase the separate graphwise-versus-bucketed update discrepancy.
- ESA tokens, outputs, all parameter gradients, three AdamW updates and RNG
  matched byte-for-byte on CPU, including residual/readout addons and atom-only
  inputs. Prepared dispatch retained substrate root hooks. Eval-mode addon
  container inheritance and mutable/replaced layout rejection were repaired
  before the final checks.
- Indexed batch fetch preserved non-contiguous member and duplicate request
  order, independent clones, source/role checks, shard mutation/size rejection,
  bounded handles and worker behaviour. Development reuse retained role/input
  fields and exact indexed member order; overflow streamed all rows and an
  incomplete pass never published a partial cache.
- Capture-only profiler wrappers/hooks detached after capture, including
  injected failures. Existing hooks, update counts and capture-off resource
  accounting survived. The profiler fixture observed a checkpoint call at a
  capture boundary and then two updates/256 presentations; that was fixture
  evidence, not a measurement of native checkpoint cadence.
- Static PPGT/ESA model import closures included the model-only context without
  training modules or dynamic imports. A prepared candidate source archive,
  cloud bootstrap and native worker were not executed.

## Results and limitations

Two configured Luna6 Max leaf workers executed the final focused scopes.
Model checks had145 passing cases, including the superseding three-case
Session rerun. IO/profile checks had107 passes,2 intended skips and1 failing
shared-snapshot check. The coordinator executed no tests.

The snapshot failure was not concealed or repaired by broad hash replacement:
the committed `family_training_runner.py` pin predated the already reviewed
disabled-EMA fix `3cdffb798`, and a concurrent unrelated test-support addition
changed the live `experiment_source_inventory.py` hash. Only the optimization's
two inventory additions were staged; the unrelated addition stayed in the
worktree. The six implementation-owned snapshot pins matched their staged
content. No full-repository compatibility pass was claimed.

Windows nested-fixture `git add` setup errors cleared with subprocess-scoped
`core.longpaths=true`; a short unique scratch retry passed the PPGT compilation
case and intentionally skipped the ESA exclusion. Neither global Git settings
nor production tests were weakened.

No real molecular cache, retained model checkpoint, protected role, GPU/DCU,
remote job or Desktop branch was accessed. Synthetic optimizer updates were
test activity, not a scientific training experiment. RML source evidence and
derived outputs were not rebuilt or relabeled. No Replay-Ready claim followed.
Independent checkpoint digest verification was retained because mutable-path
reuse did not justify removing integrity reads. CPU token expansion and device
transfer trade-offs required measurement on the target runtime; acceleration
was not inferred from code structure or CPU test counts.
