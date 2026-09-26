# Shared experiment infrastructure: server integration review

## Scope and source

On 2026-09-27, the `molgap-server` checkout was reviewed after these integrations:

- `a3ed075c959ea0c1509bad06fb9e9e10f503cbbb`: reusable EdgeState construction and
  training primitives.
- `09a827239870890dc2c4235464861ac7bb46d818`: shared local experiment CLI,
  runner, packaging, prospective planning, launch receipts and terminal wiring.

The review covered local interfaces, dependency presence, source/family mapping,
synthetic tests, documentation navigation and compatibility with retained RML.
It was not a real-data, accelerator, training or remote-platform qualification.
No scientific contract, model implementation or historical result was changed.

## Findings

1. The previously missing `experiment_runner` dependency and the local workflow
   components were present. Relative-import inspection found no missing module
   among the reviewed experiment-core modules. CLI help exposed seven commands:
   `validate-spec`, `package`, `preflight`, `run-diagnostic`, `launch-receipt`,
   `terminal`, and `plan-prospective`.
2. Registration remained weaker than executable compatibility. The legacy K1
   addon declaration selected a factory mode absent from the retained server
   PairToken implementation. The [K1 adapter document](K1_ADAPTER.md) owns the
   exact mapping and migration boundary. That addon was not qualified to run;
   schema validation and synthetic tests did not remove the mismatch.
3. K1/EdgeState real-shard preflight was explicitly unsupported by the shared
   core. The EdgeState training primitives were not a complete trainer. The
   [addon guide](EXPERIMENT_ADDON_GUIDE.md) owns the division between shared
   components, experiment training integration and platform submission.
4. Documentation contained machine-specific checkout paths, an absent K1 guide
   and pre-verification handoff wording. The documentation revision replaced
   those with checkout-relative commands, explicit capability boundaries and
   this dated verification record. Desktop-only examples were labeled as
   contextual references, not runnable server instructions.

## Executed verification

All Python commands used `.venv\Scripts\python.exe` with this checkout's `src`
on `PYTHONPATH`. Pytest used `--noconftest` to avoid unrelated repository-wide
CLI/ML imports; this was a targeted suite, not a full repository test claim.

| Suite / selection | Passed | Skipped | Deselected |
|---|---:|---:|---:|
| Spec, CLI, package, launch, prospective, terminal, RML batch planning, runner, preflight (`not real_dual_arm`) | 809 | 7 | 2 |
| EdgeState adapter, excluding real factory/forward construction cases | 14 | 0 | 6 |
| EdgeState core (`batch_contract or sampler_cursor or training_stats`) | 9 | 0 | 5 |
| V5 common, server contract and comparison readiness | 47 | 0 | 0 |
| Total | 879 | 7 | 13 |

The seven skips were unavailable Windows symlink cases. The two real dual-arm
preflight cases and eleven real-model adapter/training-core cases were
deliberately deselected. Metadata subprocesses, dummy factories and synthetic
tensors were used; no real model training or inference was executed.

Reproduction of the shared-core selection:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
.\.venv\Scripts\python.exe -m pytest --noconftest -q tests/test_experiment_spec.py tests/test_experiment_cli.py tests/test_experiment_package.py tests/test_experiment_launch.py tests/test_experiment_prospective.py tests/test_experiment_terminal.py tests/test_research_memory_plan_batch.py tests/test_experiment_runner.py tests/test_experiment_preflight.py -k 'not real_dual_arm'
.\.venv\Scripts\python.exe -m pytest --noconftest -q tests/test_edge_state_adapter.py -k 'not family_and_variants_dispatch_to_existing_factories and not depth_only_model_constructs_and_forwards and not runner_constructs_depth_variant'
.\.venv\Scripts\python.exe -m pytest --noconftest -q tests/test_edge_state_training_core.py -k 'batch_contract or sampler_cursor or training_stats'
.\.venv\Scripts\python.exe -m pytest --noconftest -q tests/test_v5_common.py tests/test_v5_server_contract.py tests/test_comparison_readiness.py
```

`molgap.research_memory validate` and `check --frozen` passed against the retained
corpus. No derived file or canonical evidence was regenerated or rewritten.
The user's uncommitted `tests/test_research_memory.py` was neither edited nor
included in the targeted test selections.

## Release boundary

The verified local identity/package/receipt/RML capabilities were reusable;
they did not imply that every registered addon could train. A real experiment
still required its frozen source, supported factory, accepted data, qualified
runtime, release gate, durable trainer and owning platform adapter. A same-run
reference binding did not authorize redundant baseline training or promote
historical partial evidence.

No remote job, protected-role access, training, inference, model promotion,
desktop modification or desktop job adoption occurred in this review.
