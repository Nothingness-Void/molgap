# Shared experiment infrastructure: server integration review

This record contains two dated review stages. The
[desktop completion](#desktop-completion-review) supersedes the initial
missing-factory and unsupported-loader findings below; historical verification
scope is retained rather than rewritten.

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

## Desktop completion review

On 2026-09-27, the user authorized copying missing reusable components from
`molgap-desktop`. The reviewed donor was
`a47b945fc50a904a3d97949fbd8bdb478a10d73a`; no branch-wide merge was performed.

| Component | Integration |
|---|---|
| Selected topology-shard preflight | Exact donor copies of `experiment_preflight.py`, `pcqm_topology.py`, `edge_state_training_core.py`, their three test files and `EDGE_STATE_ADAPTER.md` |
| K1 value-decoupled model | Exact donor `k1_pair_token.py` contents under `k1_pair_value.py`; adapter routing and copied test import adjusted, historical server file unchanged |
| GPTrans adapter coverage | Exact donor `tests/test_gptrans_adapter.py`; existing implementation already matched |
| RML public/CLI entry points | Exact donor `research_memory/__init__.py` and `cli.py`, exposing existing terminal trace wiring and optional portable checks |
| Same-run peer validation | Additive donor pointer/peer checks only; retained server prelaunch recomputation and ownership restrictions |
| Additional synthetic coverage | Copied terminal/paired replay tests with fixture owner changed to server; copied portable-closure and runtime/bundle cases without desktop-corpus or legacy submission dependencies |

Ten same-path files matched donor Git blobs exactly. The renamed K1 model also
matched its donor blob, and the old server model matched its pre-integration
blob; both are regression-guarded. The K1 guide owns precise model provenance.
No Spec identity, old scientific threshold, training recipe, frozen historical
implementation, production registry, server/desktop topology or canonical RML
record was rewritten. Desktop full/official-evaluation runners, old V4
submission/audit stack and machine-owned operational state were not copied.

The copied preflight verifies selected topology inputs only. It does not turn
the K1/EdgeState construction family into an end-to-end training launcher or
grant model-smoke, protected-role, full-scale or remote resource authorization.

### Follow-up verification

| Suite / selection | Passed | Skipped | Deselected |
|---|---:|---:|---:|
| Shared experiment suites plus topology and portable-closure tests (`not real_dual_arm`) | 827 | 7 | 2 |
| K1/GPTrans adapters and integration guards, excluding real-model construction/forward cases | 102 | 0 | 6 |
| EdgeState core (`batch_contract or sampler_cursor or training_stats`) | 9 | 0 | 6 |
| EdgeState adapter, excluding real-model cases | 14 | 0 | 6 |
| V5 common, server contract and comparison readiness | 47 | 0 | 0 |
| Terminal trace closure, same-run replay and shared runtime compatibility | 30 | 0 | 0 |
| Total | 1,029 | 7 | 20 |

The skipped tests required Windows symlink capability. Deselected tests were
real-model/real-input checks. Executed tests used synthetic graphs, dummy
factories and local temporary repositories; no real model forward/training or
protected molecular role was executed. `--noconftest` and the project virtualenv
were used as in the first review.

The retained corpus passed `RML validate` and `check --frozen`, without rebuild
or derived/canonical evidence edits. The new stronger `--portable` check found
one pre-existing closure path absent from Git HEAD:

```text
platforms/_records/kaggle/training/pcqm_k1_v4_reference_s42_v2/pcqm_k1_v4_reference/neural_atom_k1_v4/trace.json
```

Before commit, that check also correctly rejected the edited RML runtime files.
The historical trace-path gap was not papered over by changing evidence or
weakening checks. Ordinary frozen-index validation is not a claim that the
entire historical artifact closure is portable. Retaining or migrating that
source trace is separate evidence maintenance, not a reason to retrain K1.

The existing uncommitted `tests/test_research_memory.py` was preserved byte for
byte and excluded from the integration commit. No desktop branch, remote job,
monitor, protected role or training/inference run was changed.
