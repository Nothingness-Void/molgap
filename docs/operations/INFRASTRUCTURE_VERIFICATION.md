# Infrastructure verification

This page owns test selection, the claims evidence supports, and test growth.
It is not a per-fix test log. Experiment evidence stays with the named run;
past failure audits remain in the [historical audit](INFRASTRUCTURE_AUDIT_20261020.md).

## What a successful check means

| Evidence | Establishes | Does not establish |
|---|---|---|
| Focused unit check | One algorithm or interface invariant | A complete training or submission flow |
| Local workflow journey | The real owners exercised by that fixture, with its stated substitutes | Execution of the new candidate's frozen package or native environment |
| Actual package release check | Identity, dependencies and inputs checked by the selected release mode | Unexecuted model/trainer paths, GPU behaviour or training completion |
| Exact native preflight | Its retained source/recipe/data/runtime and observed updates | Later pass/chunk boundaries it never reached, or scientific quality |

A fake API is appropriate for offline submission tests. Replacing package
verification, the training kernel, serialization or the output sink means those
parts were not tested. A Linear/Dropout kernel is useful for Session mechanics;
it is not K1, GPTrans, PPGT or ESA execution evidence.

## Select an affected workflow

Use the selected checkout and dependency interpreter. No machine profile is
needed for CPU tests when that interpreter is already selected:

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python platforms/local/run.py --module molgap.local_testing -- --list
& $python platforms/local/run.py --module molgap.local_testing -- --scope train --scope resume -- -q
```

[local_testing.py](../../src/molgap/local_testing.py) owns the fixed pytest
nodeids. New files and unrelated tests in the same file do not enter a scope.
Several scopes can share a journey; one invocation deduplicates it. There is
no implicit full-suite run. Use direct pytest for an affected lower-level
invariant; full-suite maintenance is an explicit separate operation.

| Scope | Local behaviour selected |
|---|---|
| `runtime` | Checkout isolation and registered K1/GPTrans one-batch sandbox |
| `package` | Package binding and imports from extracted source |
| `prepare` | K1/3 and GPTrans/2 scientific-input compilation and frozen recovery |
| `train` | Real K1/GPTrans addons and PPGT/3/ESA/4 kernels, loaders, output publication and saved-state continuation on generated CPU graphs |
| `submit` | Real local package/release/receipt/retrieval owners with fake network and a synthetic trainer |
| `accept` | Real synthetic Session outputs and role observations through reference-first finalization and actual replay admission |
| `resume` | Real-model continuation plus the ten-pass chunk transport and concurrent accounting snapshot |

These fixtures use bounded synthetic data and model configurations. They do
not execute the cloud bootstrap, decode a complete real shard, or qualify a
new candidate archive. Source-closure import checks also do not execute the
worker's training entry. `-k`, skips or other selection overrides must be
reported as partial execution, not a completed workflow check.

## Keep the suite bounded

1. Reproduce a failure through the nearest existing workflow first. Check
   whether it was omitted from selection or replaced by a stub before adding
   anything. Fix the production owner and strengthen that workflow.
2. Reuse one canonical fixture and parameter table for a behaviour. Merge or
   remove superseded local checks when the workflow covers the same fact.
   An incident does not automatically justify a new test function or file.
3. Add a separate test only for a distinct supported behaviour, numerical
   invariant or external IO boundary. Arbitrary wrong types/private-call
   permutations and assertions of source strings are not substitutes for
   exercising the supported agent workflow. Do not expand default scopes for them.
   Routine experiment configuration changes add no tests. A new model or addon
   extends its existing model/ABI journey; do not copy the submission or
   acceptance suite for each experiment.
4. Replace only external boundaries when claiming integration: network,
   accelerator observation, clock, or an injected IO interruption. Keep the
   actual loader/kernel/optimizer/callback/checkpoint owners in a training journey.
5. Assert observable outcomes with an independent expectation: exact updates
   and samples, pass/chunk boundaries, selected artifact/trace alignment, and
   continuous versus serialized-resumed state. Calling the same helper to
   produce and verify a counter is not independent evidence.

Runtime validation still protects actual data, artifact identity and external
authority. Keep those owning checks; do not add another certificate, registry,
mandatory audit document, or a validator solely to satisfy a new test.

## Report the exercised path

Lead with the behaviour observed, not the number of passing cases. State:

- Which model/interface and source were executed: checkout, extracted fixture,
  or the actual candidate's frozen package.
- Which updates, pass-end, selection, chunk and disk-restore events occurred.
- Which components were substituted and which real-data/native paths remain unrun.

Passing unit or fixture counts never imply "ready to submit" or "will not fail".
Before an authorized submission, use the actual prepared package with the
existing [release/platform owner](KAGGLE_WORKFLOW_SUBMISSION.md). Native admission
uses the existing worker/preflight, under the run's authority. Do not create
extra cloud experiments just to replace a missing local integration check.
The legacy [CPU preflight modes](EXPERIMENT_PREFLIGHT.md) have explicit family
and version limits; they are not a universal current-family rehearsal.

Navigation changes use the existing `documentation_check --document PATH`
selection. Evidence changes use their existing RML validators. Do not rerun
unaffected successful checks or append another test-count report here.
