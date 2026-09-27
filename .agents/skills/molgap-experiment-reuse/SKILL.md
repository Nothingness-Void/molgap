---
name: molgap-experiment-reuse
description: Reuse MolGap experiment infrastructure when adding or changing runners, frozen-checkpoint diagnostics, packaging, submission preparation, acceptance, or RML closure. Select existing components before writing plumbing. Not for ordinary result reading or a healthy monitor tick.
---

# MolGap experiment reuse

Use the selected MolGap checkout and its project Python. Follow its
[AGENTS](../../../AGENTS.md) and live state first. This skill routes implementation;
it grants no training, inference, submission, retry, role access or promotion.
Do not transfer custody of desktop jobs to server.

## Before writing plumbing

State a short reuse plan in the work update (not another evidence schema):

- **Task:** training, saved-artifact analysis, frozen-checkpoint inference,
  execution-only profiling, or infrastructure maintenance.
- **Reuse:** existing callable/CLI and the owning contract.
- **Gap:** the exact unsupported behavior and smallest new adapter/hook.
- **Check:** focused static/synthetic tests and the authorized execution boundary.

Read only the applicable section of [entrypoints](references/entrypoints.md).
Inspect the named function's signature and its nearest caller/test before use.
Calling a shared library directly counts as reuse; routing everything through
the CLI is not required. A legacy script is an example, not a new run template
whose credentials, model modes, run IDs or paths may be copied unchanged.

## Pick the right operation

| User needs | Implementation route |
|---|---|
| Train a candidate | Owning family trainer + shared training primitives; freeze only the scientific delta. A model constructor is not a trainer. |
| Compare retained predictions | Owning saved-artifact acceptance/analysis; no checkpoint loading or model execution. |
| Ablate a frozen checkpoint | Matching accepted loader/factory + inference intervention hooks; separate NO_TRAIN planning, role/cost records and terminal acceptance. |
| Profile execution | Owning profile adapter and frozen equivalence criteria; do not relabel profiling as a scientific comparison. |
| Package, bind or close a run | Existing source/receipt/RML APIs where compatible; a thin adapter for genuine schema differences. |

The shared `run-diagnostic` only probes metadata or constructs a random model.
It does not load checkpoints or run inference. Do not alter a scientific recipe
to fit an unsupported registry entry, or declare a fallback supported because
syntax/schema tests pass. Document the precise capability gap before extending
the owning shared module; keep preserved model/source identities unchanged.

## Keep the extension small

- Add only the new scientific mechanism, intervention or evidence translation.
  Reuse identity, atomic IO, hashing, source packaging, role/cost semantics,
  receipt reconciliation and terminal closure from their existing owners.
- If the same plumbing needs another experiment-specific copy, extract a small
  shared helper at its existing owner and test compatibility. Do not copy a
  whole experiment, or build a replacement framework for one missing hook.
- Put reusable behavior in `src/molgap/`; experiment CLIs remain thin. A skill
  does not contain another model implementation, scheduler or RML validator.
- Do not refactor a frozen remote payload while it is running. Record required
  infrastructure changes separately; preserve active jobs and user edits.

## Before any authorized remote submission

Use the owning platform workload skill if available; otherwise follow its
repository adapter documentation and state missing capability. The shared
experiment CLI is not a submitter. Verify executable inputs/mounts, entrypoint,
explicit accelerator request and account identity before uploading. Check
title/slug consistency and local artifact shape; avoid discovering packaging
mistakes after provisioning an accelerator.

After submission, bind the actual returned job ID/version and pulled source,
not an assumed slug. A logical ID may differ only with an explicit receipt
mapping. Reconcile an unknown submit outcome before retrying. Reuse the existing
server A/B monitor binding; do not create a new conversation or cron per run.

## Finish honestly

Report reused components, genuinely added code, tests and unresolved gaps.
Preserve prospective -> trace/artifacts -> role/cost -> acceptance -> terminal
RML evidence for the applicable task. NO_TRAIN may have no training trace; do
not fabricate one or claim training replay readiness. Apply the owning V5/RML
validators rather than introducing a skill-specific scientific gate.

This skill is an operating instruction, not a security-enforced release gate.
Use the repository's actual fail-closed validators and platform receipts.
