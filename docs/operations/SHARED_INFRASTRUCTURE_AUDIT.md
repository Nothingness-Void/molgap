# Shared infrastructure drift audit

Snapshot: 2026-10-15. This is a source/interface audit, not a new research
contract, training release, historical-result reacceptance or live branch index.

## Scope and disposition

Subsequent mechanism extraction is documented by the
[Common API](../../src/molgap/common/README.md). This dated audit remains the
pre-extraction snapshot; schema/policy convergence is not implied by that API.

- Desktop: `1b8401b4ebd4c53cc97ea6c76433caaa90b58ad0`.
- Server: `d52cda1d9effe982fc855a941658b0792a75e436`.
- The desktop commit selectively integrates server `d3508cf2` with desktop
  compatibility fixes. [Port scope and verification](KAGGLE_WORKFLOW_SUBMISSION.md)
  owns that implementation. No whole server branch or research results merged.
- Later server `c5062e50` adds analysis/closure helpers; it was inspected, not
  adopted. The two later result-acceptance commits were not imported.
- Source audit used Git objects and two independent read-only reviews. It did
  not access credentials, remote schedulers, molecular data or model inference.

`src/molgap/` has 251 tracked files on desktop and 400 on server: 207 are common
paths, of which 145 have identical blobs and 62 differ; 44 exist only on desktop
and 193 only on server. These are file counts, not 299 required migrations.
Many one-sided files are research/model extensions, not missing generic wheels.

The existing `model_substrate.py`, `family_training_state.py`, `fast_sandbox.py`,
`source_dependencies.py` and `shared_model_adapter.py` are byte-identical across
these snapshots. Keep them; do not build another substrate or addon registry.

## Concrete causes

Server links below are pinned to the inspected commit. Desktop paths refer to
the integrated source. These findings do not retrospectively negate results.

| Boundary | Observed divergence | Why a selective port breaks |
|---|---|---|
| Prospective identity | Desktop [validation](../../src/molgap/experiment_prospective.py) checks source/run/action/attempt. Server [publication](https://github.com/Nothingness-Void/molgap/blob/d52cda1d/src/molgap/experiment_prospective.py#L94) replaces plan action commits with the packaged commit. | Same function name has different provenance semantics; not an OS difference. Desktop read-only readiness also does not yet exercise its executable-commit gate. |
| Same-run schema | Desktop's [generic peer validator](../../src/molgap/experiment_family_workflow.py) accepts the generic three-key envelope. Server [dispatch](https://github.com/Nothingness-Void/molgap/blob/d52cda1d/src/molgap/experiment_family_workflow.py#L695) routes the same format to a cycle-specific contract requiring authority/audit/cache fields and fixed modes/budget. | Identical format labels are not interchangeable contracts. Do not weaken either validator merely to import the other file. |
| Ownership | Server [planner](https://github.com/Nothingness-Void/molgap/blob/d52cda1d/src/molgap/research_memory/plan.py#L52) defaults/restricts owner to server; its [proposal builder](https://github.com/Nothingness-Void/molgap/blob/d52cda1d/src/molgap/execution_comparison.py#L69) fixes server/C/Kaggle3. Desktop requires explicit provenance. | Scientific ownership is configuration/authority, not the host running shared code. |
| Hidden API prerequisites | The server builder assumes `ExperimentSpec.write` and `read_packaged_text`; desktop previously had neither. | A feature commit can depend on older branch-local API changes; copying only its visible files is not a dependency-complete integration. The port reuses desktop serialization and adds the small reader. |
| Source closure | Desktop [inventory](../../src/molgap/experiment_source_inventory.py) includes serialization-only roots. Server [inventory](https://github.com/Nothingness-Void/molgap/blob/d52cda1d/src/molgap/experiment_source_inventory.py#L109) differs and includes question-specific release modules in common bootstrap. | Static imports alone do not close pickle dependencies. Unrelated experimental release contracts become apparent infrastructure dependencies. |
| Recipe and metrics | Server [recipe](https://github.com/Nothingness-Void/molgap/blob/d52cda1d/src/molgap/family_training_recipe.py#L191) extends the same format with sidecars/execution config. Its [output owner](https://github.com/Nothingness-Void/molgap/blob/d52cda1d/src/molgap/family_training_outputs.py#L119) changes training metric units for bounded modes. | Same format/column names do not prove equal fields or units. Make policies explicit; never rewrite retained traces to match a new reader. |
| Inspection | Server [snapshot API](https://github.com/Nothingness-Void/molgap/blob/d52cda1d/src/molgap/experiment_family_workflow.py#L620) reuses digest-bound inspections. Desktop [acceptance](../../src/molgap/experiment_workflow.py) retains target-encoding identity through its older path. | Copying the optimization without the target binding can lose a validation guarantee. A common snapshot must bind both. |
| Package admission | The [desktop path filter](../../src/molgap/experiment_package.py) special-cases an `outputs` filename; server has broader Python filename admission. Core JSON/build/verify serializers are otherwise equivalent. | Path policy, not another packager implementation, needs convergence. |
| SDK/platform API | Server [submission](https://github.com/Nothingness-Void/molgap/blob/d52cda1d/src/molgap/kaggle_submission.py#L41) uses global-auth-prone SDK setup and a dataset status signature incompatible with the inspected desktop SDK. | Mock APIs accepted an unverified call shape. Local SDK compatibility is not evidence of server's installed version; use one maintained SDK adapter and no-network API-contract tests. |

The [desktop port](KAGGLE_WORKFLOW_SUBMISSION.md) also preserves manifest and
pickle bindings during remote-byte revalidation and includes exact acceptance
attachments. A passing server-only suite did not exercise these desktop gates.

## False generality to remove

The server K1/GPTrans `*_infrastructure_comparison` builders bind particular
100K/seed42/T4 recipes, evidence paths and authority. They are experiment
profiles, not server-machine capabilities. Their frozen checks should remain
with that question; their serializer/IO mechanisms can be shared.

Desktop [execution_proposal](../../src/molgap/execution_comparison.py) now takes
owner/track/platform explicitly, but still embeds a seed42/T4x2 falsifier,
hardware description and estimated-cost defaults. Its present scope is the
T4 execution comparison, not a universal research proposal generator.

New server [execution_analysis](https://github.com/Nothingness-Void/molgap/blob/d52cda1d/src/molgap/execution_analysis.py#L15)
fixes 781 steps/99968 presentations per pass and a two-arm capture topology;
[execution_terminal](https://github.com/Nothingness-Void/molgap/blob/d52cda1d/src/molgap/execution_terminal.py#L19)
still embeds row identities, Track C and role/cost disposition. Reuse its pure
analysis/IO, but move those facts to a frozen profile before calling it generic.
Legacy profilers adapt a trainer ABI, not a desktop/server distinction.

## Target boundary

```text
one shared mechanism
  schema/serialization + identity + dependency closure + training/state/IO
  + digest-bound acceptance + RML translation
       |
explicit frozen question configuration
  owner / track / family / addon / data / roles / budget / selection / units
       |
platform skill and adapter
  credentials / mount root / allocation / POST / scheduler / retrieval
```

Server bounded-research policy and desktop full-admission authority may remain
different. Their wrappers must call the same mechanism; neither library should
infer owner from machine/branch or import the other controller. Registration
sets can differ, but capability reporting and schema meanings must not.

Convergence should reuse `ExperimentSpec`, `TrainingAdapter`, `ArtifactAdapter`
and the existing V5/RML owners, not add another registry or validator framework:

1. Agree one generic identity/schema contract and explicit experimental profiles.
2. Converge recipe parsing, source/pickle closure and capability ABI together.
3. Share SDK/auth/transport/manifest handling; preserve each frozen profile.
4. Isolate model/loss/sidecar changes behind reviewed family/addon hooks rather
   than adding experiment-conditioned branches to the outer training loop.
5. Build one infrastructure patch with a minimal neutral workflow fixture for
   both owner configurations, then integrate that reviewed patch into both
   long-lived branches. Batch the implementation before one grouped acceptance;
   rerun only failed/affected tests. Check actual callables and source closure,
   not merely matching module names or permissive SDK test doubles.

Do not wholesale-copy the desktop port back to server: that would break its
existing question-specific acceptance formats. Back-port SDK/auth fixes and
common mechanism changes with their caller/profile adaptations as one reviewed
batch. Frozen source packages and historical evidence remain unchanged.

This audit does not unify those interfaces yet, change V5, introduce a live
cross-machine coordinator, authorize training, or modify server/production.
