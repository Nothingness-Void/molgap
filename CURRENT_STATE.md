# Current State

> Live truth only. Historical methods, metrics, and failures live in the
> [conditional history index](docs/operations/CURRENT_STATE_HISTORY_INDEX.md)
> and the linked experiment decisions; task order lives in `ROADMAP.md`.

## Server active screen

The authorized T4x2 [final atom/bond readout dual](experiments/pcqm_gptrans_readout_100k/gpu/submission_record.md)
is closed: both arms are independently accepted, STRICT_CAUSAL, and admitted
as complete Replay pairs, but neither supports promotion; preserve the
[decision](experiments/pcqm_gptrans_readout_100k/gpu/results/decision.md).
No server GPU job or successor is released; the existing Luna heartbeat is paused.
The path + corrected-EMA combination is closed with a strict/complete Replay
pair but no supported additive gain; preserve its
[decision](experiments/pcqm_gptrans_path_ema_combination_100k/gpu/results/decision.md).

The failed CPU attempts and accepted CPU NO_TRAIN preparation are RML-closed.
Seven stale ACTIVE plans were reconciled without changing prospective bytes,
scientific results, live job bindings, or replay eligibility; see the dated
local reconciliation routed by the tracked [`research_memory/README.md`](research_memory/README.md).

## Shared experiment infrastructure

The shared local CLI, typed Spec/addon registry, source packaging, prospective
planning, strict launch binding, family output inspection, portable recovery,
allocation retention, and terminal/RML helpers are integrated. Family trainers
own model/loader/optimizer/selection/resume behavior; platform adapters own
submission, reconciliation, and retrieval. Real remote qualification remains
separate from local schema or synthetic checks.

Start with the [new-agent quickstart](docs/operations/EXPERIMENT_QUICKSTART.md),
then use the [addon guide](docs/operations/EXPERIMENT_ADDON_GUIDE.md) and
[registered workflow](docs/operations/REGISTERED_EXPERIMENT_WORKFLOW.md).
The shared CLI is a local identity, packaging, diagnostic, receipt, preparation,
inspection, and terminal entry point; it is never a platform submitter.
Compatible new PCQM graph families reuse the [shared graph owner](docs/operations/GRAPH_SCREEN_EXTENSION.md).
Local proof and repeatable checks live in the [bounded verification record](docs/operations/INFRASTRUCTURE_VERIFICATION.md).

## Production

- Recommended Track A model: repaired-2M three-GPS dense pure 2D.
- Registry: `repaired_2m_dense_2d`; lower-cost preset: `repaired_2m_equal_2d`.
- Loader: `load_repaired_2m_2d` in `src/molgap/inference.py`.
- Authority: `production/04_evaluate/project_freeze/track_a_final_decision.md`.
- Track B experiments cannot change this registry.

## Track B ownership

Neural-Atom K1 and GPTrans-T remain the frozen desktop-owned comparison
architectures. The desktop full chain and bounded continuations are terminal and
accepted; EdgeState remains the official-validation reference. Desktop owns any
new full-scale training, official evaluation, and final submission. Details and
decision pointers are in the [historical index](docs/operations/CURRENT_STATE_HISTORY_INDEX.md).

## Live boundaries and next actions

- No registered server GPU chain is active or awaiting terminal acceptance.
  The [proposed G1 EMA portability audit](experiments/pcqm_v5_route_portfolio/overnight_plan_2026-10-04.md)
  is planned only; compute is not released and Luna remains paused.
- Use one immutable baseline per matching scientific contract. Track B predicts
  direct Gap on fixed PCQM4Mv2; Track A remains on repaired-2M PubChemQC.
- ETKDG must match between training and inference. Official validation was
  consumed for desktop selection; test-dev/challenge remain sealed, and any new
  protected-role access requires the owning authority's one-time record.
- GPU/DCU work follows immutable cache acceptance, atomic checkpoints,
  independently retrievable outputs, and a dated decision. Scientific failures
  close a route; unchanged-contract infrastructure failures may retry.
- Molecular-research-server access is governed by
  [`platforms/REMOTE_HANDOFF.md`](platforms/REMOTE_HANDOFF.md). The server does
  not adopt, monitor, or resume desktop-owned jobs.

## Pointers

| Question | Authority |
|---|---|
| What is active? | This file |
| Historical state and authority map | `docs/operations/CURRENT_STATE_HISTORY_INDEX.md` |
| What happens next? | `ROADMAP.md` |
| How to add or run a family/addon | `docs/operations/EXPERIMENT_QUICKSTART.md` |
| What ships? | `production/README.md` |
| Track meanings | `TRACKS.md` |
| Experiment directory/evidence index | `experiments/README.md` and `experiments/EVIDENCE_INDEX.md` |
| Remote operations | `platforms/README.md` and `platforms/REMOTE_HANDOFF.md` |
| Code ownership | `ARCHITECTURE.md` and `docs/operations/ARCHITECTURE_MODULE_INDEX.md` |
| Artifact inventory | `models/README.md` |
