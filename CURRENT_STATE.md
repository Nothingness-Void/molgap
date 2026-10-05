# Current State

> Live truth only. Historical methods, metrics, and failures live in the
> [conditional history index](docs/operations/CURRENT_STATE_HISTORY_INDEX.md)
> and the linked experiment decisions; task order lives in `ROADMAP.md`.

## Server active screen

The Kaggle2 [independent pair-transition100K screen](experiments/pcqm_gptrans_pair_transition_100k/gpu/results/interpretation.md)
is independently accepted, STRICT_CAUSAL and admitted as a complete Replay
pair, but did not support promotion. Its final real-pair branch was disconnected
from the virtual-node readout; this limits mechanism-level interpretation and
is retained explicitly, not repaired retroactively. The exact route and Luna
binding are closed; no server training or automatic successor is released.
No GRIT, extra seed,500K training or portability inference is authorized by
this terminal result. User-reported Kaggle1 dual500K remains outside server custody.

The authorized T4x2 [final atom/bond readout dual](experiments/pcqm_gptrans_readout_100k/gpu/submission_record.md)
is closed: both arms are independently accepted, STRICT_CAUSAL, and admitted
as complete Replay pairs, but neither supports promotion; preserve the
[decision](experiments/pcqm_gptrans_readout_100k/gpu/results/decision.md).
The server-owned frozen G1 EMA audit v4 is accepted and RML-finalized as
NO_TRAIN / PAIRED_ENDPOINT: corrected EMA passed frozen-weight portability,
not 500K optimization or full-scale qualification. Preserve its
[decision](experiments/pcqm_gptrans_ema_portability/attempt_v4/decision.md).
Earlier infrastructure failures and CPU qualification remain separately
retained in the [audit index](experiments/pcqm_gptrans_ema_portability/README.md).
The [fixed500K execution qualification](experiments/pcqm_gptrans_scale_qualification/results/interpretation.md)
is accepted and RML-finalized as NO_TRAIN / CONTEXT_ONLY, not a training Replay pair.
The [isolated G1 decay-coefficient falsifier](experiments/pcqm_gptrans_decay_clock_100k/gpu/results/interpretation.md)
is accepted and RML-finalized with a complete Replay pair; the intervention
failed its promotion gate and the coefficient route is closed.
The user authorized four physical Kaggle2 studies across two isolated T4x2
notebooks. The [node capacity pair](experiments/pcqm_gptrans_capacity_nodes_100k/protocol.md)
and [local-bond plus500K EMA study](experiments/pcqm_gptrans_capacity_relations_100k/protocol.md)
were submitted as version1, physical IDs137072037 and137072203, respectively.
The node capacity pair is independently accepted, STRICT_CAUSAL and admitted as
two complete Replay pairs; neither passed promotion. Its exact configurations
are closed; preserve the [interpretation](experiments/pcqm_gptrans_capacity_nodes_100k/gpu/results/interpretation.md).
The [local-bond plus500K study](experiments/pcqm_gptrans_capacity_relations_100k/gpu/results/interpretation.md)
is independently accepted and RML-finalized. The local addon has a complete
STRICT_CAUSAL Replay pair but remains below its frozen promotion gate. The500K
shared-live EMA study supports the correction direction, remains noncausal
PAIRED_ENDPOINT and is not a causal Replay pair or full-scale qualification.
All four earlier authorized studies and the separately authorized pair-transition
job are closed; Luna has no active bound training job.
Exact receipts and the [closed shared monitor binding](experiments/pcqm_gptrans_capacity_nodes_100k/gpu/monitor_binding.json)
retain custody history. No automatic successor is authorized.
The earlier monitor bindings are closed; the monitor never adopts desktop jobs.
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

- Preserve the four-study terminal interpretation and independent qualifications.
  Reopening any local-addon transfer or new architecture requires a separate
  covered compute decision. No successor, full training, extra seed or
  protected-role access is authorized by this closed campaign.
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
