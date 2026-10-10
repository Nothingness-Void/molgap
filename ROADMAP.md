# Roadmap

> Priorities and release gates only. Live truth is in `CURRENT_STATE.md`; each
> experiment owns its methods, metrics, and historical decision.

## Goal

Select one direct-Gap PCQM4Mv2 leaderboard specialist under a default ceiling
of 12 A100 hours. Track C discovers architectures cheaply, Track B validates
target-domain transfer, and desktop alone owns full training and submission.

## Active queue

| Priority | ID | Task | Exit condition |
|---|---|---|---|
| — | C-GPTRANS-SOURCE-DEPENDENCE | Closed: source/gate observations did not establish the proposed broad cohort-collapse rationale. | Preserve [NO_TRAIN / CONTEXT_ONLY interpretation](experiments/pcqm_gptrans_source_dependence/decision_terminal.md); no automatic training successor |
| — | C-GPTRANS-TRIPLET-PORTABILITY | Closed: positive but uncertain later-cohort point gain did not pass its frozen gate. | Preserve the [NO_TRAIN interpretation](experiments/pcqm_gptrans_triplet_portability/decision_terminal.md) and applicable complete diagnostic evidence; no500K training or automatic successor |
| 1 | C-GPTRANS-TRIPLET-COMMUNICATION | Both100K positives accepted; aggregation retained as the lower-cost transfer-study priority. | Preserve the [terminal interpretation](experiments/pcqm_gptrans_triplet_communication_100k/gpu/results/interpretation.md) and two complete Replay pairs; further execution requires separate authority |
| — | C-GPTRANS-LOCAL-RELATION | Closed: A positive below nomination gate, B worse; both actual complete Replay pairs accepted. | Preserve the [terminal interpretation](experiments/pcqm_gptrans_local_relation_100k/gpu/results/interpretation.md); no automatic successor, scale or extra seed |
| 1 | C-GPTRANS-LOCAL-TRANSFER | Positive same-budget500K result accepted with a complete Replay pair; retain without automatic expansion. | Preserve the [interpretation](experiments/pcqm_gptrans_local_transfer_500k/gpu/results/interpretation.md); late-exposure/full questions require new matched authority, and protocol-bound NO_TRAIN probes require accepted-parent hashes and separate records |
| — | C-GPTRANS-LOCAL-CONTROL | Closed without promotion; the active cap worsened its matched endpoint, with strict acceptance and an actual complete Replay pair. | Preserve the [terminal interpretation](experiments/pcqm_gptrans_local_control_100k/gpu/results/interpretation.md); no cap grid, extra seed or automatic scale |
| — | C-GPTRANS-BOTTLENECK | Accepted and closed NO_TRAIN diagnostic; the separately authorized amplitude-control falsifier did not support that fix. | Preserve the [terminal interpretation](experiments/pcqm_gptrans_bottleneck_audit/decision_terminal.md) and bounded control result; a distinct research question needs separate authority |
| — | C-GPTRANS-PAIR-TRANSITION | Closed without promotion; exact intervention accepted as a complete Replay pair, with the disconnected final-branch limitation retained. | Preserve the [terminal interpretation](experiments/pcqm_gptrans_pair_transition_100k/gpu/results/interpretation.md); any corrected mechanism or new family needs separate release authority |
| — | C-GPTRANS-CAPACITY | Closed four-study campaign; no architecture promotion. Preserve the500K EMA correction observation without a causal Replay/full-scale claim. | [Capacity falsifier](experiments/pcqm_gptrans_capacity_nodes_100k/gpu/results/interpretation.md) and [accepted local/scale synthesis](experiments/pcqm_gptrans_capacity_relations_100k/gpu/results/interpretation.md); separate authority before reopening |
| — | C-GPTRANS-DECAY-CLOCK | Closed without promotion; optimizer intervention accepted and complete Replay pair verified. | Preserve the [endpoint and trajectory interpretation](experiments/pcqm_gptrans_decay_clock_100k/gpu/results/interpretation.md); no coefficient grid or automatic scale |
| — | C-GPTRANS-EMA-PORTABILITY | Accepted NO_TRAIN portability; no training Replay claim. | Preserve the [terminal audit](experiments/pcqm_gptrans_ema_portability/attempt_v4/decision.md) |
| — | C-GPTRANS-FINAL-READOUT | Closed without promotion; both actual complete Replay pairs verified. | Preserve the [terminal decision](experiments/pcqm_gptrans_readout_100k/gpu/results/decision.md); no automatic successor |

The path+EMA combination is terminal with no supported additivity; preserve its
[decision](experiments/pcqm_gptrans_path_ema_combination_100k/gpu/results/decision.md).
The readout dual is terminal; the newly authorized four-study campaign is bounded by its own protocols. CPU prerequisite and failed-attempt bookkeeping remain in the
dated reconciliation routed by the tracked [`research_memory/README.md`](research_memory/README.md).
Closed queue records and route boundaries are in the [roadmap history index](docs/operations/ROADMAP_HISTORY_INDEX.md).

## Mandatory gates

1. Before GPU/DCU submission, syntax, manifest, source identity, immutable cache
   acceptance, and optimizer-inclusive runtime preflight must pass.
2. Every new screen follows [`experiments/SCREENING_POLICY.md`](experiments/SCREENING_POLICY.md):
   one frozen baseline per complete contract, reusable runtime certificates,
   physical BS128, and no partial optimizer batch.
3. Comparison must match data and row order, features, target, seed, precision,
   optimizer, schedule, loss, target transform, selection, exposure, and role
   access. Platform identity is provenance and may differ.
4. Promotion gain must exceed both the frozen material threshold and measured
   training-run variation; row bootstrap alone is insufficient.
5. Reused development roles select candidates only. A final transfer claim uses
   one frozen candidate and one disjoint, once-read audit role.
6. Full training requires exact optimizer-step/sample-exposure scaling,
   complete RNG/optimizer/scheduler resume state, runtime/source/data
   fingerprints, and a self-contained model bundle before official evaluation.
7. Official validation and test-dev stay sealed until desktop records one-time
   authority. Production promotion also needs Track A external, loader,
   registry, hash, latency, and smoke gates.
8. Infrastructure failures may retry only with the scientific contract unchanged;
   scientific failures receive a decision and close.

## Operating boundaries

- Do not modify the production registry during Track B work. Track B predicts
  direct Gap; Track A remains on the repaired-2M PubChemQC corpus.
- External data are separately labeled audit/OOD evidence, never silent training
  augmentation. ETKDG must match between training and inference; teacher models
  and privileged geometry are prohibited for the OGB line.
- One experiment carries one material mechanism. Seeds 43/44 require an explicit
  shortlist and compute-budget decision.
- Server performs architecture discovery and explicitly authorized Kunshan V4
  screens. Desktop owns full training, official evaluation, and submission.
- Read [`platforms/REMOTE_HANDOFF.md`](platforms/REMOTE_HANDOFF.md) before any
  molecular-research-server action; access remains restricted to its allowed path.
- One remote GPU candidate runs at a time unless a frozen T4x2 screen isolates
  independent candidates. Monitoring is mechanical; the coordinator releases
  successors.

## Conditional backlog

| Question | Trigger | Action |
|---|---|---|
| K1 full execution | Desktop budget approval plus accepted preflight | Run exactly the frozen contract; no tuning |
| New pure-2D architecture | Explicitly reopened discovery plus fresh evidence audit | Release one seed-42 QM9 route |
| Alternative geometry | Rule-compliant source available identically at train/inference | Freeze one isolated geometry question |
| Paper figures/write-up | Academic delivery requested | Derive only from accepted decisions |

Read [`CURRENT_STATE.md`](CURRENT_STATE.md) for live blockers and active jobs.
Read [`EXPERIMENT_QUICKSTART.md`](docs/operations/EXPERIMENT_QUICKSTART.md)
before adding a family or addon. Do not use this file as an experiment history;
use the [history index](docs/operations/ROADMAP_HISTORY_INDEX.md).
