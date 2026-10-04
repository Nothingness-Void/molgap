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
| P1 | C-GPTRANS-EMA-SCALE | Qualify one matched fixed500K EMA attribution without adding architecture modules or mixing old reference contracts. | [Scale release review](experiments/pcqm_gptrans_ema_portability/scale_training_review.md); supported G1 trainer, prospective reference/trace/role/cost binding and bounded optimizer-inclusive feasibility |
| — | C-GPTRANS-EMA-PORTABILITY | Accepted NO_TRAIN portability; no training Replay claim. | Preserve the [terminal audit](experiments/pcqm_gptrans_ema_portability/attempt_v4/decision.md) |
| — | C-GPTRANS-FINAL-READOUT | Closed without promotion; both actual complete Replay pairs verified. | Preserve the [terminal decision](experiments/pcqm_gptrans_readout_100k/gpu/results/decision.md); no automatic successor |

The path+EMA combination is terminal with no supported additivity; preserve its
[decision](experiments/pcqm_gptrans_path_ema_combination_100k/gpu/results/decision.md).
The readout dual is terminal; no further training is released. CPU prerequisite and failed-attempt bookkeeping remain in the
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
