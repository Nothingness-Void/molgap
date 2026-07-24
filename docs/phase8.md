# Phase 8: Scaling and Architecture

> This is a dated method and decision map, not a live-status document. Read
> `CURRENT_STATE.md` for the active model and remote jobs, and `ROADMAP.md` for
> task order.

## Question

Can broader B3LYP training coverage and fixed-data architecture changes improve
the Phase 7 hybrid without sacrificing external hard-scope behavior?

## Fixed Contract

- Global target, geometry, code-boundary, and remote-durability constraints are
  defined once in `AGENTS.md`.
- Comparison: fixed common/OOD/P8-hard contracts plus explicitly sealed sets.
- Promotion: one favorable slice never changes registry selection.

## Decision Timeline

| Work | Historical outcome | Primary record |
|---|---|---|
| P8.1-P8.2 coverage audit and sampling | Identified and sampled sparse chemistry regions | `results/phase8/training_space.json`, `results/phase8/sampling_spec.md` |
| P8.3-P8.4 30k trainable-head pilots | Head complexity did not produce a transferable gain | `results/phase8/archive/legacy/pilots_30k/` |
| P8.5-P8.7 replacement300k | Broader trainable data produced the v2 historical base | `results/phase8/full_replacement_300k_summary.md` |
| P8.8 expansion500k | Replay plus broader top-up produced the v3 component hybrid | `results/phase8/full_expansion_500k_summary.md` |
| P8.9-P8.11 head/post-hoc/conformer probes | Closed as deployment paths | `results/phase8/archive/legacy/` |
| P8.12 routed dual-GPS | Passed the fixed architecture gate | `results/phase8/gps_arch_routed_decision.md` |
| P8.13 original-1M continuation | Retained only as a hard-coverage specialist | `results/phase8/expansion_1m/replay_fusion_decision.md` |
| P8.14 repair-v2 1M | Closed at the pure-2D external gate | `results/phase8/repair_v2_2d_external_eval/decision.md` |
| P8.15 additive 1.5M | Closed at the pure-2D external gate | `results/phase8/repair_v3_1p5m_external_eval/decision.md` |
| P8.16 residual specialists and additive acquisition | Established specialist evidence and a durable candidate-pool workflow | `results/phase8/broad_residual98k_external_eval/decision.md`, `results/kaggle/README.md` |
| P8.17 fixed two-expert ensemble | Passed the independent sealed comparison; deployment cost remains a gate | `results/phase8/multi2d_final_eval/decision.md` |
| P8.18 Exact-2M coverage expert | Specialist-positive but failed the global hard-scope gate | `results/phase8/multi2d_2m_coverage/decision.md` |
| P8.19 Hard20K rescue and bounded 2D+3D controls | Protocol defined; live execution state belongs in `CURRENT_STATE.md` | `results/phase8/multi2d_2m_hard20k/`, `results/phase8/multi2d_2m_1m3d_fusion_plan.md` |
| AR-01 through AR-06 | Closed architecture branches | `results/phase8/archive/README.md` |

## Method Boundaries

### Data scaling

Every additive dataset preserves identity exclusions against the frozen base,
evaluation identities, and any sealed scaffold set. Candidate counts are not
accepted until CID and canonical-SMILES reconciliation completes. The reusable
selection and acceptance logic lives in `src/molgap/multi2d_data.py`.

### Multi-expert evaluation

Pure-2D experts are compared on aligned identities with fixed ensemble formulas
and paired bootstrap deltas. The reusable evaluator lives in
`src/molgap/multi2d.py`; the CLI is
`scripts/phase8/evaluation/eval_multi2d_experts.py`.

### 2D+3D fusion controls

The 2D encoders may cover more rows than the frozen ETKDG SchNet branch. Fusion
alignment therefore uses stored source indices and never assumes positional row
equivalence. The bounded protocol is in
`results/phase8/multi2d_2m_1m3d_fusion_plan.md`.

## Reproduction Map

- Supported commands: `scripts/phase8/README.md`.
- Completed scale-up drivers: `scripts/phase8/archive/scaleup/`.
- Completed remote payloads: `scripts/phase8/archive/remote/`.
- Result evidence: `results/phase8/README.md`.
- Detailed narrative through P8.12:
  `docs/archive/phase8_detailed_history_through_p8_12_20260711.md`.

Open only the record for the decision being investigated; do not read every
Phase 8 artifact to reconstruct live state.
