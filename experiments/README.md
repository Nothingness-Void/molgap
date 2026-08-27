# Experiments

One directory per **question**, never per calendar phase. A phase number ages;
a question does not. Anything whose output does not enter the model registry or
change the recommended predictor belongs here rather than in `production/`.

Each directory holds its own decision record and evidence. Read the decision
first; the metrics beside it are backup, not the entry point.

Track ownership is defined only in `TRACKS.md`: general-model work belongs to
Track A, PCQM leaderboard specialists to Track B, and architecture discovery to
Track C.

| Question | Verdict | Directory |
|---|---|---|
| Which architecture family survives cheap elimination? | Track C complete; GPS9/GPS11-160 advance, TensorNet and EGNN eliminated | `qm9_architecture/` |
| Can bounded pure-2D pair-state repairs beat PairGPS2D? | R3 validation selected persistent EdgeState; its one-time QM9 test remains sealed | `top20_architecture_qm9/` |
| Which leaderboard-inspired architecture is feasible from scratch under 12 hours, and why did prior 2D+3D fusion fail? | Persistent EdgeState Structural GPS passed the three-seed 100K gate and is the sole repaired-2M scale-up candidate; the full-scale run has not started | `resource_bounded_architecture/` |
| Does the accepted architecture transfer to real molecules? | Two-SchNet precision fusion accepted at 100K scale | `pubchemqc100k_architecture/` |
| Can pure-2D PairGPS2D beat the fixed GPS7 plus GPS9 equal comparator? | It passed the matched validation-only stage; the test role remained sealed and no long run was authorized | `pubchemqc100k_architecture/` |
| What is the PCQM-only Gap ceiling of that architecture? | Track B complete; the four-encoder bounded fusion passed its fixed official-validation gate | `pcqm_route_b/` |
| Which bounded pure-2D architecture should serve the official PCQM4Mv2 Gap leaderboard? | Open; EdgeState remains the seed-42 comparator after learned-query and three matched local-operator challengers failed | `pcqm_gap_architecture/` |
| Can a task-level PCQM Gap specialist beat the general model? | Yes on PCQM only; stays deterministically routed | `pcqm_gine_expert/` |
| Does scaling the repaired corpus to 2M help? | Yes on the general scopes; its pure-2D presets are the frozen Track A identity, while PCQM regresses | `repaired_2m_scaling/` |
| Do multiple pure-2D experts beat one? | Fixed two-expert ensemble is strongest but needs four passes | `multi2d_experts/` |
| Can repair fix a scaled corpus without refetching? | Yes; row ledger reconciles 3.4M source rows | `data_repair/` |
| Is ETKDGv3+MMFF worth its construction cost? | Yes; bare ETKDG rejected | `conformer_protocol/` |
| Can a student compress the expert ensemble? | No; external retention fails | `distillation/` |
| Which SchNet compute shape is efficient? | `176/160/6` at 78% params and 48% time | `schnet_arch/` |
| Does 1M continuation beat the 500K base? | Specialist only; no global promotion | `expansion_1m/` |

`_closed/` holds branches that are settled and must not be rerun without a
materially new hypothesis. `_scripts/` holds entrypoints shared by more than one
experiment; single-use runners live with their experiment, at its directory root
(for example `pcqm_route_b/build_pcqm_route_b_1m.py`).

An experiment CLI resolves paths from `molgap.constants` roots, never from
`Path(__file__).parents[n]`; `tests/test_repository_layout.py` enforces this so a
future move cannot silently break an entrypoint.

## Evidence Contract

Every active experiment must satisfy this chain:

1. its row in this file points to the experiment directory;
2. the experiment `README.md` points to at least one dated decision owned by
   that directory;
3. the decision points to compact machine evidence such as `metrics.json`,
   `summary.json`, `acceptance.json`, or `manifest.json`;
4. remote submission and retrieval provenance stays under `platforms/` or the
   experiment's `STATUS.md`/`results/REMOTE_LOG.md`;
5. live state links to the decision and never copies its metrics.

`tests/test_repository_layout.py` enforces index coverage, decision reachability,
machine-evidence presence, and size limits on the two live control documents.

## File roles inside an experiment

| File | Holds |
|---|---|
| `decision.md` | What the experiment concluded. The entry point. |
| `STATUS.md` | Live operational state of its remote jobs, when it has any |
| `results/REMOTE_LOG.md` | Finished remote rounds, so `CURRENT_STATE.md` need not restate them |
| `results/*.json` | Exact metrics behind the decision |

`CURRENT_STATE.md` lists only the live production identity, active or blocked
work, and pointers here for the rest.

Live status and the recommended model are in `CURRENT_STATE.md`; task order is in
`ROADMAP.md`. Compute-environment adapters are in `platforms/`.
