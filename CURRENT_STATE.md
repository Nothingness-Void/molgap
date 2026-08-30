# Current State

> This file owns only live project truth: the production identity, active
> candidate, blocker, and immediate handoff. Metrics and historical conclusions
> belong to their dated decision records under `production/` or `experiments/`.
> Task ordering belongs only in `ROADMAP.md`.

## Production Identity

- **Recommended model:** repaired-2M three-GPS dense pure 2D.
- **Registry key:** `repaired_2m_dense_2d`.
- **Lower-cost preset:** `repaired_2m_equal_2d`.
- **Public loader:** `load_repaired_2m_2d` and
  `predict_smiles_batch_repaired_2m_2d` in `src/molgap/inference.py`.
- **Decision owner:**
  `production/04_evaluate/project_freeze/track_a_final_decision.md`.

The previous routed-v4 model remains registered for compatibility. Historical
Delta and UQ bundles remain calibrated to their historical v3 base, not to the
repaired-2M presets. Asset paths and hashes belong to `models/README.md` and the
production decision linked above.

## Active Objective

Track B is open for an official PCQM4Mv2 leaderboard specialist. The target is
the single official HOMO-LUMO Gap, not the Track A three-target contract.
Architecture selection uses only an internal 100K/10K split derived from the
official PCQM training role on Kaggle. Official validation and test-dev remain
sealed during selection.

The first matched official-PCQM question completed and froze persistent
real-bond EdgeState GPS9 as the initial comparator. Learned-query pooling,
three matched local-operator replacements, and recurrent graph state are
closed. Sparse Triangle R3 then completed its paired seeds 42/43/44
confirmation and improved EdgeState in every seed and their mean. It is the
accepted pure-2D comparator for materially new 100K questions; the exact result
is owned by
`experiments/pcqm_gap_architecture/results/sparse_triangle_edge_state_multiseed/decision.md`.

The seed-42 bottom-fusion screen established that deterministic ETKDG geometry
can help when injected into the accepted topology model inside every block.
Distance-only, angle-only, and their combination all passed the strict
single-seed arithmetic gate; distance plus angle was the clear winner. The
next server-side question is whether only that combined candidate reproduces
against fresh paired seeds 43/44. No confirmation task has been submitted. The
accepted seed-42 decision is
`experiments/pcqm_gap_architecture/results/geometry_bottom_fusion_seed42/decision.md`.

## Prior Architecture Evidence

The independent PairGPS2D line completed its matched PubChemQC-100K
validation-only search and passed that stage against the fixed GPS7 plus GPS9
equal comparator. Its test role remains sealed, and neither full-data training
nor Track B transfer is authorized. A train-role-only A100 benchmark selected a
safer higher-throughput configuration but did not establish an accuracy result.
The decisions are owned by
`experiments/pubchemqc100k_architecture/results/pair_gps_2d_fair_screen/decision.md`
and
`experiments/pubchemqc100k_architecture/results/pair_gps_2d_a100_benchmark/decision.md`.

The QM9 pure-2D tournament selected persistent EdgeState R3; R5-R10 failed their
strict validation gates and are closed. Those runs inform architecture choice
only. They do not supply weights, splits, metrics, or advancement authority to
the official PCQM screen. Their decisions are indexed by
`experiments/top20_architecture_qm9/README.md`.

The P1 conservative 2D+3D repair remains separate from the EdgeState scale-up.
Its exact-identity head, compact aligned payload, Colab runner, and resume
contract are implemented and locally tested. Model training has not started.
Its status and evidence pointers are owned by
`experiments/resource_bounded_architecture/README.md` and
`platforms/colab/conservative_2d3d_fusion/README.md`.

The previously rejected frozen-2D plus dual-SchNet residual remains closed and
must not be represented as this new conservative head.

## Execution State

- Kaggle2 kernel `kaseichou/molgap-pcqm-gap100k-local-operators-seed42`,
  version 1, completed. Its three required candidates passed no-inference
  artifact acceptance but all failed the scientific advancement gate; the
  time-gated fourth candidate was not launched.
- The official-train-derived PCQM 100K graph cache, completed seed-42
  comparisons, and the sparse topology-wedge cache are accepted. R2 terminated
  during preflight with no epoch or metrics output; its failure is recorded at
  `experiments/pcqm_gap_architecture/results/sparse_triangle_edge_state_r2_seed42/failure_diagnosis.md`.
  R3 completed, passed no-inference artifact acceptance, and strictly passed
  the seed-42 arithmetic gate. Its decision is
  `experiments/pcqm_gap_architecture/results/sparse_triangle_edge_state_r3_seed42/decision.md`.
  The paired EdgeState/Sparse Triangle seeds 43/44 task completed and passed
  no-inference acceptance; together with seed 42, Sparse Triangle improved all
  three paired comparisons. No full-data or official-role run is authorized.
- Geometry-cache version 2 completed after one infrastructure-only parent-
  identity repair. Its 110,000 aligned roles, invalid-geometry ledger, 22
  shards, and aggregate hash passed no-model acceptance.
- Geometry GPU version 1 stopped before training because its PyTorch build
  omitted P100 `sm_60`. The automatically repaired version 2 preserved the
  scientific contract, completed all three candidates, and passed no-inference
  artifact acceptance. The combined distance-plus-angle candidate is the sole
  seed-42 geometry winner. Its decision is
  `experiments/pcqm_gap_architecture/results/geometry_bottom_fusion_seed42/decision.md`.
- All accepted 100K architecture outputs have local manifests, metrics,
  predictions, and hashes under the experiment and `platforms/_records/` trees.
- No molecular-research-server access is authorized before Kaggle selects one
  three-seed winner.
- Any new remote run must first appear in `ROADMAP.md` with an input contract,
  timing bound, atomic checkpoint path, and independently retrievable outputs.

## Boundaries

- The leaderboard specialist predicts Gap only and stays isolated from the
  Track A three-target production registry.
- PCQM official validation and test-dev are not architecture-tuning data.
- The molecular-research-server boundary is `/lustre/home/users/sm2/chou/`;
  the current Kaggle stage does not access the server at all.
- Common, OOD, and P8-hard are a one-time acceptance gate after the standalone
  repaired-2M candidate completes; they are not architecture-tuning data.
- Future sealed data remains locked.
- Router, MoE, dataset replacement, and closed late-fusion branches remain
  closed unless `ROADMAP.md` records a materially new question.
- Track B remains an isolated PCQM Gap specialist and cannot replace Track A
  without a separate production gate. Its final decision is
  `production/04_evaluate/project_freeze/track_b_final_decision.md`.
- Train and inference conformer construction must remain ETKDG-consistent.

## Evidence Map

| Question | Authoritative pointer |
|---|---|
| What ships now? | `production/README.md` |
| Why is the repaired-2M pure-2D model recommended? | `production/04_evaluate/project_freeze/track_a_final_decision.md` |
| What is the active leaderboard experiment? | `experiments/pcqm_gap_architecture/README.md` |
| What exactly passed for EdgeState? | `experiments/resource_bounded_architecture/results/edge_state_100k_multiseed/decision.md` |
| What did the independent PairGPS2D validation screen decide? | `experiments/pubchemqc100k_architecture/results/pair_gps_2d_fair_screen/decision.md` |
| What did the pure-2D R3 validation tournament decide? | `experiments/top20_architecture_qm9/pair_gps_2d_r3_decision.md` |
| What is the official-PCQM 100K selection contract? | `experiments/pcqm_gap_architecture/pcqm100k_gap_screen_protocol.md` |
| Did replacing EdgeState with OGB local operators help? | `experiments/pcqm_gap_architecture/results/local_operator_search_seed42/decision.md` |
| Why is recurrent graph state the next PCQM candidate? | `experiments/pcqm_gap_architecture/architecture_route_audit.md` |
| What is the sparse triangle EdgeState question? | `experiments/pcqm_gap_architecture/sparse_triangle_edge_state_protocol.md` |
| Did Sparse Triangle pass seed 42? | `experiments/pcqm_gap_architecture/results/sparse_triangle_edge_state_r3_seed42/decision.md` |
| Did Sparse Triangle reproduce across three seeds? | `experiments/pcqm_gap_architecture/results/sparse_triangle_edge_state_multiseed/decision.md` |
| What is the bottom-fused geometry question? | `experiments/pcqm_gap_architecture/geometry_bottom_fusion_seed42_protocol.md` |
| Did bottom-fused ETKDG geometry pass seed 42? | `experiments/pcqm_gap_architecture/results/geometry_bottom_fusion_seed42/decision.md` |
| Did the R5 multi-depth readout improve R3? | `experiments/top20_architecture_qm9/edge_state_jk_readout_r5_decision.md` |
| Did R6 node-level edge conditioning improve R3? | `experiments/top20_architecture_qm9/edge_conditioned_r6_decision.md` |
| Did R7 recurrent graph memory improve R3? | `experiments/top20_architecture_qm9/graph_token_r7_decision.md` |
| Did R8 multihop local messaging improve R3? | `experiments/top20_architecture_qm9/multihop_edge_state_r8_decision.md` |
| Did R9 shortest-path sparse attention improve R3? | `experiments/top20_architecture_qm9/sparse_path_attention_r9_decision.md` |
| Where is the complete IMS record snapshot? | `platforms/_records/ims/README.md` |
| Where are all active and completed experiment questions indexed? | `experiments/README.md` |
| Where are rejected branches indexed? | `experiments/_closed/README.md` |
| How are remote jobs packaged and retained? | `platforms/README.md` |
| Which model artifacts exist? | `models/README.md` |

The immediate execution order is defined only in `ROADMAP.md`.
