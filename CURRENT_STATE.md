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
real-bond EdgeState GPS9 as the seed-42 comparator. Learned-query pooling then
failed its strict gate and is closed. The active search question is a bounded,
sequential comparison of three materially different edge-aware local operators
with one time-gated fallback, all under the same cache and optimizer contract.
Exact comparator metrics belong to
`experiments/pcqm_gap_architecture/results/seed42_structural_vs_edge_state/decision.md`.

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
  version 1, is the sole active GPU architecture search. It serializes three
  core local operators and permits the fourth only under its frozen time gate.
- The official-train-derived PCQM 100K graph cache and first matched seed-42
  comparison are accepted. The next remote stage is the single serialized
  local-operator search defined in
  `experiments/pcqm_gap_architecture/local_operator_search_protocol.md`; no
  multiseed or full-data run is authorized yet.
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
