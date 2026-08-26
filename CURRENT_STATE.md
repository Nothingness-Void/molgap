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

Track C is open for resource-bounded, from-scratch architecture discovery. Its
frozen comparator is `repaired_2m_dense_2d`; no Track C result changes the
production registry before the existing Track A external promotion gate.

Persistent EdgeState Structural GPS passed the controlled three-seed PubChemQC
100K gate and is the sole repaired-2M scale-up candidate. The complete 2M model
has not been trained. Submission is blocked until both conditions pass:

1. the immutable repaired-2M EdgeState input is complete and accepted;
2. a measured one-epoch projection keeps the bounded run below 10 hours.

The exact screen result and artifacts are owned by
`experiments/resource_bounded_architecture/results/edge_state_100k_multiseed/decision.md`.
The live execution contract and remote provenance are in
`experiments/resource_bounded_architecture/STATUS.md`.

## Parallel Workstream

The independent PairGPS2D line completed its matched PubChemQC-100K
validation-only search and passed that stage against the fixed GPS7 plus GPS9
equal comparator. Its test role remains sealed, and neither full-data training
nor Track B transfer is authorized. A train-role-only A100 benchmark selected a
safer higher-throughput configuration but did not establish an accuracy result.
The decisions are owned by
`experiments/pubchemqc100k_architecture/results/pair_gps_2d_fair_screen/decision.md`
and
`experiments/pubchemqc100k_architecture/results/pair_gps_2d_a100_benchmark/decision.md`.

The bounded PairGPS-R2 Lite repair failed its QM9 accuracy gate and is closed.
It met the parameter/stability budget and nearly matched the larger PairGPS2D,
but did not improve both required accuracy metrics. It does not authorize seed
repeats, PubChemQC-100K, or full-data work. The exact result is owned by
`experiments/top20_architecture_qm9/pair_gps_2d_r2_decision.md`.

The pure-2D R3 validation tournament completed and independently accepted its
frozen winner, `edge_state_structural_gps`. The dense PairGPS repairs did not
pass, and the conditional R4 trigger was not met. The QM9 test role remains
sealed; the one permitted test gate has not been submitted. One R5
identity-start multi-depth readout was evaluated on validation only and failed
both gates. R3 remains the sole frozen winner. The exact decisions and artifact
hashes are owned by
`experiments/top20_architecture_qm9/pair_gps_2d_r3_decision.md` and
`experiments/top20_architecture_qm9/edge_state_jk_readout_r5_decision.md`.
R6 node-level edge conditioning and R7 recurrent graph memory failed their
strict validation gates. R8 also failed after using multihop pairs as local
message edges. R3 remains frozen. R9 is the only open discovery question: it
keeps R3 local messages on real bonds and uses the accepted path cache only for
a shared shortest-path-biased sparse attention branch. The R8 decision and R9
contract are `experiments/top20_architecture_qm9/multihop_edge_state_r8_decision.md`
and `experiments/top20_architecture_qm9/sparse_path_attention_r9_protocol.md`.

The P1 conservative 2D+3D repair remains separate from the EdgeState scale-up.
Its exact-identity head, compact aligned payload, Colab runner, and resume
contract are implemented and locally tested. Model training has not started.
Its status and evidence pointers are owned by
`experiments/resource_bounded_architecture/README.md` and
`platforms/colab/conservative_2d3d_fusion/README.md`.

The previously rejected frozen-2D plus dual-SchNet residual remains closed and
must not be represented as this new conservative head.

## Execution State

- The R8 multihop CPU cache remains accepted and reusable. No architecture GPU
  job is active; the next permitted run is one R9 validation using that cache
  read-only. No test or downstream transfer is authorized.
- All accepted 100K architecture outputs have local manifests, metrics,
  predictions, and hashes under the experiment and `platforms/_records/` trees.
- No repaired-2M EdgeState job has been submitted.
- No remote job is required to support the frozen Track A or Track B claims.
- Any new remote run must first appear in `ROADMAP.md` with an input contract,
  timing bound, atomic checkpoint path, and independently retrievable outputs.

## Boundaries

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
| What did the active architecture tournament decide? | `experiments/resource_bounded_architecture/README.md` |
| What exactly passed for EdgeState? | `experiments/resource_bounded_architecture/results/edge_state_100k_multiseed/decision.md` |
| What did the independent PairGPS2D validation screen decide? | `experiments/pubchemqc100k_architecture/results/pair_gps_2d_fair_screen/decision.md` |
| What did the pure-2D R3 validation tournament decide? | `experiments/top20_architecture_qm9/pair_gps_2d_r3_decision.md` |
| Did the R5 multi-depth readout improve R3? | `experiments/top20_architecture_qm9/edge_state_jk_readout_r5_decision.md` |
| Did R6 node-level edge conditioning improve R3? | `experiments/top20_architecture_qm9/edge_conditioned_r6_decision.md` |
| Did R7 recurrent graph memory improve R3? | `experiments/top20_architecture_qm9/graph_token_r7_decision.md` |
| Did R8 multihop local messaging improve R3? | `experiments/top20_architecture_qm9/multihop_edge_state_r8_decision.md` |
| Where is the complete IMS record snapshot? | `platforms/_records/ims/README.md` |
| Where are all active and completed experiment questions indexed? | `experiments/README.md` |
| Where are rejected branches indexed? | `experiments/_closed/README.md` |
| How are remote jobs packaged and retained? | `platforms/README.md` |
| Which model artifacts exist? | `models/README.md` |

The immediate execution order is defined only in `ROADMAP.md`.
