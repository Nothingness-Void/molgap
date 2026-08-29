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
real-bond EdgeState GPS9 as the seed-42 comparator. Learned-query pooling and
three matched local-operator replacements then failed their strict gates and
are closed. The next seed-42 question is a recurrent graph state inside the
accepted EdgeState backbone, selected by the resource-bounded route audit. It
has no seed-43/44 or full-data authorization. Exact completed conclusions
belong to the decisions under `experiments/pcqm_gap_architecture/results/`.

In parallel, the strict OGB-rich EdgeState baseline completed full official-only
training from random initialization. Its public reproduction repository and
checkpoint Release are at
`https://github.com/Nothingness-Void/pcqm4mv2-edgestate`; public head `ae00b44`
passed a fresh-clone audit. The OGB-LSC form was submitted on 2026-08-28 and is
awaiting code/report validity review. No test score or leaderboard rank exists
yet. Submission and reproducibility evidence is under
`experiments/pcqm_edge_state_full/results/rich_full/`.

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
- The official-train-derived PCQM 100K graph cache and all completed seed-42
  comparisons are accepted. Kaggle2 is running the sole matched recurrent
  graph-state seed-42 screen; no multiseed or full-data run is authorized.
- All accepted 100K architecture outputs have local manifests, metrics,
  predictions, and hashes under the experiment and `platforms/_records/` trees.
- IMS continuation `1364434.ccpbs1` completed and passed artifact acceptance.
  Positive convergence evidence and closure are recorded in
  `experiments/pcqm_edge_state_full/results/convergence_40/decision.md`.
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
| What was submitted for OGB review? | `experiments/pcqm_edge_state_full/results/rich_full/submission_status.md` |
| Where are all active and completed experiment questions indexed? | `experiments/README.md` |
| How are remote jobs packaged and retained? | `platforms/README.md` |
| Which model artifacts exist? | `models/README.md` |
