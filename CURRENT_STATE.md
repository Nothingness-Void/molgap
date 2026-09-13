# Current State

> This file owns only live project truth: production identity, active candidate,
> blocker, and immediate handoff. Historical evidence belongs to dated decisions;
> task ordering belongs only in `ROADMAP.md`.

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

Track B remains an official PCQM4Mv2 Gap-only line, isolated from the Track A
three-target production contract. On 2026-09-13 the user explicitly authorized
full official-train runs for frozen Neural-Atom K1 and GPTrans-T under separate
20M-sample-exposure contracts. The first execution attempt exposed a K1 host
memory cgroup limit and a GPTrans source-path bug; K1's step-5000 atomic
checkpoint was retained and verified. K1 later resumed to step 105,500 before
the same host-memory cgroup limit terminated it again. R2 preflight passed, but
resume failed before training because the guard treated a new runtime
certificate instance ID as a changed runtime, although the accepted runtime
fingerprint and frozen asset identities were unchanged. A local, tested fix now
preserves certificate lineage without weakening contract, data, source, or
runtime-fingerprint checks. The PBS service is currently unavailable, so no
successor has been submitted. Exact R2 identities, hashes, failure, and retry
gate are recorded in
`experiments/pcqm_k1_gptrans_full_fusion/results/recovery_20260914_r2.md`.
After both base artifacts pass acceptance, the dependent fusion job will read
the official-valid role exactly once, fit one scalar on a fixed 20% calibration
partition, and compare on the remaining 80%; this consumes that validation
role for this experiment.
No test-dev/challenge-test access is allowed. The experiment protocol is
`experiments/pcqm_k1_gptrans_full_fusion/protocol.md`.

Earlier 100K architecture-screen decisions remain unchanged and are indexed in
`experiments/pcqm_gap_architecture/results/`.

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

- Official PCQM4Mv2 identities are accepted: IMS holds hardlinked 100K, matched
  500K, 1M, and full views; both Kaggle accounts mirror the 100K/500K graphs.
- Kaggle2 kernel `kaseichou/molgap-pcqm-gap100k-local-operators-seed42` version 1
  completed. Three candidates passed artifact acceptance but failed the
  advancement gate; the time-gated fourth candidate was not launched.
- The IMS K1/GPTrans full-scale task remains open under its frozen contracts. K1
  R1 `1498717.ccpbs1` completed the resume at step 105,500 then hit the same
  host-memory cgroup limit; the checkpoint was preserved. R2 preflight
  `1502817.ccpbs1` passed, but resume `1502818.ccpbs1` failed before training
  because a newly accepted runtime-certificate instance ID was incorrectly
  treated as a changed runtime. Acceptance `1502819.ccpbs1` depends on resume
  success. The live checkpoint and trace match their pre-retry hashes.
  `jobinfo -c` currently fails against both PBS servers, so no R3 is submitted
  until scheduler state is available. The V4 asset audit and minimal repair are
  recorded in
  `experiments/pcqm_k1_gptrans_full_fusion/results/recovery_20260914_r2.md`.
  GPTrans and fusion recovery history remains in
  `experiments/pcqm_k1_gptrans_full_fusion/results/recovery_20260913_r1.md`.
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
| Which official PCQM scale identities may remote jobs consume? | `platforms/_records/ims/pcqm_fixed_datasets_v1/README.md` and `platforms/_records/kaggle/pcqm_fixed_datasets_v1/README.md` |
| Where are all active and completed experiment questions indexed? | `experiments/README.md` |
| How are remote jobs packaged and retained? | `platforms/README.md` |
| Which model artifacts exist? | `models/README.md` |
