# Current State

> This file owns live truth only: production identity, active question,
> blockers, boundaries, and evidence pointers. Historical methods, metrics,
> and remote logs belong to their dated decision records.

## Production identity

- **Recommended model:** repaired-2M three-GPS dense pure 2D.
- **Registry:** `repaired_2m_dense_2d`; lower-cost preset:
  `repaired_2m_equal_2d`.
- **Public loader:** `load_repaired_2m_2d` and
  `predict_smiles_batch_repaired_2m_2d` in `src/molgap/inference.py`.
- **Authority:** `production/04_evaluate/project_freeze/track_a_final_decision.md`.
- The previous routed-v4 model remains registered for compatibility; its Delta
  and UQ bundles are not calibrated for the repaired-2M presets.

## Active objective

Track B selects one Gap-only PCQM4Mv2 leaderboard specialist under an explicit
full-run compute budget; the default ceiling is 12 A100 hours. Official
validation and test-dev stay outside server-side architecture discovery.

The no-attention distance-angle triangle GraphState was a reproducible 100K
efficiency discovery but failed to transfer at full scale: its continuation
reached only about `0.118368 eV`, versus `0.099638 eV` for the accepted
EdgeState GPS9 continuation. GraphState and all derivatives are closed and
must not be presented as active candidates or comparators. Their complete
evidence is on `archive` commit `5d1be33`; the server tombstone is
`experiments/_closed/pcqm_server_archive_index.md`. The accepted desktop
EdgeState continuation remains the strongest accepted full-scale
official-validation evidence.

The bounded Ring-GraphState, ContactState, and body-order comparisons are
closed. Their details remain in their experiment decision records.
The bounded Kunshan VectorState, moment-readout, and ComponentState questions
are closed or retained only as weak evidence in their experiment records.

PNA statistics, edge retention, directed-bond memory, SignNet-LapPE, torsion,
and atom--bond dual-stream questions are closed or weak-only. Their evidence is
retained in experiment records and the archive indexes.

The consolidated attribution is
`experiments/pcqm_gap_architecture/results/architecture_failure_attribution_2026-09-08/decision.md`.

Sparse relative-value path attention was mechanically accepted and
scientifically rejected at seed 42. It increased cost and regressed against its
fresh paired GraphState9 control; the exact mechanism is closed in
`experiments/pcqm_gap_architecture/results/relative_value_graphstate_seed42/decision.md`.

Track C owns literature-inspired architecture triage on QM9-30K. A new
architecture can reach Track B only through a paired PCQM-100K transfer and a
once-read shadow audit; only a strong transfer is handed to desktop for
full-scale work. The governing
decision, thresholds, and exact scope of the rejected pretraining surrogates
are in
`experiments/pcqm_gap_architecture/results/three_stage_screening_reset_2026-09-08/decision.md`.
The train-role-only GAPE-lite positional-encoding question completed under the
global same-task, same-platform, exact-batch-128 contract and was rejected
before PCQM transfer. Matched graph alignment did not beat either the fresh
EdgeState GPS9+RWSE16 baseline or its equal-compute shuffled-alignment control.
The decision is `experiments/qm9_gape_pretraining/decision.md`.
The paired SCNet QM9-30K BRICS FragmentState question also failed its frozen
promotion gate and is closed without PCQM transfer. Its implementation,
mechanical acceptance, and decision are retained only on `archive` commit
`bca22b2` under `experiments/qm9_architecture/results/fragment_state_seed42/`.
The matched QM9 adaptive local-denoising screen completed and is closed. Its
adaptive arm was numerically best but missed both frozen material-effect gates;
no PCQM transfer or extra seed is authorized. The accepted comparison and
attribution are in `experiments/qm9_adaptive_denoising/decision.md`.
The pure-2D cardinality-preserving channel screen is closed. Its source-content
arm recovered most of the parameter-matched size-control regression but did not
beat the fresh EdgeState GPS9 baseline, so the result does not justify a PCQM
transfer. The accepted evidence and attribution are in
`experiments/qm9_cardinality_channel/decision.md`.
The multi-slot Neural-Atom global mixer completed and is closed. Replacing dense
GPS attention with latent communication was smaller, faster, and directionally
better, but four slots did not clear the material baseline gate and added less
than known run variation over the parameter-identical one-slot control. The
accepted result and attribution are in
`experiments/qm9_neural_atom_mixer/decision.md`. One of three post-cardinality
scientific attempts is consumed; two remain, and each requires a fresh
all-history audit before submission.
The single-harmonic Fourier EdgeState question passed its frozen QM9 Track C
gate against both the fresh full-GPS anchor and the architecture-matched
one-slot control while retaining its resource advantage. It is nominated for
exactly one paired PCQM-100K transfer. Kaggle2 T4x2 kernel
`kaseichou/molgap-pcqm-fourier-edge-s42` version 1 completed and passed
mechanical acceptance. Fourier-Edge did not beat its architecture-matched K1
control and is closed for PCQM. The one-slot latent K1 control itself produced
a large paired gain over fresh full GPS, with a bootstrap interval excluding
zero and the same direction in every target quartile. K1 is frozen unchanged
as the sole candidate for one independent label-sealed shadow audit; this is
exploratory K1 evidence, not a successful Fourier transfer. The complete
attribution is in `experiments/pcqm_fourier_edge_transfer/decision.md`.
Route 2/3 succeeded at Track C but failed to transfer its Fourier mechanism.
The final route-3/3 discovery attempt remains unused while K1 has an unresolved
independent-audit path.
The user authorized preparation of a paired K1 500K scale bridge. The bridge
preserves the accepted 10K development role and changes only train cardinality;
it is released to `molgap-desktop` only after K1 passes the pending shadow
audit. The label-sealed shadow cache passed acceptance, and Kaggle2 kernel
`kaseichou/molgap-pcqm-k1-shadow-audit` version 3 is performing the frozen
one-time audit without training. Its contract is
`experiments/pcqm_k1_scale500k/protocol.md`.
This work is independent of the desktop 304-wide/pretraining/full-scale work
reported by the user; those in-progress results do not alter the accepted
incumbent yet.
SCNet and Fragment-State work are desktop-owned and are excluded from this
server queue.
The user authorized a bounded autonomous server discovery window on 2026-09-10.
Terminal evidence is mechanically collected by Luna Max and handed to the
coordinator. The coordinator may release exactly one in-scope successor at a
time under the decision funnel in `ROADMAP.md`; this does not authorize extra
seeds, official roles, desktop/full-scale work, SCNet, IMS, or concurrent GPU
candidates.
The locally attached atom/bond/functional-group reconstruction question is now
classified as optimization of the already validated EdgeState model, not a new
architecture. The user therefore authorized it to enter directly at paired
PCQM-100K under
`experiments/pcqm_gap_architecture/local_hierarchy_pretraining_seed42_protocol.md`.
Its earlier QM9 work stopped after CPU cache acceptance and did not consume DCU
training. The original
GraphState/WedgeState contract reached no training because 254 selected QM9
records failed its required canonical-SMILES round trip. The replacement v2
protocol froze a canonical-valid source pool and the accepted pure-2D EdgeState
GPS9 backbone. Its isolated Kunshan CPU cache-and-acceptance job `121340444`
completed and passed mechanical acceptance; no v2 DCU preflight or training
job was submitted.

The PCQM local-label sidecar passed independent acceptance at aggregate SHA-256
`e7d0557d7ade3469d25d91512fbe0900c7441db94427bce3ed5bc1faa74218a6`.
Kaggle2 T4x2 kernel `kaseichou/molgap-pcqm-edgestate-local-hierarchy-s42`,
version 1, completed and passed no-model acceptance. Pretrain20 plus
fine-tune20 accelerated convergence but did not beat fresh scratch40 at equal
encoder exposure, so the exact schedule is closed. No shadow, extra seed, or
scale-up was released. The decision and the only justified optional allocation
follow-up are in
`experiments/pcqm_gap_architecture/results/local_hierarchy_pretraining_seed42/decision.md`.
The user explicitly authorized one fresh paired 10-pretrain/30-Gap follow-up.
Kaggle2 T4x2 kernel `kaseichou/molgap-pcqm-edgestate-hierarchy-10-30-s42`,
version 1, completed and passed no-model acceptance. The candidate improved
over its fresh scratch40 control by `0.00234205 eV`, converting the 20/20
allocation failure into a strong positive signal, but it missed the frozen
`0.003 eV` nomination gate by `0.00065795 eV`. The local-hierarchy allocation
question is therefore closed without shadow, another allocation, extra seeds,
scale-up, desktop handoff, official-role access, or IMS work. The complete
decision is
`experiments/pcqm_gap_architecture/results/local_hierarchy_allocation10_30_seed42/decision.md`.

Every newly frozen Track C or Track B model screen now uses physical batch
exactly 128 per independent model/device. A scientific comparison is valid
only when its arms share one task, platform, accelerator class, data roles,
seed, precision, optimizer, scheduler, row-order policy, and sample exposure.
Historical cross-task, cross-platform, and non-128 metrics are context only.
The single authority is `experiments/SCREENING_POLICY.md`.

## Prior architecture evidence

- Track A EdgeState three-seed evidence remains at
  `experiments/resource_bounded_architecture/results/edge_state_100k_multiseed/decision.md`.
- The independent PairGPS2D 100K validation and A100 throughput decisions are
  at `experiments/pubchemqc100k_architecture/results/`.
- QM9 R3 persistent EdgeState is retained as the pure-2D comparator at
  `experiments/top20_architecture_qm9/pair_gps_2d_r3_decision.md`.
- QM9 TGT, PairGPS R2, R4, and R5--R10 are archive-only; see
  `experiments/_closed/qm9_top20_archive_index.md`.
- The unfinished repaired-2M PairGPS2D attempt (job `1322114`) is archive-only;
  it has no final metrics and must not be resumed. See
  `experiments/_closed/pcqm_server_archive_index.md`.
- The conservative 2D+3D repair is separate and has not started training;
  see `experiments/resource_bounded_architecture/README.md`.

## Execution and boundaries

- Accepted PCQM caches and architecture comparisons are retained in their
  experiment records.
- SCNet Kunshan card1 is runtime-compatible for bounded GraphState jobs via
  the isolated DTK 23.10/Python 3.10 environment. The single-DCU GraphState
  runtime gate `120869290` completed with mechanical acceptance against the
  accepted 100K/10K geometry cache; its protocol, launch record, compact
  summary, and decision are
  `experiments/pcqm_gap_architecture/kunshan_graphstate_runtime_protocol.md`
  and `experiments/pcqm_gap_architecture/results/kunshan_graphstate_runtime_gate/`.
  Evidence for the earlier synthetic compatibility probe remains at
  `platforms/_records/scnet/training_compatibility_20260904/`.
- SCNet Xi'an Card2 now has an isolated DTK 22.10/PyG runtime with a
  device-width-corrected HIP `torch-scatter` build. A 10K train-role probe
  matched the pinned T4 throughput at batch 96, and two exact batch-48 100K
  three-epoch gates completed with accepted checkpoints. Their validation
  trajectories were not sufficiently repeatable for single-run ranking, so
  Xi'an is restricted to twice-reproduced paired pre-screens followed by a
  canonical Kunshan confirmation. The exact boundary is in
  `experiments/pcqm_gap_architecture/results/xian_card2_runtime_gate/decision.md`.
- The user reported 200 available accelerator-hours on each SCNet region on
  2026-09-06. This is availability, not permission to broaden a frozen
  experiment: Xi'an remains the rapid repeated pre-screen backend and Kunshan
  remains the canonical paired-confirmation backend.
- No molecular-research-server run is authorized before the Kaggle selection
  gate; later access is restricted to `/lustre/home/users/sm2/chou/`.
- Track B predicts Gap directly and cannot alter the Track A production
  registry. Official validation/test-dev and future sealed data are locked.
- The database contracts are unchanged: Track B remains on the official
  PCQM4Mv2 data, and Track A remains on the existing repaired-2M PubChemQC
  corpus. External databases and public datasets are reference or separately
  labeled OOD material only; they are not replacements or silent training
  augmentation. Privileged-geometry teachers and teacher distillation are
  excluded from the OGB leaderboard line.
- Train and inference geometry must use the same ETKDG construction.
- Every new remote run needs a protocol, immutable cache acceptance, atomic
  checkpointing, independently retrievable outputs, and a dated decision.
- Scientific failures close a route; only infrastructure failures may be
  repaired and retried under the unchanged scientific contract.

## Evidence map

| Question | Authority |
|---|---|
| What ships? | `production/README.md` |
| What is active? | `ROADMAP.md` and `experiments/pcqm_gap_architecture/README.md` |
| PCQM closed routes | `experiments/_closed/pcqm_server_archive_index.md` |
| QM9 closed transfer routes | `experiments/_closed/qm9_top20_archive_index.md` |
| QM9 R3 comparator | `experiments/top20_architecture_qm9/pair_gps_2d_r3_decision.md` |
| PCQM 100K contract | `experiments/pcqm_gap_architecture/pcqm100k_gap_screen_protocol.md` |
| Remote handoff rules | `platforms/REMOTE_HANDOFF.md` |
| Artifact inventory | `models/README.md` |

Task ordering is defined only in `ROADMAP.md`; monitor handoff rules are in
`AGENTS.md`.
