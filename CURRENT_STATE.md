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

The routed-v4 model remains registered for compatibility; historical Delta/UQ
bundles still target their v3 base. Asset hashes are indexed in `models/README.md`.

## Active Objective

Track B is a PCQM4Mv2 Gap-only line, separate from Track A. New cross-platform
screens use V4, not historical Kaggle BS48 contracts. The GPTrans-T seed-42
100K/50K V4 reference completed on Kaggle2 and passed mechanical acceptance at
`0.156627 eV`. It is frozen as a reusable reference but closed for 100K
promotion; authority: `experiments/pcqm_gptrans_t_100k_v4/decision.md`.

The Kunshan 500K distance-angle OOF blend is positive nomination evidence, but
its V4 audit found no runtime certificate or matching V4 reference. It is not
formal V4 cross-platform/causal evidence. Its decision and binding comparability
audit are in `experiments/pcqm_geometry_transfer_500k/`; no scale-up is
authorized without the declared matched bridge.

The older Kaggle2 recurrent screen is closed and missed its comparator by
`0.0006301 eV`; it is archived at commit `285e1dc`. Do not resume or scale it.

The strict OGB-rich EdgeState baseline completed official-train-only training
from random initialization. Its public reproduction repository is
`https://github.com/Nothingness-Void/pcqm4mv2-edgestate`; public head `ae00b44`
passed a fresh-clone audit. OGB review status is indexed in
`experiments/pcqm_edge_state_full/results/rich_full/`.

## Prior Architecture Evidence

PairGPS2D's PubChemQC-100K validation-only screen and the QM9 R3-R10 tournament
are architecture evidence only; neither authorizes PCQM transfer. Their
decisions are indexed under `experiments/`. The conservative 2D+3D repair is a
separate untrained Track A task; frozen-2D plus dual-SchNet remains rejected.

## Execution State

- Kunshan V4 bootstrap passed source and graph checks. CPU setup `122210841`
  and dependent DCU gate `122210845` were pending in the 2026-09-16 snapshot;
  no formal training was submitted. Evidence is in
  `platforms/_records/scnet/kunshan_v4_bootstrap_20260916/README.md`.
- The matched 500K V4 experiment completed. K1 (`0.104860 eV`) and GPTrans-T
  (`0.106868 eV`) beat EdgeState (`0.111349 eV`) on the aligned 50K development
  role; K1's advantage over GPTrans-T remained below the `0.003 eV` nomination
  floor. The local matched ablation also rejected dense/sparse global attention;
  K1's molecular-slot gain over local-only was positive but sub-threshold.
  Authorities: `experiments/pcqm_500k_v4_evidence/final_decision.md` and
  `experiments/pcqm_500k_v4_evidence/local_ablation_decision.md`.
- The matched Kunshan distance-angle Triangle 500K screen passed local replay:
  candidate `0.114141 eV` versus baseline `0.117659 eV` on the same 50K
  development rows. It is nominated under its frozen point gate, but has no
  V5 transfer qualification or automatic scale release. Authority:
  `experiments/pcqm_distance_angle_500k/decision.md`.
- Xi'an full-model replay job `67440607` matched predictions but differed in
  269 cross-process gradient tensors. The audit is closed negative; Xi'an V4
  ranking remains gated. Authority: `experiments/pcqm_xian_determinism/decision.md`.
- The standalone desktop Kaggle3 Pair Update Norm 500K bridge completed 60
  epochs. Its 50K-row paired internal-development result is `0.105728 eV`
  versus GPTrans-T `0.106868 eV`; the `0.001140 eV` gain misses the frozen
  `0.003 eV` promotion floor. It is closed `NEGATIVE_UNDER_CONTRACT` after
  replay-ready artifact acceptance and RML indexing. This is separate from
  the Noisy Nodes + Pair Update Norm joint experiment recorded below.
  Authority: `experiments/pcqm_gptrans_pair_norm_500k/decision.md`.
- The full K1/GPTrans-T R3 chain completed mechanical acceptance. On the same
  73,545-row official-validation role, K1 scored 0.106672 eV, GPTrans-T scored
  0.109012 eV, and their fixed 50:50 blend scored 0.102227 eV. The calibrated
  blend scored 0.102186 eV on its frozen four-fifths holdout, but remained
  behind the 0.099638 eV EdgeState reference. The route is closed without
  promotion; exact evidence is under
  `experiments/pcqm_k1_gptrans_full_fusion/results/accepted_k1_gptrans_fusion_r3/`.
- The one-time K1/GPTrans-T official-validation calibration role is consumed.
  Test-dev/challenge remain sealed.
- The 2026-09-19 IMS K1 and GPTrans-T continuation jobs completed and passed
  local no-inference terminal acceptance. K1 stopped on patience after 79,185
  additional steps; its selected score remains `0.106672 eV` with no gain.
  GPTrans-T stopped at the frozen six-pass budget after 158,370 additional
  steps; its selected EMA score is `0.103791 eV`, improving its source by
  `0.005221 eV`. Zero stale evaluations mean GPTrans-T has no demonstrated
  convergence plateau. Official validation was consumed for selection;
  test-dev/challenge remain sealed. Authorities:
  `experiments/pcqm_k1_full_convergence/decision.md` and
  `experiments/pcqm_gptrans_full_convergence/decision.md`.
- Accepted PCQM scale identities are held on IMS and mirrored to the accepted
  Kaggle account datasets for the 100K/500K roles.
- Kaggle3 account `nvoid912` now has accepted private, byte-identical 100K and
  500K fixed-data mirrors plus the accepted V5 desktop runtime source layer.
  Script and notebook batch probes both retained `TpuV5E8` metadata but ran on
  CPU, so TPU training is blocked pending an interactive allocation and
  PyTorch/XLA/PyG compatibility gate. Evidence is under
  `platforms/_records/kaggle/preflight/`.
- IMS EdgeState continuation `1364434.ccpbs1` passed acceptance; the decision
  is under `experiments/pcqm_edge_state_full/`.
- GPTrans Noisy Nodes 500K on Kaggle 2 completed 60 epochs (`0.105056 eV`,
  +0.001812 eV over the frozen `0.106868 eV` GPTrans reference, trailing K1 by
  0.196 meV). Mechanical acceptance passed, but the scientific outcome is
  `NEGATIVE_UNDER_CONTRACT` under the frozen `< 0.103868 eV` gate.
  Authority: `experiments/pcqm_gptrans_noisy_nodes_500k/decision.md`.
- GPTrans Noisy Nodes + Pair Update Norm 500K on Kaggle 3 (`nvoid912`) completed
  60 epochs (`0.104812 eV`, 0.048 meV below K1 and +2.056 meV over the frozen
  GPTrans reference). Mechanical acceptance passed, but the scientific outcome
  is `NEGATIVE_UNDER_CONTRACT` under the frozen `< 0.103868 eV` gate.
  Authority: `experiments/pcqm_gptrans_noisy_pair_norm_500k/decision.md`.
- Every new remote run needs a frozen input contract, atomic checkpoints, and
 independently retrievable outputs before launch.
- Desktop operations follow the installed V5 topology: desktop-owned remote
  work remains durable and desktop-owned while offline; no default desktop
  heartbeat, server fallback, or cross-machine takeover is part of live work.
  Reconcile authoritative remote state when the desktop returns.

## Boundaries

- Track B predicts only Gap and cannot replace Track A without a separate
  production gate. Official validation is not a screen-tuning role; the authorized
  frozen-model scoring and fusion uses are recorded above. Test-dev/challenge stay sealed.
- IMS access is limited to `/lustre/home/users/sm2/chou/` and governed by
  `platforms/REMOTE_HANDOFF.md`.
- Common/OOD/P8-hard are one-time Track A acceptance roles, not tuning data.
  Router, MoE, dataset replacement, and closed fusion routes stay closed without
  a materially new question and stop rule in `ROADMAP.md`.
- Train/inference conformer construction remains ETKDG-consistent.

## Evidence Map

| Question | Authoritative pointer |
|---|---|
| What ships now? | `production/README.md` |
| Why is the repaired-2M pure-2D model recommended? | `production/04_evaluate/project_freeze/track_a_final_decision.md` |
| V4 reference status? | `experiments/pcqm_gptrans_t_100k_v4/decision.md`; accepted and closed for 100K promotion |
| 500K geometry nomination? | `experiments/pcqm_geometry_transfer_500k/decision.md`; positive but not V4-comparable |
| Recurrent Kaggle disposition? | `archive` branch at `285e1dc`; negative, closed |
| Full K1/GPTrans-T chain? | `experiments/pcqm_k1_gptrans_full_fusion/README.md` and its linked 2026-09-14 recovery/evaluation evidence |
| What was submitted for OGB review? | `experiments/pcqm_edge_state_full/results/rich_full/submission_status.md` |
| Experiment/data/remote/model indices? | `experiments/README.md`, `platforms/README.md`, `models/README.md` |
