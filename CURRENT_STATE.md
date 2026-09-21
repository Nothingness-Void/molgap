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
- Xi'an probe `67440607` reproduced predictions but not cross-process gradient
  hashes. Formal Xi'an V4 training remains gated; branch
  `codex/exp/xian-determinism-audit` owns the follow-up audit.
- The full K1/GPTrans-T R3 chain completed mechanical acceptance. On the same
  73,545-row official-validation role, K1 scored 0.106672 eV, GPTrans-T scored
  0.109012 eV, and their fixed 50:50 blend scored 0.102227 eV. The calibrated
  blend scored 0.102186 eV on its frozen four-fifths holdout, but remained
  behind the 0.099638 eV EdgeState reference. The route is closed without
  promotion; exact evidence is under
  `experiments/pcqm_k1_gptrans_full_fusion/results/accepted_k1_gptrans_fusion_r3/`.
- The one-time K1/GPTrans-T official-validation calibration role is consumed.
  Test-dev/challenge remain sealed.
- GPTrans/K1 convergence training remains incomplete. The 2026-09-16 attempts
  saved paused checkpoints and reported walltime plus cgroup memory warnings;
  dependent acceptance failed because completion manifests were absent.
  On 2026-09-19, K1 `1548227` and GPTrans-T `1548270` were resubmitted from
  those checkpoints with bounded PyG object caching, 16 CPU slots, one A100
  each, and 120-hour limits. Acceptance is conditional on completion. The
  frozen six-pass/patience-three stopping rule remains unchanged; neither model
  is proven converged. Deployment and prior-state evidence:
  `platforms/_records/ims/convergence_resume_20260919/README.md`.
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
  +0.003812 eV over GPTrans core baseline, trailing K1 by 0.2 meV) and passed
  mechanical and scientific acceptance as `POSITIVE_BELOW_GATE` under `< 0.103868 eV`.
  Authority: `experiments/pcqm_gptrans_noisy_nodes_500k/decision.md`.
- GPTrans Noisy Nodes + Pair Update Norm 500K on Kaggle 3 (`nvoid912`) is currently
  executing on `pcqm4mv2-ogb-fixed-500k-scnet-v1` under frozen falsifier `< 0.103868 eV`.
- GPTrans Dual-Stream Attentive Readout (DSAR) 100K Dual-Arm Screen on Kaggle 1 (`nothingnessvoid`)
  is currently executing on `pcqm4mv2-ogb-fixed-100k-v1` under dual NvidiaTeslaT4:
  - Arm A (`dsar_pair_norm` on GPU 0): Trajectory `TB-gptrans-dsar-pair-norm-100k-s42`, falsifier `< 0.153627 eV`.
  - Arm B (`dsar_noisy_pair_norm` on GPU 1): Trajectory `TB-gptrans-dsar-noisy-pair-norm-100k-s42`, falsifier `< 0.153627 eV`.
  Pre-flight replay-ready verification passed for both arms; expected duration ~1.4 wall hours (~2.8 device hours).
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
