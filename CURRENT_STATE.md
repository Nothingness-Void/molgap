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
screens use V4, not historical Kaggle BS48 contracts. The Kunshan GPTrans-T
seed-42 100K/50K V4 reference is isolated on `codex/exp/gptrans-t-100k-v4`;
preflight `121922250` passed and training `121922252` was RUNNING in the last
recorded (2026-09-12) snapshot. Verify live state before releasing successors.

The Kunshan 500K distance-angle OOF blend is positive nomination evidence, but
its V4 audit found no runtime certificate or matching V4 reference. It is not
formal V4 cross-platform/causal evidence. Keep code and records isolated on
`codex/exp/gptrans-k1-geometry-500k` at `6d8630e` pending the declared bridge.
Its V4 comparability audit remains binding after full-run code integration.

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

- Accepted PCQM scale identities (100K, 500K, 1M, full) are held on IMS; both
  Kaggle accounts mirror accepted 100K/500K graphs.
- Legacy local-operator Kaggle screen is closed; recurrent graph-state is
  negative and archived. Neither is an active Kunshan experiment.
- The full K1/GPTrans-T implementation is integrated in this branch. The
  2026-09-14 evidence accepts GPTrans training at 156,250 steps; K1 recovery
  and dependent fusion have no terminal acceptance in this snapshot.
- Independent frozen GPTrans official-valid evaluation was submitted as CPU
  job 1507573.ccpbs1 on 2026-09-14. Its MAE is not yet recorded here.
  Follow the experiment results linked below; verify scheduler state before
  any successor or resubmission. These are recorded snapshots, not live polls.
- The K1/GPTrans-T fusion has one-time authorization for its frozen official-
  valid calibration/holdout split. Test-dev/challenge remain sealed.
- IMS EdgeState continuation `1364434.ccpbs1` passed acceptance; the decision
  is under `experiments/pcqm_edge_state_full/`.
- Every new remote run needs a frozen input contract, atomic checkpoints, and
  independently retrievable outputs before launch.

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
| V4 reference status? | Branch `codex/exp/gptrans-t-100k-v4`; last snapshot: `121922252`, 2026-09-12 |
| 500K geometry nomination? | Branch `codex/exp/gptrans-k1-geometry-500k` at `6d8630e`; not V4-comparable |
| Recurrent Kaggle disposition? | `archive` branch at `285e1dc`; negative, closed |
| Full K1/GPTrans-T chain? | `experiments/pcqm_k1_gptrans_full_fusion/README.md` and its linked 2026-09-14 recovery/evaluation evidence |
| What was submitted for OGB review? | `experiments/pcqm_edge_state_full/results/rich_full/submission_status.md` |
| Experiment/data/remote/model indices? | `experiments/README.md`, `platforms/README.md`, `models/README.md` |
