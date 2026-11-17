# Current State

> This file owns the production identity and closed Track B conclusions on
> master. Full evidence stays on `molgap-desktop` and its experiment branches
> (see `PROMOTION_20261117.md`). Master keeps stable results only, no job
> status. `ROADMAP.md` on master is older than this file.

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
repaired-2M presets, and `load_uq_bundle` needs an `archive` branch checkout
(`archive_root`). Asset paths and hashes belong to `models/README.md` and the
production decision linked above.

## Track B Results

Track B predicts the official PCQM4Mv2 HOMO-LUMO Gap only. It does not change
Track A. This section lists closed, verified results. Each line names the
`molgap-desktop` commit or branch that holds the evidence; see
`PROMOTION_20261117.md` for how to fetch it. Experiment folders are not on
master.

- **K1 with FP32 Muon, 100K: adopted for the 500K transfer pair.** Selected
  Live MAE 0.132521 eV against 0.141294 eV for AdamW in the same job. It is
  not a full-scale or production replacement. Evidence:
  `experiments/pcqm_k1_muon_100k_20261106/decision.md`, desktop commit
  `5039fa0c`.
- **K1 with Muon, 500K: positive but below the gate.** Muon 0.102468 eV against
  AdamW 0.104282 eV, a gain of 1.814 meV (95% interval -2.386 to -1.250 meV).
  The frozen bar was 3 meV, so nothing is adopted. The 100K gain of 8.774 meV
  shrank to 1.814 meV at 500K. Evidence:
  `experiments/pcqm_k1_muon_500k_20261106/decision.md`, branch
  `codex/exp/k1-muon-500k-20261106`, commit `6b824342`.
- **GPT-T full-scale levers (decided 2026-11-16).** Tail holdout, scored once
  per endpoint, one-sided 99% bound.
  - A passed. Four B1024 inits (seeds 42-45), equal weights: tail MAE 83.992
    meV, 1.357 meV better than the two-init pair (85.349 meV). Init-only noise
    std is 0.099 meV, which understates full training noise.
  - B (batch 256), C (depth 20), D (90 passes) and E (cross-recipe ensemble)
    failed their bars. B256 was worse than B1024 by 1.4-1.5 meV.
  - The tail picked the lever, so 83.992 meV is optimistic. The final
    four-init ensemble scored 0.08281 eV on official valid (reported, not
    used to choose). Official test was not read. No adoption or production
    change follows.
  - Evidence: `experiments/pcqm_gptrans_full_levers_20261115/`, branch
    `codex/exp/gptrans-full-levers-20261115`, commit `33ab4571`.
- **K1 plus GPT-T fusion, full scale: FAIL.** Tail MAE: K1 97.026 and 99.006
  meV, GPT-T 87.941 and 87.846 meV. The fusion was 0.229 meV worse than GPT-T
  seed 42. Evidence:
  `experiments/pcqm_full_minus_tail_fusion_20261113/decision_full_minus_tail.md`,
  branch `codex/exp/gptrans-full-minus-tail-20261110`, commit `e32009ae`.
- **TokenGT bond tokens, 100K: NEGATIVE_UNDER_CONTRACT (2026-11-17).** One seed
  on the internal 50K development set: 140.676 meV against 139.440 meV for the
  control, +1.236 meV (95% interval +0.281 to +2.174 meV). The bar was -3.0 meV.
  Official valid and test were not read. Evidence:
  `experiments/pcqm_gptrans_tokengt_bond_100k_20261117/decision.md`, branch
  `codex/exp/gptrans-tokengt-bond-100k-20261117`, commit `1dc9c014`.
- **GPT-T width and FFN, 500K: no arm passed, route still open (2026-11-17).**
  Tail MAE, two seeds. Base R 110.913 and 110.924 meV. W320 was 2.387 meV worse.
  F3 gained 0.363 meV, below the 1.5 meV bar. The depth-20 calibration arm
  (L20) showed no signal (-0.440 meV; full scale gave +0.89 meV). By the
  pre-registered rule, 500K cannot show a capacity effect here, so these
  zero results do not close the width or FFN route. No full-scale run
  follows. Evidence: `experiments/pcqm_gptrans_width_500k_20261117/decision.md`,
  branch `codex/exp/gptrans-width-500k-20261117`, commits `532be162` and
  `ac983785`.

Other reusable code is in `src/molgap/` (K1 and GPT-T models, tail-holdout
scoring support, registry). Weights are not in the repository.

## Boundaries

- The leaderboard specialist predicts Gap only and stays isolated from the
  Track A three-target production registry.
- PCQM official validation and test-dev are not architecture-tuning data.
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
| Where do the Track B results above come from? | `PROMOTION_20261117.md` |
| What is the Track B final decision? | `production/04_evaluate/project_freeze/track_b_final_decision.md` |
| Which model artifacts exist? | `models/README.md` |
| Which reference models were used for which role? | `models/REFERENCE_INDEX.md` |

The immediate execution order is defined only in `ROADMAP.md`.
