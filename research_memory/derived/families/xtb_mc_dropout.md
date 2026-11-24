# xtb_mc_dropout (C2 xTB, C3 MC-dropout)

## Quick answer
Last closed (of 3, 0 are CLOSED reference arms):
TB-pcqm-c3-full-tail | NO_TRAIN | pcqm-c3-full-tail | no-training-dropout-layers-only-in | -0.21 | -
TB-pcqm-c3-checkpoint-space | NO_TRAIN | pcqm-c3-checkpoint-space | no-training-enable-dropout-layers-only | -0.75 | -
TB-pcqm-c2-xtb-diagnostic | NEGATIVE_UNDER_CONTRACT | pcqm-c2-xtb-diagnostic | no-backbone-training-compute-20-gfn2 | -0.14 | below_gate
Open (ACTIVE) 0: -
Reopen: Closed for post-hoc GFN2-xTB residual correction on these backbones; only a different training-time  / Explicit user authority and frozen single-score protocol to test K=32 MC-dropout on full-scale B1024 / A different ensemble (more diverse members or fewer members) or a different tail-free inference add-
src (src/molgap/): -
Experiments: pcqm_c2_xtb_diagnostic, pcqm_c3_checkpoint_space, pcqm_c3_full_tail_20261123

## Closed (3)
TB-pcqm-c3-full-tail | NO_TRAIN | pcqm-c3-full-tail | no-training-dropout-layers-only-in | -0.21 | -
TB-pcqm-c3-checkpoint-space | NO_TRAIN | pcqm-c3-checkpoint-space | no-training-enable-dropout-layers-only | -0.75 | -
TB-pcqm-c2-xtb-diagnostic | NEGATIVE_UNDER_CONTRACT | pcqm-c2-xtb-diagnostic | no-backbone-training-compute-20-gfn2 | -0.14 | below_gate

## Open (0)

## Levers (first 10)
family | arm | outcome | effect_meV | failure_mode | mechanism
pcqm-c2-xtb-diagnostic | pcqm-c2-xtb-diagnostic | NEGATIVE_UNDER_CONTRACT | -0.14 | below_gate | no-backbone-training-compute-20-gfn2
pcqm-c3-checkpoint-space | pcqm-c3-checkpoint-space | NO_TRAIN | -0.75 | - | no-training-enable-dropout-layers-only
pcqm-c3-full-tail | pcqm-c3-full-tail | NO_TRAIN | -0.21 | - | no-training-dropout-layers-only-in

## Reopen conditions
Closed for post-hoc GFN2-xTB residual correction on these backbones; only a different training-time or architecture mechanism under a new protocol reopens xTB features.
Explicit user authority and frozen single-score protocol to test K=32 MC-dropout on full-scale B1024 checkpoints (single and 4-init ensemble) on tail_holdout_v1.
A different ensemble (more diverse members or fewer members) or a different tail-free inference add-on under a new frozen protocol; this endpoint is scored once.

## src

## Experiments
- experiments/pcqm_c2_xtb_diagnostic: scripts probe_c2.py
- experiments/pcqm_c3_checkpoint_space: scripts probe_c3.py
- experiments/pcqm_c3_full_tail_20261123: scripts mc_tail.py
