# Evidence reviewed before arithmetic

RML lookup --mechanism fusion on desktop5f55a7b5 finds the prior fixed500K
fusion and the historical full-fusion closure. Their accepted decisions were
read before this plan.

- experiments/pcqm_k1_gptrans_pairnorm_fusion_500k/decision.md: older Clean-BN
  K1+GPTrans equal0.097580 versus two-GPTrans equal0.097525 eV. Model count and
  opportunity cost matter; a gain over one expert alone is insufficient.
- experiments/pcqm_k1_gptrans_full_fusion/decision.md: older full K1/GPTrans
  fusion improved its constituents but failed the accepted EdgeState reference.
  Complementarity at500K does not guarantee full-scale superiority.
- New accepted K1 owner codex/exp/k1-read-batched-muon-500k-20261107,
  commit6a006272164bc01d5f6848dca49defa9ded67cf8, decision.md and strict terminal
  evidence: original K1 batched Muon0.101863694448 eV, 60passes, strict replay.
  Current_read is negative and excluded from this diagnostic.
- experiments/pcqm_gptrans_bondlocal_pairnorm_lr_500k/terminal_acceptance/decision.md:
  selected EMA GPTrans LR1e-3 0.099987377815 and LR5e-4 0.100502566586 eV;
  both60passes and the same50K development cohort. Frozen accepted original
  owner tipbaedffc211490b1d0e805891c04a5628317908b9 is retained in archive.

The new K1 evidence remains on its owner. Reference it by immutable Git commit,
artifact digest and external locator; do not copy its canonical evidence or
claim that desktop's older RML index already includes its new trajectory.
This diagnostic changes only the expert set, not any historical disposition.
