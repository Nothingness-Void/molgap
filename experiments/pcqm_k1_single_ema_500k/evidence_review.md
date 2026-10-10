# Evidence review

2026-10-10. Query RML and owning records before creating this question.

- [Complete K1 audit](../pcqm_k1_complete_local_audit/analysis_zh.md) identifies
  package/output-state confounding; it does not prove weak readout or inadequate
  capacity. Width/slot/readout/tail/endpoint-average rescue is not repeated.
- [Clean-fit attribution](../pcqm_k1_clean_fit_generalization_500k/attribution.md)
  observes improved late clean training fit without matching development gain.
- [BN mechanism](../pcqm_k1_bn_mechanism_a100/terminal_decision.md) recovers
  approximately1.245meV with unchanged parameters on consumed development;
  it does not justify disabling training Dropout.
- [Single/mean2 owner](D:/w/k1-t4-cost-quality/experiments/pcqm_k1_t4_cost_quality/terminal_acceptance/decision.md)
  found40.93--49.22% fixture step saving, but single failed100K quality
  noninferiority. That outcome is preserved, not reclassified.
- [Clean-second](../pcqm_k1_clean_second_100k/terminal_acceptance/decision.md)
  failed its material gate; this experiment does not repeat that intervention.
- True K1 EMA100K owner `codex/exp/k1-ema-100k-night-20261006` has no qualified
  scientific endpoint. Its contract copies live BN and does not calibrate;
  it neither proves nor disproves EMA with matching clean BN on500K.

The new discriminator is a same-A100 matched-exposure EMA comparison on the
single-forward500K recipe, with identical normalization controls. The cheapest
falsifier is the user-limited four-hour partial screen; failure or ambiguous
partial observations do not release another expensive run.
