# Evidence review

The RML trajectory index retains the seed43 single/mean2 qualification failure,
the negative clean-second pair and local execution diagnostics. Follow their
own accepted decisions; do not reinterpret them as positive qualification.

- `experiments/pcqm_k1_clean_second_100k/terminal_acceptance/decision.md`:
  fresh mean2 was0.140257285566;0.141020240435 was clean-second, not single.
- `docs/operations/K1_EXECUTION.md`: adopted initialized-layout preserves the
  observed100K trajectory with7.31% TRAIN reduction on its native T4 pair.
  It does not establish500K speed or quality, or qualify a new EMA recipe.
- The retained single/EMA500K T4 run137983743 stopped for cost at matched18
  epochs with EMA0.999. It compared two single-forward arms, not fresh mean2
  versus EMA0.9999; it is not a60-epoch endpoint or authority to repeat it.
- User-supplied local true-EMA100K directory is uncommitted and has unresolved
  Clean-BN/source and comparison issues. Treat it as context, not strict V5
  qualification or proof of expected500K benefit.

The new question changes comparator and EMA decay and establishes a concurrent,
same-allocation500K pair. It is explicitly released by the user, not by positive
100K promotion. Reuse current shared training/qualification/output/RML owners.
No old reference is silently retrained to patch a historical evidence gap.
