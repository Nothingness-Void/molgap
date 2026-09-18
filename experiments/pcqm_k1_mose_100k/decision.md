# Decision: K1 MoSE replacement on fixed PCQM 100K

On 2026-09-19, Kaggle1 seed-42 version 5 completed the frozen V5 screen after
four infrastructure-only failures. No official validation, test-dev, or
test-challenge role was read, and local acceptance executed no model inference.

The artifact and execution contracts passed. The candidate used 3,661,697
parameters, reached its best development Gap MAE of `0.1402545124 eV` at epoch
36, and preserved every non-intervention contract field relative to the
immutable K1-v4 reference. K1-v4's development Gap MAE was `0.1413736343 eV`.

The MoSE replacement therefore produced a directional gain of
`0.0011191219 eV`. Its paired row-bootstrap error-delta interval was
`[-0.0020334447, -0.0002045147] eV`, entirely favorable. This establishes a
within-run prediction improvement, but row bootstrap does not measure
between-training-run stochasticity.

The predeclared promotion gate was `0.003 eV`. The candidate did not pass it,
so the V5 scientific outcome is `NEGATIVE_UNDER_CONTRACT` with a secondary
classification of `DIRECTIONAL_SUBTHRESHOLD`. It received no additional seed,
500K bridge, protected-role evaluation, full training, or desktop handoff.
The protocol's cheapest falsifier also closes MoSE subsets, RWSE concatenation,
hyperparameter rescue, and extra-seed rescue under this question.

The threshold was not revised after observing the result. Historical
same-database evidence contains a `0.0026921320 eV` difference between two
nominally identical scratch-through-40 jobs, so this point gain cannot yet be
distinguished safely from training-level variation. V5 permits a different
threshold only in a later prospectively frozen contract backed by an explicit
repeatability calibration; it does not permit retroactive promotion.

Evidence:

- `results/acceptance_v5.json`
- `../pcqm_gap_architecture/results/pretraining_dual_account_seed42/summary.json`
- local retained artifact root:
  `platforms/_records/kaggle/training/pcqm_k1_mose_seed42_v5/pcqm_k1_mose_100k`

