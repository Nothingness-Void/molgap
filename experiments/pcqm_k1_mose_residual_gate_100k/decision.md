# Decision: K1 selective MoSE residual gate

On 2026-09-19, Kaggle1 version 1 completed the frozen V5 seed-42 screen. Local
acceptance executed no model inference and verified the source, cache, runtime,
training, artifact and sealed-role contracts.

The candidate reached `0.1393276304 eV` at epoch 39 with 3,673,394 parameters.
The immutable K1-v4 reference was `0.1413736343 eV`, a credible improvement of
`0.0020460039 eV`. The paired candidate-minus-reference row-error interval was
`[-0.0028973962, -0.0011806805] eV`, entirely favorable. The result therefore
supports the separate MoSE residual mechanism, but it does not pass the frozen
`0.003 eV` material-gain gate declared by this experiment. That value is an
experiment-local prospective policy threshold, not a permanent V5 constant.

Trajectory analysis found that the candidate improved the two hardest K1
absolute-error quintiles by `0.01860` and `0.04563 eV`, while degrading the two
easiest quintiles by `0.03757` and `0.01717 eV`. Only 50.76% of rows improved,
although 59.68% of corrections moved toward the target. The learned correction
shared only `R^2=0.312` with the earlier MoSE-replacement correction, so this was
not merely prediction interpolation. From epoch 34 to 39, development MAE
improved only `0.000116 eV`; extending the same training is not justified.

The V5 outcome is `POSITIVE_BELOW_GATE`. The candidate is not promoted to an
extra seed, 500K bridge, protected-role evaluation, full training or desktop
handoff. The next isolated question may refine only the gate so that molecule
context suppresses MoSE residuals on K1-easy rows; widening the residual,
changing normalization or adding epochs is not supported.

Evidence:

- `results/acceptance_summary_v1.json`
- `results/trajectory_v1.json`
- local raw artifact root:
  `platforms/_records/kaggle/training/pcqm_k1_mose_residual_gate_seed42_v1`
