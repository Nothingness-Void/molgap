# Decision: K1 molecule-context MoSE gate

On 2026-09-19, Kaggle1 version 1 completed the frozen V5 seed-42 screen. Local
acceptance executed no model inference and verified the source, immutable
cache, runtime, training, artifact and sealed-role contracts.

The candidate reached `0.1419007480 eV` at epoch 39 with 3,687,682 parameters.
The immutable K1-v4 reference was `0.1413736343 eV`, so the candidate regressed
by `0.0005271137 eV`. It also regressed by `0.0025731176 eV` relative to the
positive selective-MoSE predecessor. The paired candidate-minus-reference
row-error interval was `[-0.0003508736, +0.0014102968] eV`; it crossed zero and
did not support a reliable gain.

The best checkpoint occurred at epoch 39 of 40, so the result does not support
an undertraining explanation or an extension of the same run. Richer
molecule-level context failed to preserve the predecessor's benefit. Under
this contract, the evidence closes this contextual-gating mechanism and the
immediate MoSE gate-refinement family rather than motivating another gate
width, seed or schedule variant.

The V5 outcome is `NEGATIVE_UNDER_CONTRACT`. No retry, extra seed, 500K bridge,
protected-role evaluation, full training, desktop handoff or successor screen
is authorized by this result.

Evidence:

- `results/acceptance_summary_v1.json`
- local raw artifact root:
  `platforms/_records/kaggle/training/pcqm_k1_mose_context_gate_seed42_v1`
