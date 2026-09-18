# Decision: K1 MoSE hidden normalization

On 2026-09-19, Kaggle1 version 1 completed the frozen V5 seed-42 screen. Local
acceptance executed no model inference and verified the complete artifact,
source, cache, runtime, training and sealed-role contracts.

The candidate reached `0.1426464468 eV` at epoch 39 with 3,662,081 parameters.
The immutable K1-v4 reference was `0.1413736343 eV`, so the candidate regressed
by `0.0012728125 eV`. Its paired candidate-minus-reference row-error interval
was `[+0.0003173248, +0.0022565160] eV`, entirely unfavorable. It also regressed
by `0.0023919344 eV` relative to the prior unnormalized MoSE replacement.

The trajectory/subgroup audit showed that hidden BatchNorm worsened every MoSE
count-magnitude quintile relative to the unnormalized predecessor and had its
largest regression in the highest quintile. The failure is therefore not an
epoch-budget issue and does not support another normalization variant.

The V5 outcome is `NEGATIVE_UNDER_CONTRACT`; hidden normalization is closed.
No retry, extra seed, 500K bridge, protected-role evaluation, full training or
desktop handoff is authorized.

Evidence:

- `results/acceptance_v1.json`
- `results/postmortem_v1.json`
- local raw artifact root:
  `platforms/_records/kaggle/training/pcqm_k1_mose_hidden_bn_seed42_v1`
