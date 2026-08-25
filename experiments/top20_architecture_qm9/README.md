# Top-20 Architecture Transfer Screen

**Question:** Can a transferable idea from the current PCQM4Mv2 top-20
architectures replace or improve the frozen MolGap Track C architecture on the
deployment-matched QM9 protocol?

This is a new, bounded hypothesis approved on 2026-08-22. It does not reopen
any closed result record and it does not change the production registry. The
official leaderboard snapshot and the transfer analysis are in
`top20_audit.md`; execution state and the predeclared gate are in
`decision.md`.

## Candidate selected for the bounded screen

The first candidate is `tgt_lite`, a MolGap-native implementation of the
transferable part of TGT/EGT and Transformer-M:

- global node attention rather than bond-only attention;
- an explicit pair channel containing ETKDG distances and bond features;
- a bounded triplet update so two pairs sharing a node can interact;
- the existing ETKDGv3 + MMFF200 graph path, so training geometry matches the
  deployment protocol.

This is intentionally not presented as an exact reproduction of the official
TGT model. The official model is a much larger, two-stage PCQM system with a
distance-prediction stage. The first screen isolates the architectural idea
before paying for a full PCQM retraining.

The seed-42 screen completed on IMS but failed the predeclared replacement
gate: test average MAE was `0.0930039063 eV` and Gap MAE was `0.1095121875 eV`.
No seed confirmation, production replacement, or PCQM Route B training is
authorized from this candidate.

## QM9 gate

- Split: 100,000 / 10,000 / 10,000, split seed 42.
- Targets: HOMO, LUMO, Gap in eV.
- Geometry: ETKDGv3 + MMFF200, seed 42 for the primary screen.
- Frozen old-architecture reference: average MAE `0.0708138843 eV`, Gap MAE
  `0.084272936 eV` from the promoted precision fusion record.
- A candidate is not a GO based on one lucky seed. It must improve average MAE
  by at least `0.003 eV` and Gap MAE by at least `0.002 eV` on the same
  successful-row intersection, then repeat the comparison across encoder seeds
  42/43/44. A close or mixed result is a rejection/hold, not a promotion.

No local model training is part of this experiment. The IMS adapter is
resumable and writes checkpoints, per-epoch progress, metrics, and independent
payloads below the approved project root.
