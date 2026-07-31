# Presentation Evidence Pack

Every number the interview deck quotes, resolved to a local file.

```powershell
.venv\Scripts\python.exe production\04_evaluate\scripts\evaluation\build_presentation_evidence.py
```

Output: `presentation_evidence.json`. It recomputes the derived statistics the
deck needs (R2 per scope, worst-decile residual share, parameter counts) from
already accepted records. It creates no new model claim.

## Slide to evidence

| Slide | Claim | Where the number comes from |
|---|---|---|
| 5 | DFT is expensive | `cost_vs_dft.dft.summary` — 23.15 min median wall clock, 5.64 core-hours, per molecule |
| 6 | Task and corpus | `corpus` — 2,000,000 rows, 500,000 immutable targeted, B3LYP/6-31G* gas phase |
| 7 | Architecture | `architecture.presets` — passes, parameter counts, and that no preset touches 3D |
| 8 | What was ruled out | `rejected_paths` — four closed branches, each with the metric that closed it |
| 10 | Accepted accuracy | `accuracy.scopes` — MAE and R2 on 1,973 paired molecules across all/OOD/hard |
| 11 | Transferability | `transferability` — 0.112011 eV on 4,981 fixed-validation rows, plus the non-compliance reason |
| 12 | Delta and uncertainty method | `delta_and_uq` — including `bound_to_registry_key`, which is **not** the recommended model |
| 13 | The four gaps | `delta_and_uq.bound_to_recommended_model` (false), `delta_and_uq.delta_test_rows` (695), `geometry_leverage`, `experimental_offset` |
| 14 | Why the gate sits on the external set | `rejected_paths[3]` — the residual head passed its internal gate and lost +0.023251 eV externally |
| 16 | Experimental alignment is unmeasured | `experimental_offset` — the size and sign of the gap a solid-state head would need to close |
| 17 | Measured cost | `cost_vs_dft.ml` and `latency` — 0.75 ms/molecule batched, 62,051x one geometry step |

## Numbers in the outline that need fixing

Three claims in the draft outline do not match the local records.

**Slide 12: Delta/UQ is bound to `phase8_expansion_hybrid` (the v3 single
hybrid), not to routed-v4.** The outline says routed-v4. Both are previous-
generation bases and the substance of the limitation is unchanged, but the
registry key is wrong. `production/06_uq/results_v3/feature_config.json` is
authoritative.

**Slide 10's footnote is now stale.** It says packaging is unfinished and the
registry still points at routed-v4. Packaging closed on 2026-07-31; the
recommended key is `repaired_2m_dense_2d`. See
`../track_a_final_decision.md`.

**"worst 10% = 37.7%" is a Gap-only figure from the v3 model.** On the current
model the same statistic is 38.2% for Gap and 32.0% when averaged over all three
targets. The tail-heaviness conclusion holds; the number belongs to a different
model and a different target scope than the outline implies.

## Two row counts, both correct

The outline's rigor slide says "same 1,977 molecules" while the results table
says 1,973. Both are right, and the distinction is worth keeping straight if
asked:

- **1,977** is the fixed common-evaluation set. The shipped pure-2D presets run
  on all of it — no row is dropped, because 2D graphs build from SMILES alone.
- **1,973** is the paired comparison table. Four molecules failed ETKDG
  conformer generation, and the rejected 3D residual arms could not score them.
  Those four were excluded from *every* method so the comparison stayed paired,
  rather than letting the 2D models quietly score on more molecules than the 3D
  ones.

On the full 1,977 the dense preset gives 0.097735 / 0.106759 / 0.088516 eV
average MAE across all/OOD/hard, versus 0.097638 / 0.106655 / 0.088407 on the
1,973 subset. The difference is in the fifth decimal place; the paired table is
the one to quote because it is the one with a matched baseline.

## Deliberate omissions

The pack does not compute R2 against experimental values, and the experimental
record is filed under `experimental_offset` rather than under accuracy. Those
are different physical quantities. Presenting a Kohn-Sham eigenvalue against a
thin-film CV measurement as model error would be a category error, so the record
states the offset and its consistency instead.
