# Phase 8 Unified Comparison And Reuse Analysis

## Comparison contract

External ranking uses the identity-aligned protocol with 1,977 common rows,
including 999 OOD rows and 978 P8-hard rows. PCQM uses the paired 4,981-row
ETKDG-valid proxy. Lower MAE is better.

| Model | Passes | Common avg | OOD avg | P8-hard avg | PCQM Gap |
|---|---:|---:|---:|---:|---:|
| M01 routed-v4, targeted 500K | 3 | 0.103654 | 0.112783 | 0.094329 | **0.291691** |
| M07 best established 1M pure-2D ensemble | 4 | 0.100215 | 0.112358 | **0.087810** | 0.311851 |
| M25 repaired-2M retention-D seed42 | **1** | **0.100074** | **0.109555** | 0.090390 | 0.309216 |

M25 is best on common and OOD at one GPS pass. M07 remains best on P8-hard but
requires four GPS passes. M01 remains much better on PCQM and is still the
registered production model. M25 is not promoted until seeds 43 and 44 agree.

M26 control A is not in this table. Its model-specific random test is average
MAE `0.102431` and Gap MAE `0.121676`, but that split follows the exact-2M
mixture and is not comparable with the fixed external protocol. The database
records it as internal evidence rather than pretending it is an external win.

## What changed

The strongest evidence is distribution retention, not parameter count:

1. GPS7 parameter count stayed approximately fixed while the training mixture
   changed from 500K to 1M/2M.
2. Uniform scale-up improved model-specific random validation but diluted the
   targeted 500K objective.
3. Retention-B restored a 50% targeted replay share and improved common, OOD,
   and P8-hard.
4. Repaired data plus the same replay produced M25, improving all three general
   scopes again relative to retention-B.
5. PCQM moved in the opposite direction, confirming that it is a separate
   label/domain contract rather than a suitable global promotion gate.
6. The 1M ensemble's P8-hard advantage proves that useful complementary signal
   exists, but unconditional multi-expert inference is an expensive way to
   recover it.

## Optimization order

1. Complete M25 seeds 43 and 44. Do not add architecture variables before the
   controlled GPS7 result is stable.
2. If stable, train repaired-2M GPS9 and recompute out-of-fold per-target gain.
   Do not copy the old `Gap < 4` route.
3. Use M07 as a hard-region teacher. Train a bounded residual correction on top
   of M25, with an identity path and explicit inference-cost limit.
4. Keep PCQM as a task-level Gap specialist. Do not force one molecular Router
   to reconcile B3LYP HOMO/LUMO and PCQM Gap.
5. Revisit 3D fusion only after 3D is trained on the same repaired-2M identity
   manifest and split as 2D.

## Fine-tuning reuse

| Model | Reuse decision |
|---|---|
| M25 retention-D | Preferred future general warm start after the three-seed gate. |
| M21 retention-B | Stable fallback warm start and direct ablation baseline. |
| M07 1M ensemble | Teacher/oracle for P8-hard compression; not default serving. |
| M01 routed-v4 | Freeze as production and paired baseline; fine-tune only a copy. |
| M26 control A | Valid controlled warm start, but not an external-quality reference. |
| M08 coverage-2M | OOD/coverage specialist or teacher only. |
| M02 original 1M fusion | Reusable alignment reference, not a global replacement. |

The database's `reuse_mode` and `reuse_constraints` fields are authoritative
for programmatic filtering; detailed experiment decisions remain linked by
each evaluation's `source_path`.
