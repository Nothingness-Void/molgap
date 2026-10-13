# Execution Attribution

## What is supported

The added recovery guard/interface work preserves the measured numerical
endpoints on these synthetic graphs. Operation-count removal is not sufficient
to claim a faster session. GPTrans compute direction varies; K1 compute is
slightly slower in all three windows. No major compute speedup is established.

Under this source diff, K1's compute-only hot-path change is private layout H2D
transport; session/optimizer-participation changes are not exercised by the
no-persistence compute loop. The result does not support calling non-blocking
transport a speedup here, but it cannot isolate host scheduling/background GPU
effects or establish behavior for every graph shape/runtime.

Candidate sessions include staged selected/trace/checkpoint bundles, journal
hash checks, recovery validation and renames. Extra publication work is a
plausible contributor to their overhead. Its exclusive contribution was not
timed independently: checkpoint cursor, trace and layout also differ. Do not
claim a causal percentage from this endpoint comparison.

## Limits and disposition

Only32-step synthetic passes were used. Fixed per-pass protection is strongly
amplified relative to a real500K pass; saving every16 steps also differs from
many real128-step checkpoint configurations. Reduced cumulative metric copying
should matter more for a long pass, but that benefit remains unmeasured here.
Construction and final manifest inspection are outside both timed windows.

All generated chains are small (8--24 atoms), geometry-free and scratch-labeled.
This cannot diagnose generalization, determine MAE, qualify default Torch2.11,
or predict T4/A100/full-epoch speed. Three repeat ranges are not confidence
intervals; unchanged peak allocation is not unchanged total desktop VRAM.

Keep correctness repairs, preserve the non-positive speed evidence, and do not
relax durability or verification to force a favorable result. Larger performance
claims need a separately authorized representative long-pass/operator diagnostic,
not a silent rerun of this closed synthetic question. No new model route selected.
