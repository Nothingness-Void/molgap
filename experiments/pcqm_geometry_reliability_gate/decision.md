# Geometry fusion screen decision, 2026-09-28

The hash-gated local screen passed its frozen exploratory gate. On the same
50,000 previously consumed internal-development rows, the accepted 2D model
scored 0.1176588802 eV and the accepted distance-plus-angle model scored
0.1141408510 eV. A fixed 50:50 prediction blend scored 0.1112922879 eV.
All five out-of-fold fits independently chose geometry weight 0.60 from the
frozen 0.05 grid; the pooled cross-fitted MAE was 0.1111033469 eV, improving
over the geometry component by 0.0030375042 eV. All five folds improved.
The descriptive [2,4) eV and >=8 eV tails improved by 18.970 and 12.973 meV
relative to the geometry component. Neither true target Gap nor residual error
was used to choose a weight for an inference row.

This is positive **prediction-complementarity screening**, not a new trained
model or independent generalization result. Both component checkpoints were
selected on this same development role. Cross-fitting the scalar weight does
not undo that reuse. The historical pair lacks observed V5 optimizer,
presentation, timing, and checkpoint trace identities; this screen cannot make
its source pair replay-ready. The original training protocol excluded fusion.

The result justifies examining a strict geometry candidate on a stronger
reference. A direct Kaggle submission of this 2D/geometry blend would merely
recompute an already consumed role. The accepted V4 500K K1 and GPTrans-T
scores, 0.104860 and 0.106868 eV, are contextual because their contracts are
not matched to this historical geometry pair. The older geometry GPTrans-T/K1
blend also lacks its matching V4 reference and runtime certificate. No
cross-contract model ranking, causal geometry gain, official-role evaluation,
or accelerator release is claimed here.

Before a training release, freeze a separate architecture-level question and
strict reference binding, verify a geometry-capable 100K/500K trainer and the
Kaggle mounted graph shape, and require real-shard/physical-batch preflight,
runtime certificate, actual cost cap, source package, and independently complete
RML acceptance for each new arm. Reuse the existing GPTrans, Kaggle, and RML
infrastructure; do not retrain a baseline merely for bookkeeping. Protected
official validation and test roles remain sealed.
