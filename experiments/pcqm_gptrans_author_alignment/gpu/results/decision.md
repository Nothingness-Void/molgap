# G1/G2 saved-output terminal comparison

On 2026-10-01, Kaggle2 kernel 136543794 version 1 completed both independent
arms. The actual restored package, receipt, initialization bindings, native
60-observation traces, selected EMA predictions, checkpoints and chunk hashes
passed [saved-output acceptance](acceptance.json). No local model was loaded or
executed; analysis used only saved internal-development tensors.

| Arm | Development MAE (eV) | Gain vs reference (eV) | Parameters | Selected epoch |
|---|---:|---:|---:|---:|
| Reference | 0.1560144881 | — | 5,246,817 | 59 |
| Degree initialization scale | 0.1508848917 | 0.0051295964 | 5,246,817 | 59 |
| Chemical shortest-path mean | 0.1514874509 | 0.0045270371 | 5,246,817 | 59 |

Both endpoint gains exceeded this protocol's frozen 0.003 eV gate. The paired
row-bootstrap candidate-minus-reference 95% intervals were [-0.00624463,
-0.00404870] and [-0.00547804, -0.00352589] eV. These quantify row uncertainty,
not seed variability or a fresh independent evaluation role.

The initialization result supports investigating input conditioning without
adding parameters; the path result supports retained chemical information along
paths. Neither establishes an isolated causal claim under the full terminal
gate: the frozen reference has shared role/cost/acceptance pointers, incompatible
with distinct observed bindings. Its immutable qualification was not rewritten.
Both RML transactions closed as `PAIRED_ENDPOINT`; replay admission is blocked by
the reference's lack of canonical replay enrollment. See per-arm decisions and
comparison readiness under `../degree_scale/results/` and
`../path_bond_mean/results/`.

The early advantage shrank substantially: G1's matched EMA gain fell from
0.049858 at epoch 19 to 0.005130 at 59; G2 fell from 0.037112 to 0.004527.
Late EMA still improved and lagged live weights. The observed advantage may
include optimization/convergence effects, so it must not be extrapolated to
500K or full training. The two arms did not test their combination.

Post-hoc reference-error quintiles improved mostly in harder rows while the
easiest reference-error rows regressed. These target-derived bins are descriptive,
subject to selection/regression-to-the-mean effects, and not deployable routing.

Native allocation cost was 3.57619 wall hours and 7.15238 T4 device-hours. Each
arm was charged one allocated device's wall reservation, not both devices;
utilization-derived cost was unavailable. Official validation/test roles were
untouched. No successor, extra seed, scale bridge or full training was released.

Infrastructure repair and remaining qualification boundary:
[release review](../../../../docs/operations/RELEASE_FAST_PATH_REVIEW.md).
