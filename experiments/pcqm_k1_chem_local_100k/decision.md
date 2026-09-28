# Bounded K1 chemistry-local decision — 2026-09-28

The accepted Kaggle2 T4x2 seed-42 screen tested two equal-capacity, one-layer
chemistry-separated local updates against the frozen K1-v4 reference. The
prospective [protocol](protocol.md) was motivated by the [GPS++ full-depth
negative](../pcqm_k1_gpspp_local_100k/decision.md), [nonportable
relation-resolution audit](../pcqm_k1_relation_resolution_100k/audit/decision.md),
and [joint reconstruction nonpromotion](../pcqm_k1_joint_atom_reconstruction_100k/decision.md).
It tested an MMGNN-inspired adaptation, not a faithful reproduction.

| Model | Parameters | Best epoch | Internal-development Gap MAE | Gain vs K1 | Paired candidate-minus-K1 row-bootstrap 95% |
|---|---:|---:|---:|---:|---:|
| Frozen K1-v4 | 3,658,817 | 39 | 0.1413736343 eV | — | — |
| Atom-pair local color | 3,717,121 | 35 | 0.1438376755 eV | -0.0024640411 eV | [+0.001576879, +0.003383487] eV |
| Bond-type local color | 3,717,121 | 39 | 0.1429534107 eV | -0.0015797764 eV | [+0.000684922, +0.002492793] eV |

The saved-artifact acceptance recomputed all 50,000 aligned development rows,
verified completion/sha256 of both model, prediction and checkpoint assets,
matched the source, frozen 100K identity, FP32/BS128/40-epoch/31,240-step
contract, two independent T4 workers, native canonical traces and untouched
official validation/test roles. No local model inference or training was run.
Both arm intervals favored the frozen K1 on these paired rows. Row bootstrap
does not estimate seed variation, and repeated use of this internal role
limits external validity. Neither arm passed the frozen +0.003 eV nomination
gate; the separate fixed500K NO_TRAIN audit, extra seeds, 500K training, full
training and official evaluation were not released.

The mechanism was active: all real bonds had four colors, the zero-start
return exactly nested K1, candidate gradients and resume equivalence passed.
At epoch 39, normalized train MAE was 0.072665/0.074220 for atom-pair/bond
versus K1's 0.077263, while development MAE was worse. This supports an
added-capacity/overfit explanation rather than a failed optimizer or inactive
adapter; it does not uniquely establish the reason for generalization loss.
Bond-type grouping was numerically better than atom-pair grouping but still
lost to K1. This specific one-layer delayed-mixing hypothesis was closed.

Saved-prediction attribution exposed the familiar tradeoff rather than a
uniform loss. In post-hoc K1-error quintiles, the atom-pair/bond arms raised
the easiest 10,000 rows' MAE by +0.04427/+0.04213 eV, but lowered the hardest
10,000 rows' MAE by -0.04648/-0.04257 eV. Candidate wins occurred on
48.31%/49.35% of rows. These quintiles were defined using the observed K1
errors and true targets: they are descriptive and cannot be used as a
deployable router or as an independent validation gate. They do support
investigating why local chemistry capacity trades easy-row accuracy for
hard-row correction before authorizing any new architecture.

The run consumed 7,830.725 job-wall seconds and 15,661.451 summed allocated
T4-device seconds; account billing was not inferred. The separate wrong-slug
submission incident is retained in [launch_error_v1.json](launch_error_v1.json),
with unknown prior execution/cost rather than an assumed zero. Per-arm
acceptance, paired evidence, native allocation and RML terminal records are
under `arms/atom_pair_local/results/` and `arms/bond_type_local/results/`.
