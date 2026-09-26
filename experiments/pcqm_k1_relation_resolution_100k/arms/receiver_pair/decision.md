# Receiver-pair seed42 terminal training decision

On September 27, 2026 kernel 136015255/v1 completed both isolated T4 workers.
The receiver-pair arm passed saved-artifact acceptance without model inference.
Its best internal-development MAE was **0.1395080686 eV** at epoch 38
(zero-based), versus K1 **0.1413736343 eV**: a **0.0018655658 eV** gain.
The paired row-bootstrap candidate-minus-reference 95% interval was
**[-0.0027391601, -0.0009729316] eV**. This was `POSITIVE_BELOW_GATE`, not a
promotion under the predeclared 0.003 eV policy. Reused development data and a
single training seed limit generalization/stochasticity claims.

The model had 3,681,665 parameters, retaining every K1 layer and adding one
receiver-resolved relation branch at layer 6. It completed all 31,240 optimizer
steps and 3,998,720 presentations with the immutable dataset, FP32/no-TF32,
BS128 and seed42. Native trace, roles, costs and checkpoints were retained.
The submission receipt explicitly binds the embedded logical run name to the
Kaggle-generated physical slug containing `and`; no second run was invented.

There was no extra training, multi-seed, full-scale or protected-role release.
The separately predeclared frozen500K internal-development inference audit
remained eligible after acceptance of all study arms; that audit is not 500K
training and cannot by itself establish scale robustness.
