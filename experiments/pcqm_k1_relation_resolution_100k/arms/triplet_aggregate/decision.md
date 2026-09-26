# Triplet-aggregate seed42 terminal training decision

On September 27, 2026 kernel 136015255/v1 completed the triplet-aggregate arm
and its independent receiver-pair sibling. Saved-artifact acceptance passed
without model inference. Best internal-development MAE was **0.1398579329 eV**
at epoch 39 (zero-based), versus K1 **0.1413736343 eV**: a **0.0015157014 eV**
gain. The paired candidate-minus-reference 95% row-bootstrap interval was
**[-0.0024195643, -0.0006212131] eV**. This was `POSITIVE_BELOW_GATE`, not a
promotion under the prospectively frozen 0.003 eV gate.

The model had 3,683,843 parameters. Its inward vector triplet aggregation
passed the independent small-loop equation check, as well as initialization,
masking, graph-isolation, permutation, gradient/resume and memory preflight.
All forty epochs, 31,240 optimizer steps and 3,998,720 presentations completed.
The result was numerically worse than the receiver-pair parent, so this screen
did not show an incremental benefit from triplet aggregation; a scalar
difference alone does not establish a statistical negative mechanism claim.

Native trace, role truth, source/model/payload hashes and allocated notebook
cost were preserved with the exact physical-run alias receipt. No seed,
continuation, scale-up, protected-role access or new training successor was
released. Only the separately predeclared accepted-checkpoint NO_TRAIN audit
remained eligible once all training arms were accepted.
