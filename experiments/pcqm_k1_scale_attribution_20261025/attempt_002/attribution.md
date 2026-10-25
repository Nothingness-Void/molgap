# Tail mechanism attribution

The fixed experiment resolves the discriminator left open by the completed
tail run. TRAIN-only recalibration improves every development endpoint by
0.841–1.313 meV, but does not remove any tail-versus-initial regression. The
same five learned states and membership reproduce the retained outputs first.
This distinguishes a real BN output-state penalty from the remaining learned
function difference. It is not explained by loading a wrong checkpoint or
silently changing the normalizer.

Clean TRAIN error falls by 1.043–2.217 meV while development error increases by
1.118–1.655 meV. This is conditional fit/generalization divergence on the fixed
cohorts. It contradicts the proposed remedy of simply extending this tail to
fix inadequate fitting. It does not establish a universal K1 capacity ceiling
or prove that every longer schedule is harmful.

The tenfold WD increase does not retain its small selected advantage under
common Clean-BN, and its final clean endpoint is worse. The arithmetic change
in selected advantage is about .177 meV; it is not a causal percentage assigned
to training-time buffers. No additional WD grid is justified by this result.

The parent-to-tail change bundles single-forward loss, optimizer reset and LR
restart. The two tail arms isolate WD conditional on that bundle, not the other
changes. TRAIN metrics are in-sample for BN calibration; development selected
historical checkpoints. Row bootstrap is conditional, without seed replication
or multiplicity correction. Unique training-time causation remains unresolved.
The broader [scale report](../analysis.md) distinguishes these findings from
teacher omission, changed reference/cohort and historical selection-state effects.
