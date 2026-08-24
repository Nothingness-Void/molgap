# Resource-Bounded Architecture Screen on SCNet

This adapter runs the accepted PubChemQC 100K split as a controlled six-task
array: GPS9-192 and RWSE Structural GPS9-192 for seeds 42, 43, and 44.

Execution order:

1. Stage and hash-check the base graph, RWSE graph, split CSV, and source tree.
2. Submit `preflight.slurm` on one DCU.
3. Submit `train_array.slurm` with an `afterok` dependency on the preflight.
4. Each task writes an atomic epoch checkpoint, best model, metrics, test
   predictions, and independently validated `completion_manifest.json`.

The array does not train geometry-biased GPS, evaluate external sets, or modify
the model registry. Live job IDs and output paths belong in the owning
experiment status record, not in this adapter.
