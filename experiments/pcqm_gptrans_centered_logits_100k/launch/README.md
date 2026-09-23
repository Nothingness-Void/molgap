# Kaggle3 launch observation

The desktop agent pushed kernel
`nvoid912/molgap-gptrans-centered-logits-100k-s42-v1` once on 2026-09-24
(Asia/Tokyo). The Kaggle CLI returned `Kernel version 1 successfully pushed`.
The next status query returned `KernelWorkerStatus.RUNNING`. Kaggle listed the
last run at `2026-09-23T15:59:22.457000+00:00`.

`kaggle_observation.json` records the observed CLI facts and input dataset
references. `platform_response.json` is the caller-supplied canonical envelope
for the local receipt core. `1e6b96ff6b6e2bf04e75d9e67abd29a1d5eb25239f4b1a7eb4c4883c4a9e8fbf.json`
binds the accepted submission observation to the frozen spec and source
package. The shared receipt core does not submit jobs itself and does not
independently authenticate caller observations.

No runtime certificate, training metric, hardware cost, or scientific
acceptance is asserted here. Reconcile the same version against durable
Kaggle outputs before any retry or successor action. Retrieve only artifacts
required for replay-ready terminal acceptance.
