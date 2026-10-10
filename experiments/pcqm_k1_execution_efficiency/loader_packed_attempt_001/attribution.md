# Packed-input attribution

Previous list-only acceptance was insufficient to establish actual packed-input
compatibility. The two legacy readers defined dataset classes inside functions;
Windows spawn could not import those classes. Extracting their unchanged two-item
reader fixes importability without a graph, feature, target or training algorithm.
The registered source inventory/import gate now includes that exact source owner;
the package guard exception is exact-path only, not a relaxation for data assets.

Real TRAIN-only packed controls pass original-list batch hashes. Subsequent
retained windows avoid the large cold first-fetch wait, while first use remains
cold. This distinction prevents misattributing the whole cost to constructor
timing. Source changes remain on the execution-efficiency branch; active jobs
and accepted historical source/evidence are untouched.

Missing discriminator: GPU consumer and native full-source execution. The
[A100 study](../../pcqm_k1_colab_execution_profile/attribution.md) already found
small steady loader/H2D waits under its bounded condition, so this Windows result
does not establish CPU collation as the cause of A100/T4 parity.

Registered100K `_train_loader` has no explicit generator; constructing its native
iterator draws from global Torch RNG. Worker reuse can change that main-process
sequence as well as worker seeds. Current reuse rejects it rather than pretending
bitwise checkpoint/RNG parity. Main RNG/worker semantics must be qualified before
integrating reuse there; a separate generator is not a free equivalence fix.

No new training, scientific role, GPU budget or promotion is released. Native
500K owner integration also requires separately frozen source/runtime controls.
