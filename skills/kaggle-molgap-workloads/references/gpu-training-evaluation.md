# GPU training and evaluation

Use the owning experiment's scientific and resource contract to choose stages.
Separate graph preparation, preflight, training, continuation, embedding,
fusion, or evaluation only where their dependencies, resource needs, or Kaggle
time limit require it. Each remote stage must identify its exact published
inputs and preserve enough output for the next stage without relying on a
prior worker filesystem. Use CPU resources for parsing and graph construction
unless the owning contract supports a GPU implementation.

## Resume and artifact identity

A resumable training result needs its selected checkpoint and a continuation
checkpoint with the model, optimizer, scheduler, precision/scaler state, step
or epoch cursor, data/split identity, and RNG state where the trainer supports
it. Keep metrics, trace, and provenance outside the checkpoint as retained
artifacts. The next run must mount the checkpoint explicitly, verify the
expected keys and shapes, and report its actual resume cursor. A failed load
must not silently become fresh training.

Before using a metric or embedding, verify model-state compatibility, sample
identity and order, split and target definitions, finite values, and the
experiment's comparison gate. For 3D routes, preserve the approved geometry
relationship between training and inference; current generated-3D production
inference uses ETKDG. Access a protected or sealed evaluation role only when
the owning contract has released it.
