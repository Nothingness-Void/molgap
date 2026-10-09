# Reuse and custody

Only reviewed src/molgap/k1_frozen_inference.py and k1_bn_diagnostic.py are
integrated from D:/w/k1-consistency-500k, no history cherry-pick. Their owning
raw hashes are2984b4abccab3eeec2f59d763649bcba41b77fdf40b4d9c4fc88a80c8d00a3c5
and a2a373ad91d251141c9967ff5f9b3c15e6bd74e3a79ad97a81f572272a1a85b1.
The diagnostic module permits this explicit non-model driver in its import audit;
all frozen factory/source checks remain enforced. Existing BN helper is unchanged.

Reuse _frozen_imports/_verify_frozen_modules, _cpu_runtime, _load_graphs,
load_native500k_k1, predict_clean, recalibrated_batch_norm, hashing/atomic IO,
RML plan/finalizer and paired_bootstrap_mean. New behavior only adapts exact
retained selected buffers, adds train predictions, measures clean-fit gaps and
paired endpoint/arm changes. Tests are synthetic before authorized inference.
