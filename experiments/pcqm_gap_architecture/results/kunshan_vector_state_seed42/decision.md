# Kunshan persistent-VectorState seed-42 decision

Job `121002721` completed the frozen paired comparison on one Kunshan Hygon
DCU. Both models used the accepted official-train-derived 100K/10K geometry
cache, seed 42, FP32, batch 48, and the same optimizer and schedule. Official
PCQM validation and test-dev remained unread.

The GraphState9 control reached `0.1307169088 eV` internal-validation Gap MAE
at epoch 35 with 3,665,809 parameters and `300.10 graphs/s`. Persistent
VectorState16 reached `0.1301084455 eV` at epoch 39 with 3,696,209 parameters
and `265.11 graphs/s`. The paired change was `-0.0006084633 eV`, a 0.47%
relative MAE reduction, while throughput fell by 11.66%.

The first artifact acceptance exposed a provenance-only exporter defect: PyG
increments batched attributes whose names contain `index`, so `row_index` was
not a safe batch field. The accepted source cache contained 10,000 unique
validation identities. A no-inference repair rebuilt only the row identities
from the hashed cache, changed neither prediction nor target values, and the
repaired artifacts passed acceptance. The reusable runner now aliases dataset
identity to `row_id` before batching so future exports remain exact.

The numerical reduction is below the discovery plan's `0.001 eV` promotion
threshold and does not compensate for the measured slowdown. Persistent
VectorState is therefore a valid weak observation but not a promoted
architecture. Do not run seeds 43/44, full-data training, or official-role
evaluation for this mechanism. GraphState9 remains the frozen comparison
anchor; the next bounded question is K2 projected first/second-moment readout.
