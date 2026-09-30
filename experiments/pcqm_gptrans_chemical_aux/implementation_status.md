# Local preparation status

The shared wheel now supports the candidate loss through the existing GPTrans
preflight and training loop. No training loop was copied. The optional adapter
binds objective and label-cache identity to runtime calibration, checkpoint,
completion reuse and final records. Gap and auxiliary metrics stay separate.
Resumable state includes the head and EMA; exported best-model state does not.

The CPU cache builder retains immutable export/role pins, named descriptor
identity, labels hash, source indices, wall time and row failure reasons.
The loader rejects modified bytes, wrong roles, unknown/duplicate batch rows,
nonfinite descriptors and malformed arrays. Attachment happens on CPU before
GPU transfer. Cache identity is never inferred from a directory name.

The Spec registry supports chemical_aux/1 with typed parameters and exact
objective-digest binding. `run_arm.py` exposes preflight/train for each arm;
`build_labels.py` exposes CPU cache construction. Both CLI help paths execute.
Two physical GPU assignments and durable platform packaging still belong to
the Kaggle adapter and a future frozen paired launch.

Local verification: 146 checks passed across labels/cache/objective/Spec and
existing non-model V4 tests; after final Spec binding changes, 126 affected
tests and one dedicated addon-metadata test passed. Local tests use synthetic
data and tensor fixtures, not model execution. Python compilation and diff
whitespace checks passed. No hardware throughput or scientific result exists.

The 2026-09-30 submission preparation verified the accepted archive, fixed
graph shards and their source-index mapping, official train membership and
exact CSV prefix order. The narrow adapter exported only the frozen train
SMILES. The full label build then hit a confirmed parsing blocker, recorded
in `cache_decision.md`; it produced no accepted cache and released no GPU.

Remaining gates, in order:
1. Resolve the explicit chemical-label input policy described in
   `cache_decision.md`, then prospectively prepare and accept a complete CPU
   cache without silently changing frozen training membership.
2. Bind verified assets and frozen source to two Spec v2 arms and prospective
   same-run replay records, using the existing experiment CLI.
3. Package with the existing source builder and Kaggle adapter. Run actual
   T4 equivalence, resume and timing preflight; <=5% overhead is unmeasured.
4. Decide whether to release the bounded 100K pair. No 500K successor is implied.

The experiment remains unresolved, not submission-ready. The initial feasibility
record does not substitute for either training-arm trajectory. The independent
geometry continuation and its running source were not modified.

The GPTrans trainer can now emit the shared family epoch/selection/checkpoint
events without a copied training loop. The owning arm wrapper binds those
outputs to Spec/package/contract identities. Its short execution profile is
GPU-resident optimizer-step timing only, not end-to-end pipeline overhead;
actual CUDA validation and the frozen wall-time gate remain unmeasured.
