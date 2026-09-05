# Official PCQM Geometry Warm-Start

This record covers a training-strategy experiment for the accepted
`ogb_distance_angle_triangle_edge_state_gps9` architecture. The architecture
was selected by the frozen 100K/10K three-seed screen. It is initialized from
the separately accepted official full-data OGB-rich EdgeState GPS9 epoch-30
checkpoint.

Warm-starting does not count as architecture-gain evidence. The final result
must be reported separately from the from-scratch 100K comparison. Official
test roles remain unopened; model selection uses only the official validation
split.

The remote chain is fail-closed: CPU smoke, 75 resumable geometry shards,
exhaustive cache acceptance, A100 mapping/forward/backward/timing preflight,
then at most 12 training epochs. Any failed predecessor prevents GPU training.

No result was available when this execution record was created.

Machine-readable execution contract: `protocol.json`.
IMS submission record: `launch.json`.

Two CPU-smoke infrastructure failures were repaired without changing the
scientific contract. The accepted recovery smoke mapped all 391 source tensors
and 4,771,073 pretrained parameters, reproduced the source function with zero
maximum absolute difference, and produced valid ETKDG geometry. Exact evidence
is in `infrastructure_recovery.json`; replacement job identities are in
`recovery_launch.json`.

## Terminal Result (2026-09-05)

The replacement CPU chain completed all 75 geometry shards. Exhaustive
acceptance covered 3,378,606 official-train rows and 73,545 official-validation
rows; 3,442,387 of 3,452,151 rows had valid geometry (99.717%), with 9,764
deterministic invalid-geometry records retained by identity. The accepted cache
used ETKDGv3 plus MMFF94s and did not read official test data.

A100 preflight job `1410316.ccpbs1` reached the frozen throughput gate and
projected 63,027 seconds (17.51 hours) for the contracted 12 epochs. This
exceeded the 12-hour ceiling. The preflight therefore exited nonzero and strict
`afterok` prevented training job `1410317.ccpbs1` from starting. No checkpoint,
validation MAE, or architecture-quality conclusion was produced.

The outcome rejects this exact full-scale training contract on runtime, not the
previously accepted architecture. Any reduced epoch count, larger batch,
different optimizer schedule, or altered geometry implementation would be a
new protocol rather than a retry. Machine-readable terminal evidence is in
`terminal_result.json`.

An explicitly authorized eight-epoch successor was submitted under the new
contract in `eight_epoch_protocol.json`. It reuses the accepted geometry cache
without rebuilding it and writes to an independent artifact namespace. Job and
monitor identities are recorded in `eight_epoch_launch.json`; this paragraph
records execution only and makes no result claim.

## Eight-Epoch Result (2026-09-05)

A100 preflight mapped every one of the 391 source tensors and preserved the
source function to a maximum absolute difference of `2.38e-7`. It projected
11.76 hours for eight epochs and passed the frozen runtime and memory gates.

Training job `1443247.ccpbs1` completed seven epochs and stopped after four
consecutive non-improving epochs. The best checkpoint was epoch 2 at
`0.115825 eV` official-validation Gap MAE. The exact source checkpoint scored
`0.099638 eV` on the same split, so the geometry warm-start regressed by
`0.016186 eV` (16.25%). The first trained epoch had already regressed to
`0.120730 eV`; later epochs never recovered the source performance. This
supports optimization-induced forgetting after geometry modules were enabled,
not an initialization or graph-alignment failure.

The selected checkpoint, final validation predictions, metrics, and completion
manifest passed hash checks. PBS emitted `Cgroup mem limit exceeded` after the
complete result JSON, while the final per-epoch progress marker remained stale
at epoch 6. Because all declared scientific artifacts had already been written
and verified, the artifacts were accepted with an infrastructure warning. The
local implementation now marks progress complete before writing the completion
manifest. The route was closed without a rerun; no official test role was read.

Machine-readable evidence is in `eight_epoch_result.json`.
