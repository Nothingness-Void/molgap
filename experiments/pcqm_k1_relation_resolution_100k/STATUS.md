# Execution state

## Separate frozen-intervention diagnostic

On September 27, 2026 the user authorized one Kaggle2 NO_TRAIN diagnostic.
The [protocol](diagnostic/protocol.md) freezes four inference conditions per
accepted model on the two already consumed internal development roles.
Source/package validation precedes submission. No new training, coefficient
selection, scale-up, protected role or successor is authorized.

## Training and separate audit complete; exact routes closed

All three arms completed 40 epochs / 31,240 optimizer steps and passed strict
saved-artifact acceptance on September 27, 2026. All three native training
traces were finalized and included in the RML replay pool. The immutable K1
reference development MAE is 0.1413736343383789 eV; it was not retrained.

| Arm | Parameters | Best epoch (zero-based) | Dev MAE, eV | Gain vs K1, eV | Outcome |
|---|---:|---:|---:|---:|---|
| Receiver-pair | 3,681,665 | 38 | 0.1395080686 | 0.0018655657 | POSITIVE_BELOW_GATE |
| Triplet-aggregate | 3,683,843 | 39 | 0.1398579329 | 0.0015157014 | POSITIVE_BELOW_GATE |
| RRWP-pair | 3,681,953 | 36 | 0.1389245242 | 0.0024491102 | POSITIVE_BELOW_GATE |

All paired row-bootstrap intervals against K1 were favorable, but row bootstrap
does not estimate training stochasticity. None cleared this study's prospective
0.003 eV material gate. RRWP had the lowest observed score; it is not a confirmed
new incumbent or evidence of successful 500K/full training transfer.
Frozen per-arm authorities: [receiver](arms/receiver_pair/decision.md),
[triplet](arms/triplet_aggregate/decision.md), [RRWP](arms/rrwp_pair/decision.md).

| Slot | Actual scheduler identity | Observed allocation |
|---|---|---|
| dual | `kaseichou/molgap-k1-receiver-and-triplet-s42`, kernel 136015255, v1 | two independent T4 workers |
| rrwp | `kaseichou/molgap-k1-rrwp-pair-s42`, kernel 136015256, v1 | P100 requested; two T4 allocated, one used |

Cost records count observed allocated devices, including the unused second
RRWP T4 and setup time. These are resource-occupancy records, not a claim about
Kaggle's billing formula. No idle allocation is hidden as zero cost.

Release/source identity and the exact downloaded entry/metadata hashes are in
`submission_receipt_v1.json`. Scientific source commit:
`f2d794f953af437af3fab60b33d49720531c7195`; source archive SHA256:
`5a0e5f29b836d4e92dc7b52937f4ca05521777d434f9314de548d5be0120297f`.
The source dataset was private and ready; all five bootstrap files including
the unexpanded `source_payload.bin` were verified remotely before submission.

Kaggle generated the dual notebook slug from its title, adding `and`.
The frozen source's embedded trace/run identifier remains the predeclared
logical identity without `and`. The receipt binds that identity to actual
kernel **136015255/v1** with matching source and entry; nothing was resubmitted
or rewritten. Terminal provenance must retain this explicit alias and receipt,
not invent a second physical run. Monitoring uses only the actual slug.

## Acceptance and separate portability audit

The initial RRWP acceptance failure was a local consumer field-name mismatch,
not a training failure. The repair requires both actual producer checks and
rejects missing/false values; remote training artifacts were not changed.
See [interface diagnosis](acceptance_interface_diagnosis.md).

The prospectively registered NO_TRAIN audit was submitted only after all three
training arms passed acceptance. Kernel
`kaseichou/molgap-k1-relation-audit-s42`, numeric ID **136030465**, version **1**,
completed and passed independent saved-tensor acceptance. It reproduced every
candidate's original-development predictions, then inferred the fixed500K
internal-dev 50K rows. Already accepted K1 predictions were reused byte-for-byte.
All three candidates regressed against K1 on this role, before any change of
training scale. Paired intervals, ordered-row concentration and the limits of
causal attribution are retained in the [audit decision](audit/decision.md).
No optimizer, official validation, test-dev, test-challenge, or other protected
role was opened. The actual two-T4 allocation consumed 760.935 device-seconds,
including the unused device and setup, below the 5,400-second cap.

- Released checkpoint, reference prediction, scientific source and audit-helper
  hashes: [audit release](audit_release.json).
- Actual submission and downloaded entry identity:
  [audit receipt](audit_submission_receipt_v1.json).
- Three training trajectories remain replay-ready. The separate prospective
  audit completed terminal publication, RML validation and derived rebuild;
  its `NO_TRAIN` outcome does not add a training-prefix replay entry.
- Training/audit static checks and synthetic saved-record checks passed;
  RML `validate` and `check --frozen` passed. No local model construction or inference.
- `monitor_binding.json` and its stores retain the two closed training events;
  `audit_monitor_binding.json` binds only the new audit to the same A/B pair.
- Existing heartbeat `molgap-k1-conjugated-dual-kaggle2-monitor` is paused after
  Luna B's successful idempotent terminal handoff. Event
  `evt-f2d487371d32940d6be7d5a1` was claimed and closed by A. No model/reasoning
  override or additional controller conversation was used.
- No automatic scale-up, extra seed, training successor, or protected-role
  access is released by these results.
