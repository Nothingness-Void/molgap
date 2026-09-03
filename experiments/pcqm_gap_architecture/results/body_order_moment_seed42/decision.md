# Body-order moment seed-42 decision

## Decision

The exact compact Cartesian invariant body-order mechanism is closed as a
scientific failure. It does not authorize confirmation seeds, full-data
training, official-role evaluation, or molecular-research-server use.

## Matched evidence

| Model | Parameters | Best epoch | Internal-validation Gap MAE (eV) | Throughput (graphs/s) |
|---|---:|---:|---:|---:|
| Fresh GraphState9 baseline | 3,665,809 | 39 | 0.1294583529 | 566.55 |
| GraphState9 + body-order moment | 3,681,329 | 39 | 0.1298706830 | 564.57 |

The candidate is worse by `0.0004123300 eV`, adds 15,520 parameters, and
retains 99.65% of baseline throughput. Both models completed 40 epochs on
separate T4 devices, row and target hashes match, and frozen no-model
acceptance passed.

## Diagnosis

The body-order branch reaches a lower final training MAE than the baseline
(`0.1129584388` versus `0.1139767716 eV`) while producing a worse validation
MAE. The compact scalar, vector-norm, and rank-2 invariant moments therefore
add fitting capacity but no useful generalization bias on this split. Because
the candidate and control use the same accepted ETKDG geometry cache and the
throughput change is negligible, the failure is not attributable to missing
geometry, insufficient runtime, or compute pressure.

The result closes this exact one-shot pre-message body-order injection. It does
not by itself reject all equivariant or many-body architectures, but width,
radial-count, seed, optimizer, and schedule variants would be retuning the same
failed mechanism and are not pursued.

## Provenance

- Protocol: `../../body_order_moment_seed42_protocol.md`
- Launch: `account1_v1_launch_manifest.json`
- Compact data: `summary.json`
- Local remote record:
  `platforms/_records/kaggle/training/pcqm_gap100k_body_order_graphstate_seed42_account1_v1`
- Source commit: `9625db4237584efc6cafe98432b090370af4c8c8`
- Geometry-cache SHA-256:
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`
