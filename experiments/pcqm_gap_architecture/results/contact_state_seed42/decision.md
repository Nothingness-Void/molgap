# ContactState seed-42 decision

## Decision

The exact non-covalent ContactState mechanism is closed as a scientific
failure. It does not authorize confirmation seeds, full-data training,
official-role evaluation, or molecular-research-server use.

## Matched evidence

| Model | Parameters | Best epoch | Internal-validation Gap MAE (eV) | Throughput (graphs/s) |
|---|---:|---:|---:|---:|
| Fresh GraphState9 baseline | 3,665,809 | 35 | 0.1292508543 | 510.51 |
| GraphState9 + ContactState32 | 3,700,321 | 39 | 0.1311521530 | 456.56 |

The candidate is worse by `0.0019012988 eV`, adds 34,512 parameters, and
retains only 89.43% of baseline throughput. Row and target hashes match, both
models completed 40 epochs, and frozen no-model acceptance passed.

## Diagnosis

The contact branch lowers training MAE at its best epoch
(`0.1128097359` versus `0.1155950059 eV`) while worsening validation MAE.
Thus it adds fitting capacity without a useful generalization bias. A single
ETKDG conformer and a hard 5 A contact rule introduce many geometry-dependent
relations that are not reliably tied to the electronic Gap target; four
recurrent contact exchanges also oversmooth or amplify those noisy relations.
The result is not explained by missing optimization time: the candidate's best
validation score is its final epoch under the same complete schedule, yet it
remains clearly behind the matched baseline.

Per the frozen protocol, cutoff, width, exchange-depth, seed, and optimizer
variants are not pursued. Future geometry work must change the representation,
not retune this contact graph.

## Provenance

- Protocol: `../../contact_state_gpu_seed42_protocol.md`
- Launch: `gpu_launch_manifest.json`
- Compact data: `summary.json`
- Local remote record:
  `platforms/_records/kaggle/training/pcqm_gap100k_contact_graphstate_seed42_v1`
- Source commit: `a28862c3d91590b0827c3cfd6d7d2586a9c6ab47`
- Contact-cache SHA-256:
  `49725b92c2c0d33e17633abf8ffa7148ebc8bc9721d3e5b3635f1309891bc826`

