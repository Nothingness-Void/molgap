# PCQM Gap100K Recurrent Graph-State Seed-42 Decision

Decision date: 2026-08-28

## Question

Did a compact recurrent molecule state improve the frozen persistent-EdgeState
GPS9 comparator for direct Gap prediction on the internal official-train-derived
PCQM Gap100K split?

## Frozen contract

- Kaggle2 kernel: `kaseichou/molgap-pcqm-gap100k-recurrent-state-seed42`,
  version 1, Tesla P100-PCIE-16GB.
- Runtime source commit:
  `96ca9ba22a021fcdd8fdf8daecfe60fc0878c5c8`.
- Accepted cache aggregate SHA-256:
  `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`.
- Seed 42, FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay
  `1e-6`, at most 40 epochs, patience 8, nine GPS layers, width 192, four
  attention heads, mean pooling, and one direct Gap head.
- Parameter ceiling: `5,200,000`.
- Strict comparator threshold: internal-validation Gap MAE below
  `0.13798263211250306 eV`.
- Official validation and test-dev roles were not read; no model inference was
  executed during acceptance.

## Acceptance

The terminal Kaggle output was downloaded under
`platforms/_records/kaggle/training/pcqm_gap100k_recurrent_graph_state_seed42_v1/`.
The no-inference [`acceptance.json`](acceptance.json) passed with zero errors.
The source commit, cache SHA, contract fields, parameter ceiling, finite
preflight checks, and both sealed-data role flags matched. The best model,
checkpoint, validation payload, and trace hashes also matched their recorded
values. The launch-time identities are retained in
[`launch_manifest.json`](launch_manifest.json).

## Result

Throughput is the arithmetic mean of the 40 per-epoch trace values. Peak memory
is the preflight `torch.cuda.max_memory_reserved()` value. Comparator delta is
candidate MAE minus frozen-comparator MAE.

| Candidate | Parameters | Best epoch | Internal-validation Gap MAE | Mean throughput (range) | Peak memory | Training elapsed | Total elapsed | Delta vs comparator |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OGB recurrent graph-state GPS9 | 4,818,897 | 36 | 0.1386127572 eV | 600.1358 graphs/s (576.7501–618.3876) | 232 MiB | 6,666.7169 s | 6,848.4592 s | +0.0006301251 eV (+0.4566698%) |

The runner completed all 40 epochs. Total remote elapsed time was
`6,848.459237606 s` (1.90235 hours) against the nominal `14,400 s` budget.
The candidate remained `0.000630125069618209 eV` above the frozen comparator,
so it did not strictly improve it.

The declared artifact hashes are:

| Artifact | SHA-256 |
|---|---|
| `best_model.pt` | `c718dae4a71d80d8c169b764bae60f910c8da119e12a97c9d1927f8a611487b3` |
| `checkpoint.pt` | `efb0fe5e2120fd4e44672b58e0c78d060e4b30a206942d5eee3c0a0b09115966` |
| `validation_payload.pt` | `70e2f4ca4ca1e20803b0993999c1c9f104403bd072d158b3f25edefe51718598` |
| `trace.json` | `b15ee116309b39455b07221d4ccdff6e82682b1926a847266c1064d08c055322` |

## Decision

The recurrent graph-state mechanism is closed as a non-advancing seed-42
architecture result. Persistent real-bond EdgeState GPS9 remains the
comparator. This result does not authorize seeds 43/44, full-data training,
official-validation evaluation, test-dev inference, or molecular-research-
server work. A later candidate requires a separate materially new pure-2D
information-flow question and contract; no retry by seed, width, optimizer,
schedule, or initialization is authorized.

Compact numerical evidence is retained in [`summary.json`](summary.json) and
[`acceptance.json`](acceptance.json). Selection, preflight, metrics, trace,
artifact hashes, and the downloaded Kaggle log are retained under the platform
record. Large model, checkpoint, and validation-payload files remain local and
are not included in Git.
