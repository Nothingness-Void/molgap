# PCQM Gap100K Local-Operator Search Seed-42 Decision

Decision date: 2026-08-28

## Question

Did any predeclared edge-aware local operator improve the frozen persistent-
EdgeState GPS9 comparator on the internal official-train-derived PCQM Gap100K
split under the fixed pure-2D, Gap-only contract?

## Frozen contract

- Kaggle2 kernel: `kaseichou/molgap-pcqm-gap100k-local-operators-seed42`,
  version 1, Tesla P100-PCIE-16GB.
- Runtime source commit:
  `bfaffe332d4c89dc041669679d6fb066e01bce1f`.
- Accepted cache aggregate SHA-256:
  `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`.
- Seed 42, FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay
  `1e-6`, at most 40 epochs, patience 8, nine GPS layers, width 192, four
  global-attention heads, mean pooling, and one direct Gap head.
- Strict comparator threshold: internal-validation Gap MAE below
  `0.13798263211250306 eV`.
- Official validation and test-dev roles were not read; no model inference was
  executed during acceptance.

## Acceptance

The terminal Kaggle output was downloaded under
`platforms/_records/kaggle/training/pcqm_gap100k_local_operators_seed42_v1/`.
The no-inference [`acceptance.json`](acceptance.json) passed with zero errors.
All three core candidates have complete metrics, traces, best models,
checkpoints, and validation payloads with matching recorded hashes. The
optional GATv2 candidate was not launched (`optional_launched=false`).

## Result

Throughput is the arithmetic mean of the 40 per-epoch trace values. Peak
memory is the preflight `torch.cuda.max_memory_reserved()` value. Comparator
delta is candidate MAE minus frozen-comparator MAE.

| Candidate | Parameters | Best epoch | Internal-validation Gap MAE | Training elapsed | Mean throughput (range) | Peak memory | Delta vs comparator |
|---|---:|---:|---:|---:|---:|---:|---:|
| OGB GatedGCN local GPS9 | 5,137,921 | 35 | 0.1404080225 eV | 4,972.659 s | 804.43 graphs/s (782.89–813.05) | 214 MiB | +0.0024253904 eV (+1.7578%) |
| OGB bond-conditioned TransformerConv local GPS9 | 4,479,553 | 39 | 0.1429827002 eV | 5,850.891 s | 683.67 graphs/s (675.10–688.66) | 204 MiB | +0.0050000680 eV (+3.6237%) |
| OGB GENConv local GPS9 | 4,142,611 | 38 | 0.1427755404 eV | 5,455.235 s | 733.53 graphs/s (665.81–750.17) | 234 MiB | +0.0047929083 eV (+3.4736%) |

The strongest completed candidate was GatedGCN local GPS9, but it remained
`0.0024253904 eV` above the frozen comparator. Therefore no candidate passed
the strict advancement gate.

The runner reports `16,501.3046 s` total elapsed time against the nominal
`14,400 s` search budget. The three mandatory candidates were allowed to
finish; the fourth was not launched, and no retry, extension, parameter
change, or additional GPU task occurred.

## Decision

The three completed local-operator mechanisms are closed as non-advancing
seed-42 architecture results. Persistent real-bond EdgeState GPS9 remains the
comparator. This result does not authorize seeds 43/44, full-data training,
official-validation evaluation, test-dev inference, or molecular-research-
server work. Any later candidate requires a separate materially new pure-2D
information-flow question and contract.

Compact numerical evidence is retained in [`summary.json`](summary.json) and
[`acceptance.json`](acceptance.json). Selection, preflight, metrics, traces,
artifact hashes, and the downloaded Kaggle log are retained under the platform
record. Large model, checkpoint, and validation-payload files remain local and
are not included in Git.
