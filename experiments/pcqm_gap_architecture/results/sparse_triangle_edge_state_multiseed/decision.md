# Sparse Triangle EdgeState GPS9 Multiseed Decision

Decision date: 2026-08-30

## Question

Did the seed-42 Sparse Triangle improvement reproduce in fresh paired runs at
seeds 43 and 44 under the unchanged official-train-derived PCQM Gap100K
contract?

## Acceptance

Kaggle2 kernel `kaseichou/molgap-pcqm-triangle-r3-confirm-s43-s44`, version 1,
completed. The downloaded models, checkpoints, validation payloads, traces,
metrics, and selection file passed the dedicated no-inference acceptance. All
recorded hashes match the retained artifacts. Official validation and test-dev
were not read.

## Result

| Seed | EdgeState GPS9 | Sparse Triangle GPS9 | Candidate minus comparator |
|---:|---:|---:|---:|
| 42 | 0.1379826321 eV | 0.1379017737 eV | -0.0000808584 eV |
| 43 | 0.1376674175 eV | 0.1372336000 eV | -0.0004338175 eV |
| 44 | 0.1379338056 eV | 0.1371266544 eV | -0.0008071512 eV |
| Mean | 0.1378612851 eV | **0.1374206760 eV** | **-0.0004406090 eV** |

Sparse Triangle improved every paired seed and the three-seed mean. It uses
4,878,257 parameters versus 4,771,073 for EdgeState GPS9. On the seed-43/44
confirmation task its mean training throughput was approximately 504.91 versus
656.03 graphs/s, so the accuracy result carries a material throughput cost.

## Decision

Sparse Triangle passed the predeclared multiseed gate and is the accepted
pure-2D architecture winner of this comparison. This result authorizes it as a
frozen comparator for materially new 100K architecture questions; it does not
authorize full-data training, official validation/test-dev evaluation, or
molecular-research-server work.

## Evidence

- No-inference acceptance: `acceptance.json`
- Compact arithmetic: `summary.json`
- Remote identity: `launch_manifest.json`
- Downloaded immutable output:
  `platforms/_records/kaggle/training/pcqm_gap100k_sparse_triangle_edge_state_r3_multiseed/`
