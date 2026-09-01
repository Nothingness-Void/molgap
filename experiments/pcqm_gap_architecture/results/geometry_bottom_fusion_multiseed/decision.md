# Geometry Bottom-Fusion Paired Multiseed Decision

Decision date: 2026-08-31

## Question

Did deterministic single-conformer ETKDGv3+MMFF94s distance-plus-angle
bottom fusion reproduce against freshly trained pure-2D Sparse Triangle
comparators at seeds 43 and 44?

## Acceptance

The downloaded immutable output was retained at
`C:\Users\Adminn\Documents\molgap\platforms\_records\kaggle\training\pcqm_gap100k_geometry_bottom_fusion_multiseed_v1`.
The repository acceptance script passed against its nested result root. The
selection, preflight, progress, metrics, traces, best models, resumable
checkpoints, and validation payloads were complete and their recorded hashes
matched. No model inference was executed during acceptance. Official
validation and test-dev roles were not read.

## Result

| Seed | Sparse Triangle comparator | Distance + angle candidate | Candidate minus comparator |
|---:|---:|---:|---:|
| 42 | 0.13790177369117737 eV | 0.13559719920158386 eV | -0.0023045744895935083 eV |
| 43 | 0.1386183649301529 eV | 0.13639135658740997 eV | -0.00222700834274292 eV |
| 44 | 0.1376376748085022 eV | 0.13758981227874756 eV | -4.786252975463867e-05 eV |
| Mean | 0.13805260447661083 eV | **0.13652612268924713 eV** | **-0.0015264817873636982 eV** |

All three paired seed deltas are negative and the arithmetic mean is lower.
The seed-44 improvement is marginal, so the result carries a stability
caveat even though it passes the predeclared strict gate.

## Decision

`ogb_distance_angle_triangle_edge_state_gps9` is the accepted geometry
candidate and is frozen as the 100K comparator for one separately contracted
seed-42 sparse torsion-state question. This result does not authorize full
training, official validation/test-dev access, desktop submission, or
molecular-research-server work.

## Evidence

- No-inference acceptance: [`acceptance.json`](acceptance.json)
- Compact arithmetic: [`summary.json`](summary.json)
- Remote identity: [`launch_manifest.json`](launch_manifest.json)
