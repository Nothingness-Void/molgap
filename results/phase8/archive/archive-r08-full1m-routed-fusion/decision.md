# Full-1M Routed Dual-GPS Fusion

## Decision

**Closed negative for the fixed route.** The complete original-1M
GPS7+GPS9+SchNet fusion is valid, but copying the production `Gap < 4 eV`
route onto it makes it worse than always using the dual-GPS fusion head.
Do not replace production routed-v4 or reopen external/sealed evaluation for
this routing hypothesis.

This result does not reject the full-1M always-dual model. It reproduces that
model exactly enough to confirm its internal result, while the earlier common
and OOD evaluation still limits it to specialist status.

## Controlled Setup

- Original-1M aligned rows: 997,445.
- Fixed seed-42 split: 797,956 train / 99,744 validation / 99,745 test.
- Full-1M embeddings: GPS7 384 dimensions, GPS9 384 dimensions, and SchNet
  192 dimensions.
- Existing full-1M dual checkpoint: GPS7+GPS9+SchNet.
- Newly trained base head: GPS7+SchNet, with the same fusion implementation.
- Fixed production-style rule: use the dual head when the base predicts
  `Gap < 4 eV`.
- The future sealed 20K was not mounted or read.

## Results

| Variant | HOMO MAE | LUMO MAE | Gap MAE | Average MAE |
|---|---:|---:|---:|---:|
| Full-1M GPS7+SchNet base | 0.075847 | 0.073736 | 0.091931 | 0.080505 |
| Full-1M always-dual fusion | **0.074267** | **0.072370** | **0.089785** | **0.078807** |
| Full-1M fixed routed fusion | 0.075401 | 0.073356 | 0.091331 | 0.080029 |
| Molecule-level oracle | 0.070392 | 0.068966 | 0.083953 | 0.074437 |

Routed minus always-dual is `+0.001222 eV` average MAE and
`+0.001547 eV` Gap MAE. Paired molecule-level normal-approximation 95%
intervals are `[+0.001139, +0.001305] eV` and
`[+0.001411, +0.001682] eV`, respectively.

The existing always-dual reference was reproduced within
`-0.000000276 eV` average and `-0.000000589 eV` Gap, validating checkpoint,
split, label, and row alignment.

The atomic progress file retains `complete: false` because it is the resumable
epoch marker written during training. The separately written completion
manifest is authoritative and records all final outputs after evaluation.

## Why The 500K Topology Did Not Transfer

The fixed rule routes only 17,148 of 99,745 test molecules (17.2%) to the
stronger dual head. On these full-1M representations, the base head beats the
dual head for only 45.5% of molecules by average error and 42.9% by Gap error.
The route therefore leaves most rows on the globally weaker model.

A diagnostic threshold sweep approaches the always-dual result as more rows
are routed: average MAE is 0.080029 at 4 eV, 0.078930 at 6 eV, 0.078824 at
7 eV, and 0.078807 when every row uses the dual head. This is diagnostic only;
no threshold was selected on the test split.

The 500K gain came from a particular complementarity pattern between its base
and expert models. Scaling the encoders changed that pattern: more parameters
and data strengthened the dual model globally, so conditional execution no
longer supplies the old accuracy benefit.

## Artifacts

- Acceptance record: `acceptance.json`
- Local preflight: `preflight/`
- Immutable Kaggle download:
  `results/kaggle/evaluation/runs/molgap_1m_routed_dualgps_fusion_v2/`
- Reproduction payload:
  `scripts/phase8/archive/archive-r08-full1m-routed-fusion/`
- Kaggle kernel:
  `nothingnessvoid/molgap-1m-routed-dualgps-fusion`

No production registry entry or default checkpoint changed.
