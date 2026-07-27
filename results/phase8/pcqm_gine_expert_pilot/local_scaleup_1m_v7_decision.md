# PCQM GINE Local 1M Scale-up V7 Decision

Date: 2026-07-26

## Decision

Accept `local_scaleup_1m_v7_frozen_bn` as the strongest local
leaderboard-oriented PCQM Gap specialist candidate. It does not replace the
general HOMO/LUMO/Gap model and is not a leaderboard score.

The official test set and sealed 20K were not used. The production registry was
not changed.

## Fixed comparison

| Model | Train sample | Scaffold dev MAE (eV) | Fixed official-valid 5K MAE (eV) |
|---|---:|---:|---:|
| Accepted remote v5 | 250K | 0.191690 | 0.187320 |
| Local continuation v6 | 250K | 0.188671 | 0.185272 |
| Local scale-up v7 | 1M | **0.187982** | **0.184618** |

V7 improves fixed official-valid by `0.000654 eV` over v6 and `0.002702 eV`
over v5. The v7 checkpoint was selected only by scaffold development MAE;
official-valid was evaluated once after early stopping.

## Data and training contract

- The deterministic 1M sample contains every accepted v5 250K training row.
- Graph acceptance reconciles `1,004,986` valid graphs and `14` invalid rows.
- Split counts are `917,746` train, `82,240` scaffold dev, and `5,000`
  fixed official-valid.
- Architecture is the checkpoint-compatible virtual-node GINE:
  hidden size `256`, `5` layers, dropout `0.10`.
- Initialization is local v6 best; batch size is `512`, learning rate is
  `1e-5`, and early-stop patience is `6`.
- BatchNorm running statistics remain frozen at the accepted v6 values.
  Affine parameters and all other model weights remain trainable.
- Best is epoch `2`; training stopped after epoch `8`.

## Streaming correction

The first packed streaming trials updated BatchNorm running statistics from
source-ordered graph shards. This biased evaluation toward the final shards:

- `5e-5` reached dev `0.205157/0.197986 eV` at epochs 0/1.
- `1e-5` reached dev `0.272413 eV` at epoch 0.

Both trials were stopped before official-valid evaluation. Freezing only the
running statistics removed the failure: a 250K packed control remained at
`0.18888 eV` dev after one epoch, versus a `0.18868 eV` baseline.

## Artifacts

- Run: `results/phase8/pcqm_gine_expert_pilot/local_scaleup_1m_v7_frozen_bn/`
- Metrics: `local_scaleup_1m_v7_frozen_bn/metrics.json`
- Acceptance: `local_scaleup_1m_v7_frozen_bn/acceptance.json`
- Best checkpoint: `local_scaleup_1m_v7_frozen_bn/pcqm_gine_best.pt`
- Graph contract:
  `data/cache/phase8/pcqm_gine_1000000_nested_seed42_43/manifest.json`

Best checkpoint SHA256:
`9b971c1c9f770b80966af38107b30da21143c75b5567879a941173afb2d5955e`.

## Next gate

Package this checkpoint as a PCQM-only submission candidate. Do not inspect or
tune against official test labels. Keep v6 as the cheaper reproduction control.
