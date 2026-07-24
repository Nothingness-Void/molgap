# Exact-2M Pure-2D Coverage Expert

## Dataset

This experiment extends the accepted additive 1.5M table with exactly 500,000
rows selected from the strict 824,411-row acquisition inventory:

| New source family | Selected training rows | Future sealed rows |
|---|---:|---:|
| General overnight | 350,000 | 10,000 |
| Broad rounds 01-03 | 100,000 | 5,000 |
| Hard rounds 04-05 | 50,000 | 5,000 |
| Total | 500,000 | 20,000 |

The resulting training table contains exactly 2,000,000 rows. The current 10K
sealed set and every candidate sharing one of its scaffolds were excluded from
the new append. The future 20K contains 20,000 distinct scaffolds absent from
the original-1M and repair candidate-union scaffold caches. It is forbidden for
training, model selection, ensemble weighting, or Router tuning.

Artifacts:

- `assembly_report.json`: input files, counts, hashes, and invariants;
- `selection_audit.csv`: selected rows by source file and bucket;
- `future_sealed_20k.csv`: locally locked final acceptance set;
- `scaffold_cache/`: durable 25K-row scaffold chunks.

The selected top-up SHA256 is
`f14ec14643807f5bb53daf03e8b415d39e88282b63d7df8ea02f7b15891b88c0`.

## Training chain

SCNet jobs were dependency-gated and resumable:

| Stage | Job | Final state |
|---|---:|---|
| Validated 2M graph-cache reuse | preflight | complete |
| GPS7 coverage encoder | 700257 | complete |
| GPS9 coverage encoder | 700258 | complete |
| dual-GPS coverage head | 700259 | complete |
| development-only expert comparison | 700260 | failed before inference: ensemble baseline rejected |
| corrected development comparison | 702259 | complete |

The first durable graph shard and progress marker were verified at 50,000 /
500,000 rows. Large outputs live through
`results/phase8/multi2d_2m_scnet`, a symlink into the 400 GB team allocation.

## Gate

Development evaluation compares the incumbent `anchor + repair` equal average
against `anchor + coverage2m` and `anchor + repair + coverage2m`. It may use
common/OOD/P8-hard/PCQM and the prior 10K development set. Only after one formula
is selected and recorded may the future 20K be uploaded and opened once.

No 3D work or production registry change is authorized by this run. The final
result is specialist-positive but fails the global gate because P8-targeted-hard
regresses significantly. Keep the future 20K sealed. See `decision.md`.
