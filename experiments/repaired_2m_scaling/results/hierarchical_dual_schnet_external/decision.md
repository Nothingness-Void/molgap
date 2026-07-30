# Repaired-2M Same-Molecule External Comparison

## Decision

The repaired-2M pure-2D candidates passed the paired external comparison
against the frozen routed-v4 500K production model. The hierarchical dual-
SchNet residual heads failed external transfer and are rejected.

The comparison used identical `eval_set/CID/SMILES/target` rows. Of the fixed
1,977-row common set, 1,973 rows had a valid deterministic
ETKDGv3+MMFF200 conformer and were used by every method in the table. The four
ETKDG failures were excluded from all methods, not imputed selectively.

## Paired Results

Average MAE in eV:

| Method | All (1,973) | OOD (998) | P8-hard (975) |
|---|---:|---:|---:|
| Routed-v4 500K | 0.103580 | 0.112721 | 0.094222 |
| Repaired-2M GPS7/GPS9 equal | 0.098467 | 0.108798 | **0.087892** |
| Repaired-2M three-GPS dense | **0.097638** | **0.106655** | 0.088407 |
| Equal + dual SchNet residual | 0.121718 | 0.126702 | 0.116615 |
| Dense + dual SchNet residual | 0.121877 | 0.127216 | 0.116412 |

Paired average-MAE deltas versus routed-v4 500K:

| Candidate | All delta (95% CI) | OOD delta (95% CI) | P8-hard delta (95% CI) |
|---|---:|---:|---:|
| GPS7/GPS9 equal | -0.005113 `[-0.007516, -0.002762]` | -0.003923 `[-0.006984, -0.000987]` | **-0.006330** `[-0.010104, -0.002813]` |
| Three-GPS dense | **-0.005942** `[-0.008381, -0.003674]` | **-0.006066** `[-0.009031, -0.003243]` | -0.005815 `[-0.009745, -0.002251]` |

The dense candidate also improved all-set Gap MAE from `0.122549` to
`0.114202 eV` (`-0.008348 eV`, 95% CI
`[-0.012188, -0.004657]`). The equal candidate obtained the best P8-hard Gap
MAE, `0.101813 eV`, improving routed-v4 by `0.010525 eV`.

## Fusion Rejection

The external dual-SchNet residual did not merely lose to routed-v4. It
regressed against its own frozen 2D identity:

- Equal identity: `+0.023251 eV` average MAE on all rows.
- Dense identity: `+0.024239 eV` average MAE on all rows.
- Both paired 95% intervals were strictly above zero.

The residual head therefore learned an internal-distribution correction that
did not transfer to common/OOD/P8-hard. The accepted SchNet checkpoints remain
valid standalone assets; only this hierarchical residual use is rejected.

## Promotion Boundary

The production registry was not changed. The practical candidates for the
freeze decision are:

- accuracy mode: repaired-2M three-GPS dense;
- lower-cost mode: repaired-2M GPS7/GPS9 equal.

Any promotion record must include encoder-pass and latency accounting. No
further tuning may use these external labels, and the sealed 20K remains
unread.

Evidence:

- `metrics.json`
- `predictions.csv`
- `progress.json`
