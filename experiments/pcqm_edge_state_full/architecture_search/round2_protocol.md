# Round 2 Field-Aware Input Protocol

Round 1 showed that additional depth and edge-state width did not improve the
fixed OGB-rich control and specifically worsened radical molecules. The OGB
categorical stem currently embeds each field at full model width and sums all
fields before message passing. Round 2 tests whether preserving each field in
its own channel before projection improves the rare radical signal.

## Candidates

| ID | Categorical stem | GPS body |
|---|---|---|
| `fieldconcat16` | 16 channels per field, concatenate, linear projection | GPS9-192, four heads, edge state 64 |
| `fieldconcat32` | 32 channels per field, concatenate, linear projection | GPS9-192, four heads, edge state 64 |

Every other input, split, schedule, target, seed-42 gate, and contamination
boundary is identical to `README.md`. The existing summed stem remains the
default in reusable code; only these packaged candidates enable
`concat_project`.

Only a seed-42 candidate that improves overall MAE by at least `0.002 eV`
while keeping radical and non-radical regression within `0.002 eV` may open
seeds 43 and 44. If neither passes, the field-aware stem branch closes without
an official full-data run.
