# Round 3 Radical Context Protocol

Rounds 1 and 2 changed general capacity or the complete categorical stem; all
four candidates left non-radicals approximately flat but worsened radicals.
Round 3 tests a graph-level radical context while preserving the accepted
summed OGB stem and GPS9-192 body.

## Candidates

| ID | Added operation | Closed-shell behavior |
|---|---|---|
| `radicalctx16` | Sum 16-channel embeddings of nonzero radical-electron categories, project to 192, add through a learned gate | Exact zero residual |
| `radicalctx32` | Same with 32 context channels | Exact zero residual |

The radical category zero is a fixed padding vector and both projection layers
are bias-free. Thus the added path cannot change a closed-shell forward pass
before training and receives no closed-shell context during training.

Inputs, split, Gap supervision, 20-epoch warmup/cosine schedule, seed-42 gate,
and contamination boundaries remain identical to `README.md`. Only a seed-42
candidate that improves overall MAE by at least `0.002 eV` while keeping both
subset regressions within `0.002 eV` may open seeds 43 and 44. If neither
passes, this bounded architecture search closes; no additional candidate or
official full-data run is authorized by these screens.
