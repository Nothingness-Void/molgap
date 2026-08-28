# Round 4 Equal-Ensemble Confirmation

After Round 3 closed the single-model search, a read-only analysis aligned the
accepted seed-42 control, `radicalctx16`, and `radicalctx32` predictions. Their
pairwise residual correlations were `0.903-0.917`, leaving a material but
bounded ensemble opportunity.

The only candidate in this round is frozen before confirmation:

```text
prediction = (control + radicalctx16 + radicalctx32) / 3
```

No weights, calibration, Router, or row-specific selection may be fitted. The
seed-42 exploratory result was overall/radical/non-radical MAE
`0.143317/0.240734/0.135258 eV`, versus the matched control
`0.150062/0.253604/0.141497 eV`.

Seeds 43 and 44 are confirmation roles. Their accepted control checkpoints and
predictions already exist; only the two radical-context members may be trained
for each seed. Every run uses the same official-train-only 100K/10K graph
cache, fixed 20-epoch warmup/cosine schedule, and contamination boundaries.

The equal ensemble passes only if:

1. overall MAE improves versus the matched control in seeds 42, 43, and 44;
2. mean overall improvement across the three seeds is at least `0.002 eV`;
3. neither radical nor non-radical MAE regresses by more than `0.002 eV` in
   any seed.

Passing this confirmation records an ensemble architecture candidate. It does
not authorize official validation, official test, or full-data training.
