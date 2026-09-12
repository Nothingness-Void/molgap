# Status

Attempt 2 completed and passed no-inference acceptance on 2026-09-13, but was
scientifically negative. Four channel-wise atom-selection heads produced
development Gap MAE 0.1432285458 eV versus the frozen K1-v4 reference
0.1413736343 eV; the paired bootstrap interval was entirely unfavorable. The
mechanism is closed without another seed or scale-up. See `decision.md` and
`results/attempt2_summary.json`.

Attempt 3 is limited to one molecule-conditioned query and one atom
distribution. No local training or sealed-role access occurred.
