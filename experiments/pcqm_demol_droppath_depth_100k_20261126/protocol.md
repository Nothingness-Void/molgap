# Fixed-strength DropPath allocation in DeMol-CIN 100K

The accepted uniform 0.10 arm reduced pass-60 MAE by 6.238 meV versus no DropPath. This follow-up holds mean drop probability at 0.10 and compares the existing uniform schedule against eight evenly spaced layer probabilities from 0 to 0.20. Both expect 7.2 retained blocks. Keep the same per-molecule masks shared across states and one RNG draw per molecule and block in both arms. Enable all blocks at evaluation. The incumbent's selected prediction and terminal state are retained, but the standard workflow has no qualified reference bundle for that former candidate against this pass-60 gate. Use an exact same-job two-arm contrast.

Use fixed 100K train and 50K internal development only, FP32/no TF32, seed 42, BS128, 60 passes and 46,860 updates. Preserve initialization, shared DeMol architecture, AdamW, four-pass warmup, cosine to pass 60, primary EMA .999, auxiliary EMA .9999, and the six-hour T4x2 allocation.

Rescore each verified pass-60 EMA checkpoint once using the existing shared scorer on the exact internal development rows. Match native trace endpoints and compute aligned row-bootstrap 95% bounds (2,000 draws). Nominate only if candidate-minus-control <= -3.0 meV, 95% upper <0 and candidate native main EMA give-back <1.0 meV. Report the best-EMA comparison separately. Close through the same registered strict RML workflow.

[Predecessor decision](../pcqm_demol_droppath_100k_20261125/closure_v1/decision.md) · [Stochastic Depth](https://arxiv.org/pdf/1603.09382)
