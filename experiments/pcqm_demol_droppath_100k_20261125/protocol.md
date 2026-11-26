# DeMol-CIN 100K residual DropPath

Wave 1 combined DropPath with dropout, other regularization changes and a different execution path. Test stochastic depth alone on the existing FP32 DeMol-CIN. For each block and molecule, use one Bernoulli mask across atom, bond, ring and virtual residual updates, with inverse-survival scaling during training. Evaluate the same unmasked computation. The two arms use DropPath 0 and 0.10, no ordinary dropout, and the same .05 weight decay.

Use the same fixed 100K train and 50K internal development, FP32/no TF32, BS128, seed 42 as required by the shared paired declaration compiler, main EMA .999, 60 passes, 46,860 steps, and identical other recipe settings. Cap 6 T4x2 wall h including cleanup. The generated-3D owner path lacked a qualified sidecar acceptance binding. This pair uses pure topology and the standard strict lifecycle.

Report the pass-60 main EMA scalar endpoints from traces and paired row-bootstrap interval (2,000 draws) by scoring only the exact frozen development rows once from the retained endpoint EMA checkpoints via the existing scorer. Reproduce the trace MAEs and record source/checkpoint/role/row/target identity. Separately retain selected-checkpoint comparison and strict acceptance. Nominate only if pass-60 delta <= -3.0 meV, 95% upper < 0 and candidate give-back (pass-60 minus trace minimum) < 1.0 meV. Insufficient endpoint evidence is inconclusive. No protected role or automatic scale-up.

[Stochastic Depth](https://arxiv.org/abs/1603.09382)
