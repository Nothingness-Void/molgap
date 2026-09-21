# Decision

The corrected Kaggle1 T4x2 run completed both 60-epoch arms under the frozen
PCQM-100K contract. Mechanical acceptance passed: both arms used one visible
Tesla T4, FP32 with TF32 disabled, physical batch 128, deterministic preflight,
the expected parameter counts, and paired source indices and labels.

The scientific result is negative against the immutable GPTrans reference.
The reference development MAE was `0.1566272043 eV`. The conditional readback
arm reached `0.1609358453 eV` (delta `+0.0043086410 eV`), and the conditional
pair-recurrence arm reached `0.1598777551 eV` (delta `+0.0032505508 eV`). The
bootstrap intervals are also unfavorable. Neither arm reaches the required
`0.003 eV` improvement gate.

The trajectory is closed as `NEGATIVE_UNDER_CONTRACT`. Do not scale either
variant, add seeds, or combine them with K1 on the basis of this run. The
negative result rules out this specific conditional allocation mechanism under
the present GPTrans training contract; it does not invalidate the earlier
finding that both unconditional propagation paths are necessary.
