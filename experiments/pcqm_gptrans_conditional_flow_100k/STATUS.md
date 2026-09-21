# Status

Prospective trajectory frozen and submitted as one isolated Kaggle1 T4x2
kernel. Both candidate arms are configured as independent one-T4 processes;
the kernel is currently RUNNING and has no scientific result yet.

Source commit v2: `d6ca0625be909388de9b2fce44b7db104379859e`.
The first source package failed before training because the fixed graph cache
requires the historical `molgap.pcqm_wedge` deserialization module. It is
preserved as a raw failed attempt and will not be reused. The corrected source
dataset is `nothingnessvoid/molgap-gptrans-conditional-flow-source-v2`.
Kernel v1 failed in both preflights before training because the source package
omitted `molgap.pcqm_wedge`. Kernel v2 used the corrected source dataset and
completed both 60-epoch arms under the same frozen scientific contract.

Mechanical acceptance passed, but both candidates regressed against the frozen
GPTrans reference. The trajectory is closed as `NEGATIVE_UNDER_CONTRACT`; see
`results/acceptance_v2.json` and `decision.md`. No full-scale release is
authorized.

Do not submit a successor while this attempt is running. On completion or
failure, preserve raw output first, then run the frozen no-inference acceptance
against the GPTrans reference before making a scientific decision.
