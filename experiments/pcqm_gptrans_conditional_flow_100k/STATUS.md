# Status

Prospective trajectory frozen and submitted as one isolated Kaggle1 T4x2
kernel. Both candidate arms are configured as independent one-T4 processes;
the kernel is currently RUNNING and has no scientific result yet.

Source commit: `60a8c0799e08d186ca7f30d3f1293eb9a75fad7f`.
The first source package failed before training because the fixed graph cache
requires the historical `molgap.pcqm_wedge` deserialization module. It is
preserved as a raw failed attempt and will not be reused. The corrected source
dataset is `nothingnessvoid/molgap-gptrans-conditional-flow-source-v2`.
Kernel v1 failed in both preflights before training. A corrected v2 will use
the new source dataset and the same frozen scientific contract.

Do not submit a successor while this attempt is running. On completion or
failure, preserve raw output first, then run the frozen no-inference acceptance
against the GPTrans reference before making a scientific decision.
