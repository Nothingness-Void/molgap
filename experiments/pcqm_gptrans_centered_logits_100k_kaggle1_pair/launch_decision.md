# Kaggle1 paired 100K release decision

Decision date: 2026-09-24 (Asia/Tokyo).

The user cancelled the Kaggle3 single-arm version and directed the desktop
agent to submit this experiment on Kaggle1. The user selected a GPTrans-T
baseline plus `centered_logits` dual-arm run. RML shows the frozen GPTrans-T
100K reference was accepted on Kaggle2, while the Kaggle3 candidate was
cancelled without accepted training. A matched Kaggle1 control tests whether
the candidate effect persists under a single platform allocation and gives
both T4 devices an independent, decision-relevant arm.

This releases one Kaggle1 T4x2 kernel after the gates in `protocol.md`. Both
arms must keep separate trajectories and artifacts. A failed preflight is an
execution result, not a scientific result. The accepted historical GPTrans-T
reference remains unchanged. No successor, seed expansion, protected-role use,
or scale-up is authorized by this decision.
