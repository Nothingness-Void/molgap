# GPTrans centered-logits paired Kaggle1 screen

Desktop-owned 100K/50K V4 comparison of the unchanged GPTrans-T reference and
the parameter-free `centered_logits` candidate in one Kaggle1 T4x2 kernel. The
original Kaggle3 single-arm attempt was cancelled and has no scientific result;
its disposition is in `../pcqm_gptrans_centered_logits_100k/decision.md`.

`protocol.md` owns the comparison gate. `launch_decision.md` owns this one
paired submission. `training_contract.json`, `experiment_spec.json`, and the
per-arm prospective trajectories freeze the identities. `run_pair.py` unpacks
the frozen source archive and selects a fixed profile of the existing GPTrans
T4x2 runner. `STATUS.md` tracks the submitted job; `release_gate.json` and
`launch/` retain the local preflight and Kaggle submission observations.
