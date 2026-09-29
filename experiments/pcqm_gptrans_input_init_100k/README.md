# GPTrans-T input-embedding initialization screen

This desktop-owned experiment asks whether changing only the initial scale of
GPTrans-T's input embedding tables improves the fixed 100K/50K PCQM4Mv2 Gap
screen without adding parameters or forward operations.

- `protocol.md` owns the scientific question, comparison limits, and decision
  gates.
- `training_contract.json` owns the machine-readable intervention, V4 recipe
  identity, resource ceiling, and acceptance thresholds.
- `launch_decision.md` records why this one candidate screen was selected.
- The accepted reference and its immutable artifacts are owned by
  `../pcqm_gptrans_t_100k_v4/`; they are reused, not retrained here.
- The prospective trajectory, once published through the experiment CLI, owns
  this experiment's action, cost, role, and terminal evidence pointers.
- The archived `prepare_initial_identity.py` verifies the original V4 state
  artifact and derives the candidate's complete state hash without constructing
  a model.
- The launcher and model intervention are preserved in the archived experiment
  history; this desktop evidence copy does not carry the rejected implementation.

The accepted retry3 result is indexed below. It does not authorize another
remote run or use of an evaluation role.

The intervention is an initialization-scale test of this adapted GPTrans-T
implementation. It is not a claim that the entire implementation matches the
paper's atom or bond input semantics.

The first P100 planning route closed `NO_TRAIN` after Kaggle retired P100.
`decision_p100_no_train.md` owns that disposition. `protocol_t4.md` and
`launch_decision_t4.md` describe the T4x2 planning revision; neither is a
remote training result.

The later Kaggle3 single-arm T4 declaration closed `NO_TRAIN` before remote
submission (`decision_t4_single_no_train.md`). The user-directed Kaggle1 T4x2
route uses an unmodified GPTrans-T control and the input-initialization
candidate concurrently. `protocol_kaggle1_pair.md`,
`training_contract_kaggle1_pair.json`, and `launch_decision_kaggle1_pair.md`
own the revised paired comparison. The archived `run_pair.py` is its thin entrypoint;
`prospective_kaggle1_pair/` owns its per-arm canonical plans.

The first paired Kaggle1 attempt failed in candidate GPU preflight because its
runtime initialization hash differed from the locally derived hash. It trained
neither arm; `decision_kaggle1_pair_v1_infra.md` owns the observed failure and
retained startup evidence. The unpublished custom-launcher retry plan closed
`NO_TRAIN` in `decision_kaggle1_pair_retry2_no_train.md`. The next paired
source reuses `../pcqm_gptrans_pair_norm_100k/run_candidates.py` for two-T4
dispatch; the archived `run_pair.py` only extracts the frozen source and
selects its initialization profile.

The shared-runner retry is frozen in `experiment_spec_kaggle1_pair_retry3.json`
and `prospective_kaggle1_pair_retry3/`. Its Kaggle1 submission and latest dated
platform observations are recorded in `attempts/kaggle1_pair_retry3/`.
`decision_kaggle1_pair_retry3.md` owns the accepted negative 100K result;
`results_kaggle1_pair_retry3/` owns its paired comparison and per-arm mechanical
acceptance, and `attribution_kaggle1_pair_retry3.md` owns the failure-mode
interpretation.
