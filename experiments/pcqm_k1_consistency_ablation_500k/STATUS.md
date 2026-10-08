# Execution status

Kaggle1 V1 scheduler COMPLETE; both arms accepted as bounded STAGE_COMPLETE,
23/60 epochs, not scientific terminal completion. Exact kernel137483676/version1
and source/dataset/package/Spec identities match the frozen submission.

| Arm | Selected stage MAE, eV | Best epoch (1-based) | Resume epoch index |
|---|---:|---:|---:|
| k1_pretrained_mean2, coefficient0 | 0.109245628119 | 23 | 23 |
| k1_pretrained_consistency, coefficient0.1 | 0.110386952758 | 21 | 23 |

Both have89,838 optimizer steps and11,499,264 sample presentations. Required
selected/resume artifacts are retained and hash/cursor/runtime verified.
Training allocation17.389569095 T4-hours;34.610430905 remain under the52hour
training ceiling. No continuation was submitted in this acceptance.

Both prospective RML trajectories remain ACTIVE with measured stage costs;
neither is strict replay-ready. Complete37remaining epochs per arm and the
predeclared full50K/clean-BN analyses before scientific disposition.
The retained continuation adapter currently requires unneeded per-epoch prediction
files; reconcile that retention gap before preparation, preserving the original
manifest and scientific contract. See [accepted stage record](submission_v1/terminal_inspection/stage_acceptance.md)
and [return procedure](REMOTE_HANDOFF.md).

Frozen executable source commit:920fe036ef030dab3052f0245bdc650ce33b9203.
Spec:65fc71b0a622a8661df33b061915e0630dfb57abf5e01546cfc1803489f23c12.
Package:9da691f5e3fb4f406c9af9c50493c40f80c231a19dcf8f07fc0cc04628e9f4d0.
Archive SHA256:acb6c461482c4b73f392b18bd69096bb3c01205877ff5a3c7ab8fbb653af4374.
