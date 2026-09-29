# T4x2 resource amendment for the input initialization screen

`protocol.md` owns the unchanged scientific question, 15-table intervention,
historical V4 comparator, data and target identities, 60-epoch recipe, 3 meV
endpoint gate, and causal limits. Its P100-specific resource paragraph applies
only to the unsubmitted first declaration closed in `decision_p100_no_train.md`.

For this revision, request one Kaggle T4x2 allocation and expose exactly one
T4 to the candidate process. The second GPU is not used for duplicate reference
training; Kaggle offers T4x2 as an allocation. The remote preflight must verify
the visible device is a T4, run the same real-batch reference/candidate
alternating latency measurement, and require both training-step and inference
median ratios at most 1.05. Its candidate training estimate must be at most
six T4 device-hours. Keep T4 device-hours, wall/queue time, and CPU time
separate. A different accelerator or missing runtime evidence requires a new
declaration; no platform conversion from the historical P100 result is allowed.

This is local preparation only. No Kaggle kernel or GPU preflight has been run.
