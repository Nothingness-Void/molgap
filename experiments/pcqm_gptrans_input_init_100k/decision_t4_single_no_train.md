# Single-arm T4 declaration closed without training — 2026-09-29

The published Kaggle3 T4x2 declaration exposed one T4 to the initialization
candidate and left the other T4 idle. Before dataset publication or kernel
submission, the user requested a Kaggle1 dual-arm comparison. A fresh
same-allocation GPTrans-T control is now decision-relevant: it removes the
historical P100/T4 runtime confound and uses the second allocated device.

No remote job, GPU preflight, model training, development result, protected
role or T4 device-hour was consumed under the single-arm trajectory. Its
expected cost remains unmeasured, not zero. The exact declaration and package
remain retained as historical planning evidence.

Outcome: `NO_TRAIN`. The paired route is a separate prospective revision on
this experiment branch and follows `protocol_kaggle1_pair.md`.
