# Operational state

Kaggle2 kernel `kaseichou/molgap-k1-edge-conditioned-slot-s42` version 1
terminated `ERROR` before candidate execution. The launcher requested exactly
one P100, but Kaggle allocated a Tesla T4 and the explicit compatibility guard
raised `RuntimeError: Candidate requires one P100, got Tesla T4`.

No training epoch, metric, checkpoint, or acceptance output was produced. This
is an infrastructure/resource-selection failure, not a scientific result for
the edge-conditioned slot architecture. The downloaded source archive and
terminal log are retained under the platform record directory; the terminal
handoff records the failure and its delivery state. No retry, repair, extra
seed, scale bridge, or official/test evaluation is authorized by this record.

Submission identities and the terminal fields are in `results/submission.json`.
