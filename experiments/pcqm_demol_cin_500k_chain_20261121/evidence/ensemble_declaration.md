# Pre-declaration: DeMol + GPT-T equal-weight ensemble on tail_holdout_v1 (written 2026-11-26, before any ensemble tail read)

Candidate (one only): per-row mean of the saved tail predictions of J3 (DeMol-CIN 500K s42, selected checkpoint) and
J5 (GPT-T pair_update_norm 500K s44, platform control, selected checkpoint). Equal weights. No other pair, no weight search,
no third model in the primary result.

Inputs: `D:\molgap-artifacts\tail_eval_demol_ims_20261125\preds_j3\j3_tail_holdout_v1.pt` and `preds_j5\j5_tail_holdout_v1.pt`
(keys pred_ev, truth_ev; 128,606 rows). The two truth_ev arrays must be identical, else stop.
Each member's own tail score was already read once (J3 105.071, J5 107.308 meV). This is a new candidate, so the ensemble
is read once; its result is final for this question and is not repeated with other members or weights.

Measure: tail MAE of the mean prediction (meV); paired row bootstrap (2000 resamples, seed 42) of
(ensemble - J3), (ensemble - J5) and (ensemble - min member).

Rule, fixed now:
- KEEP the pairing if (ensemble - J3) <= -1.0 meV and the 95% CI upper bound of (ensemble - J3) < 0.
- Gain between -1.0 and 0 with CI upper < 0: weak; keep only as a descriptive note.
- Otherwise: the pairing adds nothing over DeMol alone; do not pair.
Dev-set averages (J3+J5 96.09, J3+J4 96.79, all three 95.49 meV) were computed on the same rows used to select the
checkpoints, so they are descriptive and optimistic; they do not decide anything.
Not read: official validation, test, sealed roles. No tuning on tail.
