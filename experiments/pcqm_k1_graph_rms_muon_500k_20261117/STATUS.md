# Status

Prepared and segment 1 submitted 2026-11-17 JST on `codex/exp/k1-graph-rms-muon-500k-20261117`: `SUBMISSION_BOUND` (`kaseichou/molgap-k1-graph-rms-muon-500k-20261117-s42`, `kernel_id=145454564`, `version=1`, `KernelWorkerStatus.QUEUED`; planned pause at step 187488 / pass 48, followed by segment 2 continuation to step 234360 / pass 60).

## Segment 2 (2026-11-17 JST)

Segment 1 ended `STOP_FOR_COST` (wall cap) before the planned step-187488 pause and was retrieved with `--paused` to `D:/molgap-artifacts/retrieved/pcqm-k1-graph-rms-muon-500k-20261117-s42/v1/version1`. Arm cursors were unequal: `muon_bn` 185,743 steps, `muon_graph_rms` 165,999 steps. `build-resume` restores each arm from its last complete-epoch checkpoint: `muon_bn` step 183,582 (pass 47), `muon_graph_rms` step 164,052 (pass 42). Each arm therefore replays its own tail; recipe, seed, horizon (234,360) and `pause_after_step` 234,360 are unchanged.

Quota check before submit: kaseichou 21.05 h remaining (8.95 h used of 30). Authority allows 2 sessions, 18 h wall, 36 T4 device hours; segment 1 used about 8.95 h wall.

Continuation submitted: `SUBMISSION_BOUND`, `kaseichou/molgap-k1-graph-rms-muon-500k-20261117-s42`, `kernel_id=145454564`, `version=2`, status RUNNING at submit. Recovery authority: `authority.json` (sha256 `87c4d4ab3b72359aa1d7d2036494020db2222730ecb13cf150b787042a7cbb40`, as read at submit).
Receipts (outside Git, under `D:/molgap-artifacts/runs/pcqm-k1-graph-rms-muon-500k-20261117-s42/v1/`): `operations_segment2/` (journal `submission_state.json`, `launch_receipt.json`, quota records), resume plan `resume_segment2/resume_plan.json`, prepared `resume_segment2/prepared`.
Retrieval and `draft-closure` take `--continuation-receipt operations_segment2/launch_receipt.json --resume-prepared resume_segment2/prepared`; acceptance adds `--resume-authority`.

## Closed (2026-11-17 JST)

Both arms reached step 234,360. Live dev MAE: muon_bn 101.864 meV, muon_graph_rms 104.631 meV; effect +2.767 meV, 95% CI [+2.140, +3.369]. Candidate NEGATIVE_UNDER_CONTRACT; closure complete. See decision.md.
