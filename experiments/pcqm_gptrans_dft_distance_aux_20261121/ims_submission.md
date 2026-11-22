# IMS submission record

Submitted 2026-11-21 (JST) from commit 75de92842b30835b484106496d0b70d7b469e4d4 (clean tree).

- Mode: serial chain, one training job, no preflight job.
- Run id: gptdft-chain-20261121
- Scheduler job: 2111727.ccpbs1 (queue H, 1 A100, 16 cores, 16 h limit). State at submit: Queue (gpu).
- Payload (keep until the run is closed): D:/molgap-artifacts/ims_payloads/gptdft-chain-20261121
- Sidecar on IMS: pcqm4mv2-fixed-v1/sidecars/pcqm-dft-train-geometry-500k-v3/ (manifest 5b1d8f47...; 11 files, sha256 equal to local, read-only).
- Reviews: Opus pre-submit review 1 (BLOCKED: init files not committed, bond-order-blind mapping), fixed; review 2 (BLOCKED: candidate resume untested), fixed; final verdict "SUBMIT: no blockers".
- Local rehearsals from git archive HEAD: full chain (local_rehearsal_record.json), candidate kill/resume d100 and d500 (local_rehearsal_resume_d*_record.json). All passed.

## Why a second job beside 2097505

Job 2097505 (DeMol chain) stays untouched. Group GPU limit is 6 (4 running,
15 queued at check time), so both jobs wait their turn; this job asks for one
A100 and about 1,220 points (16 cores x 16 h + 60 x 16 GPU-hours) against about
885K group points left. Waiting for 2097505 first would add a full queue wait
for no gain: the two jobs share no state. c100 and c500 repeat j2 and j5 of
2097505 byte for byte (recipe, init, seed); that time is booked here, and if
both run the pair is a determinism check.

## Items, in order

1. c100: GPT-T pair-update-norm 100K, s43 init, seed 43. Budget 4,140 s.
2. d100: same + train-only DFT distance head. Budget 5,900 s.
3. c500: GPT-T 500K control, s44 init, seed 42. Budget 15,300 s. Gate: d100 - c100 <= -3.0 meV (dev).
4. d500: candidate 500K, s44 init, seed 42. Budget 22,000 s. Same gate.

Total 47,340 s, about 13 h if both gates pass; about 2.8 h if the gate fails.

## Resume and retrieval

Only exit code 2 (paused) resumes: run the same submit command with the same
payload. Exit 0/1 is final for this run id. Retrieve with the repository tools,
not by hand. Decision rules: protocol.md. The in-job gate is a point estimate;
the paired bootstrap upper bound < 0 is checked after retrieval. Tail scores
(once per final checkpoint) only if 500K passes, booked in the separate
3D-supervised ledger.

## d500 resubmit (2026-11-23)

Job 2111727 finished c100, d100 (dev 0.13549 vs c100 0.13919, delta -3.70 meV, gate passed) and c500 (dev 0.10038), then skipped d500 for time: it needed about 22,000 s and 11,458 s were left. Run `gptdft-d500-20261123` runs d500 alone (s44 init, seed 42, budget 22,000 s) with a 24 h walltime, the adapter maximum. No gate: it passed in the first job. Specs: `serial_spec_d500.json`, `run_spec_d500.json`. Retrieval and decision rules are unchanged; compare with c500 from job 2111727.

Submitted 2026-11-23 from commit a828c35d9: job 2134319.ccpbs1, run gptdft-d500-20261123, queue H, 1 A100, 16 cores, 24 h. Payload: D:/molgap-artifacts/ims_payloads/gptdft-d500-20261123. Opus pre-submit review: SUBMIT, no blockers. Old run gptdft-chain-20261121 is closed for training: do not resume it (d500 moved here). Retrieval still needs the paired bootstrap upper bound < 0 on d100 vs c100.
