# IMS submission record

Submitted 2026-11-21 (JST) from commit 7e404a7fae5b1c021deccadfea69991a082098b9.

- Mode: serial chain, one training job, no preflight job.
- Run id: demol500k-chain-20261121
- Scheduler job: 2097505.ccpbs1 (queue H, 1 A100, 16 cores, 24 h limit)
- Payload (keep until the run is closed): D:/molgap-artifacts/ims_payloads/demol500k-chain-20261121
- Opus reviews: pre-submit review, ruling (opus_ruling_20261121.md), final check.
- Queue state at submit: job showed Queue with reason "other"; group GPU limit 6, 2 running.
- Read-only checks before submit: jobinfo -c (no job of ours), showlim -c (group 1,601,362 of 2,500,000 points).

Items, in order: J1 DeMol-CIN 100K s43, J2 GPT-T 100K s43, J3 DeMol-CIN 500K s42
(gate dev delta <= -3.0 meV), J5 GPT-T 500K control (same gate), J4 DeMol-CIN 500K s43
(gate J3 - J5 <= 0.0 meV, budget 7200 s, resume job finishes it).

Resume rule: only exit code 2 (paused) can resume, with the same command and the same payload.
Pull-back and closure: see protocol.md. Retrieve with the repository retrieval tool, not by hand.
