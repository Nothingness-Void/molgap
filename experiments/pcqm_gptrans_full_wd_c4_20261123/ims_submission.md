# C4 IMS submission

Submitted 2026-11-23 (server clock) from the remote root `/lustre/home/users/sm2/chou/molgap-gptrans-c4-wd022-s43-20261123`: job `2142970.ccpbs1`, name GPTWD43, queue H, 1 node, 8 cpus, 1 A100, walltime 12 h (WALL_S 43200). Command, from the root: `jsub -q H code/platforms/ims/gptrans_minus_tail/job_c4_wd022_s43.pbs`. Source commit fc6cdc2e7 (protocol note 005c2ca13). Opus pre-submit review: SUBMIT, no blockers.

Identity: runner sha `120d5571593c56647afe8f7c6f6060afeb7781ff8aea101fb9454fdac0ee5b37`; launcher `121525c568db4f68268d35abca890f4a5920538f234ab187ef44c1ca7e557569`; PBS `57d5ac4636a046a460c3c3c3272b5fb5e41251f6598846866240337e1d220b8f`; s43 init file `2ccd1eb7...5f3c`.

Run dir: `runs/gptrans-b1024-wd022-minus-tail-s43-20261123/` (look for `training_complete.json`). Expected 8.4-8.6 h of run time after the queue.

Recovery: if walltime cuts it, check `jobinfo -c` shows no copy, then resubmit the same command; the runner resumes from `last_checkpoint.pt`. Never rename the run dir. Retrieve the final EMA weights, score tail once (protocol.md), then write decision.md.
