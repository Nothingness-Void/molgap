# Status

Prepared 2026-11-11 JST; not yet submitted. Run `pcqm-k1-muon-wd-500k-20261111-s43`, account `nothingnessvoid`, kernel `nothingnessvoid/molgap-k1-muon-wd-500k-20261111-s43`.

Planned: segment 1 pauses at step 187488 (pass 48). Segment 2 (pass 48 to 60, step 234360) is submitted by the desktop owner.

Continuation steps (follow `../pcqm_k1_read_batched_muon_500k_20261107/continuation_v2_20261108.md` on that branch):
1. `retrieve_family_outputs.py --all-arms --paused --run-id pcqm-k1-muon-wd-500k-20261111-s43 --attempt-id v1 --profile D:/molgap-machine-profile.json --repo-root . --execute`
2. `build-resume` per arm with the retrieved version directory as `--source-output`, then `prepare-resume`.
3. `submit_workflow.py` with `--prepared`, `--recovery-authority authority.json` (+ sha256), `--source-version 2`, `--kernel-version 2`, from this worktree.
4. After the successor ends: retrieve strict with `--continuation-receipt` and `--resume-prepared`, `draft-closure`, then `accept_run.py --strict`.
Recheck `check_quota.py` before the continuation.
