# Opus ruling on the 500K serial chain, 2026-11-21 (before submit)

Author: Opus adjudicator (model Opus 5.5), acting on the pre-submit review that found three blockers (B1-B3).
Scope: the gates, budgets and admission of `serial_spec.json` in this directory. It changes no science rule of the
frozen 500K protocol (branch `codex/exp/demol-cin-500k-ims-20261120`, commit `a7c5f7d74`, `protocol.md` and `admission.md`).
Terms: RSS = resident memory of a process; OOM = out of memory; dev = the view's own 50K development rows; tail = `tail_holdout_v1`.

## Verdict

The chain is **admitted under an Opus amendment of ruling Q4**, not under Q4's literal text. Q4 as written can never pass on IMS.
Admission still needs two checks on submit day (Q2, Q3 below). F0 already passed: Δtail = −6.102 meV ≤ −2.0 (frozen Q1).

## (i) Q4 replacement: amended, memory fact accepted; no in-job memory gate

Q4 asked for an IMS log showing ≥ 64 GB at 16 cores, plus a local CPU RSS ≤ 0.5 × that.
The RCCS manual (jsub page) fixes host RAM at 1.875 GB per core with no `mem=` option, so 16 cores give 30 GB.
The 64 GB figure rested on a wrong premise and no IMS log can ever show it. Q4 also sized the old host cell list
(about 30 GB est.); the GPU-resident path has a different memory model.

Amended Q4 (one-time admission fact, not an in-job gate): keep Q4's ratio test, swap the impossible IMS figure for the manual's share.
- Measured locally on the real 500K build (train 500,000 + dev 50,000 rows, GPU-resident path, one process):
  peak working set 4,064 MiB, peak private commit 6,401 MiB, GPU peak 1,507 MiB; the whole one-pass chain peaked at 4.4 GiB.
- Test: private commit 6.4 GiB + about 1 GiB of CUDA library pages that Linux `ru_maxrss` may count = about 7.4 GiB.
- Limit: 0.5 × 30e9 bytes = 14.0 GiB. 7.4 ≤ 14.0: **passes**. GPU: 1.5 GiB of 80 GB.

Ruling on the in-job memory gate: **removed from J3 and J5.**
- The peak comes early. In the 500K memory check the build peak equals the whole-run peak. If J3 runs out of memory, it fails
  within about 5 minutes (the 312 s build), not hours in.
- Both bad outcomes end the run id. A false skip ends in exit 0 (SKIPPED_FOR_MEMORY is final); a failed J3 ends in exit 1. Neither resumes.
  So choose by likelihood. A gate at factor 2, margin 1.3, fraction 0.5 trips only when J1 peaks above 5.37 GiB. By the measured
  growth (1.8× from 100K to 500K) J3 would then sit near 10 GiB against a 28 GiB share: every trip would skip a run that fits.
- J5's gate read a DeMol item, which says nothing about GPT-T memory. J5 fact: on Linux it forks one loader worker and `ru_maxrss` reports
  only the largest process, so its local 7.0 GiB could reach about 14 GiB in total. Still under 30 GB.
- Cost of the rejected risk: if J3 does fail, J5 still runs (up to its 15,300 s budget) and J4 skips. J5 keeps value alone as the platform shift
  check (frozen J5 text: it also helps read Cutlass/GRIT 500K).

## (ii) J4 gate: −1.5 meV rejected; line set to 0.0 meV

Rule: J4 runs only if dev MAE(J3) − dev MAE(J5) ≤ 0.0 meV, so only when DeMol is no worse than the same-platform control.
Any dev_delta line also needs J3 and J5 COMPLETE, which is wanted.
- Why not −1.5: the frozen PASS needs each DeMol tail ≤ G − 1.5. A gate at −1.5 sits on that bar and inside the noise of a
  difference between two single-seed models (about √2 × 1.6 ≈ 2.3 meV). With a true gain of −2 meV it skips J4 about 40% of the time;
  at 0.0 about 20%. Skipping J4 caps the best outcome at INCONCLUSIVE, so the line must fire only where J3 alone makes PASS unlikely.
- Limits, stated once: the gate reads dev, the frozen rule reads tail (the job cannot score the tail); one seed each; J3 and J5 differ in
  architecture and init; both are best-of-60-pass dev selections. A skip leaves the transfer verdict INCONCLUSIVE under the frozen one-seed
  cap; a NEGATIVE closure would then need J4 in a separate, separately decided job. This gate is a cost rule, not a science verdict.

## (iii) Memory gate ratio

Moot: the gate is removed (see (i)). The review's fallback (factor 2, margin 1.3, fraction 0.5, about 5.37 GiB on J1) is rejected for the
reason given there.

## Budgets and resume (B3 and the time plan)

- Budgets (s): J1 7,200, J2 4,140, J3 28,800, J5 15,300, J4 7,200. Sum 62,640 ≤ 84,480 (86,280 wall − 1,800 reserve).
  J4's 7,200 is a start threshold, not a run estimate: J4 starts if 7,200 s remain, checkpoints every 512 steps, pauses at the wall
  (exit 2), and a resume job runs the rest.
- Only driver exit 2 can resume. The launcher writes `paused` and `resume_run` accepts only that. Exit 0 (`complete`) and exit 1 (`error`)
  are final for the run id; a new run id means a new queue wait and fresh J1 and J2 runs.
- A FAILED item is retried, and gate skips are judged again, only inside an exit-2 resume. Example: J3 FAILED → J5 still runs,
  J4 is SKIPPED_BY_GATE, the job exits 1, nothing resumes.
- At most 2 resume jobs. Each must pass the launcher's maintenance guard; the last must be submitted before 2026-12-06 03:00 JST
  (first Monday 2026-12-07 09:00 minus 1.25 × 24 h). The guard ignores queue wait.

## Step rate (one set of numbers)

- Local RTX 5060: at most 0.099 s/step for DeMol 500K, derived from the J3 one-pass rehearsal (699 s wall − 312 s build, over 3,906 steps,
  one dev pass and start-up inside). The rehearsal operator reports about 0.087 s/step steady state; the record holds no trace of it.
  The 0.136 s/step figure is a 50-step check with warm-up; 0.141 is the plan's figure. Neither is a steady rate.
- A100: not measured. Frozen estimate 60-110 ms per step; DeMol is launch-bound and may be no faster than the 5060.
  So J3 and J4 each need about 4-9 h for 234,360 steps.

## Submit-day preconditions (Q2, Q3; this ruling cannot connect to IMS)

- Q2: run `jobinfo -c` and `showlim -c` under `platforms/REMOTE_HANDOFF.md`. This chain counts as one job. Non-terminal MolGap chains
  (each preflight + train pair counts once) + 1 ≤ 4. Jobs 2077090-2077094 (old J1/J2) must be gone; if any is live or left outputs,
  stop and reconcile, do not submit (J1 and J2 would run twice).
- Q3: (remaining − hard line) ≥ 3 jobs × 24 h × 73.1 points per A100-h ≈ 5,263 points (one job plus two resumes). Use the measured
  73.1, not the draft's 76. Output unreadable → submit nothing.
- Then the 11-19 rule: Opus review of the final commit before `submit_from_windows.py`.

## Quick checks rerun on the amended spec (2026-11-21, local, `local_rehearsal.py early` and `launcher`, fresh work dir)

- Real spec + real 100K/500K mirror, `ogb` set to None, `PYTHONNOUSERSITE=1`: `validate_spec` [] and `check_inputs` [].
- Budgets [7200, 4140, 28800, 15300, 7200], sum 62,640, limit 84,480: pass. Exit-3 matrix: 14 of 14 broken inputs refused, no child.
- Launcher: real `run_spec.json` validates; `train.pbs` passes `bash -n` (select ncpus=16, ngpus=1, walltime 24:00:00, no `mem=`);
  args expand to `--wall-s 86280` and `--memory-limit-bytes 30000000000`; worker → driver → 20-step J2 child paused with exit 2.
- Not rerun: the long chain parts (A-D). They ran on the earlier spec (memory gates present, J4 line −1.5, J4 budget 28,800);
  the driver code is unchanged and this change only removes a gate kind and moves a number.
