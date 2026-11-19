# DeMol-CIN replication on IMS (J1 DeMol s43, J2 GPT-T s43), 2026-11-20 evening, frozen before any submit

## RML review (2026-11-20, before any run)
- Desktop RML (`cards --term demol|cellular|ring`): no card. DeMol-CIN lives only on `codex/exp/demol-cin-100k-20261118`.
- That branch's RML (`cards --term demol`): `T-pcqm-demol-cin-100k-20261118-r2-s42-demol_cin` is **INCONCLUSIVE** (`discriminator_missing`: strict block, runtime-certificate scope mismatch); its same-job control is CLOSED; the first attempt (`-s42-`) arms are ACTIVE (superseded by r2). No arm is closed as NEGATIVE_UNDER_CONTRACT or DUPLICATE_EVIDENCE.
- Closed neighbours (plan section 1): norm_ema, chemical_aux, width_500k, levers, late give-back A. None tested a cellular-complex model on a second seed or at 500K. So this question is new and decision-relevant: seed replication, tail check, platform transfer. No closed duplicate; no STOP.
- Status of the lead: one seed, dev-selected -4.80 meV at 100K (best of 60 passes on the same dev rows; final pass +0.00 to -1 meV). Single-seed evidence is weak. Nothing here is strict. Every IMS verdict is a non-strict nomination (Cutlass/GRIT precedent).

## Jobs
- J1 `pcqm-demol-cin-100k-ims-20261120-s43`: DeMol-CIN, 100K, seed 43, ncpus 8, preflight 00:45:00, train 06:00:00, `--wall-s 19800`; estimate 1.0-1.8 A100-h.
- J2 `pcqm-gptrans-pun-100k-ims-20261120-s43`: GPT-T pair_update_norm, 100K, seed 43, `--initial-state init/s43.pt` (500K cohort init s43, state sha c95da412), train 04:00:00, `--wall-s 12600`; estimate 0.6-1.5 A100-h. Hours are estimates, not measurements.
- Recipe diff against the r2 Kaggle recipes (`codex/exp/demol-cin-100k-20261118`, generated/2e8b995a...): seed 42->43; torch_version 2.11.0->2.7.1 (recipe and execution_policy); manifest `pcqm_fixed_100k_v1/...` (1b0e8fd5) -> `subsets/ogb-train-100k/manifest.json` (fed575a4); `auxiliary_ema_decays` and `execution_policy.auxiliary_ema` removed; `initialization_sha256` a65034da -> f504fa4e (J1 placeholder, equals the Cutlass 500K value) / c95da412 (J2, s43 state); inert Kaggle-only key `resume_chunks` removed (only `family_training_runner` reads it). Roles unchanged: train rows 0..99,999, dev 100,000..149,999. Role identities rebound to the IMS driver's own check: `train_role_identity` a4c655c3 -> baa5f49f (= `tensor_digest(arange(0,100000), role=source_idx)`; the same method gives the Cutlass 500K value be4b3c53), `development_role_identity` b7d8bc6d -> 3270b300 (digest of dev rows 100000..149999, equal to `acceptance_requirements.source_idx_sha256`). The Kaggle values failed the driver's check in the local dry run.
- Driver: `platforms/ims/family_recipe_run.py` after infra commits 3169a41c8 and 88a3e64af (dev MAE in `train_log.jsonl`, `selected_eval.json`, auxiliary EMA own file). Init transport: `init/s43.pt` is committed with `git add -f` (ruling Q6).


## Preregistered rules (verbatim from the weekend plan, section 2: F0, J1, J2)

### Tonight, before any submit: F0, a local tail falsifier (not an IMS job)
- Score the two existing 100K s42 selected checkpoints (DeMol, pair_update_norm, same Kaggle
  job) on tail_holdout_v1 once each. Also score dev 100000:150000 with `--expect-dev-mae`
  0.134635 / 0.139440 as the scorer check.
- Why: the -4.80 was picked as the best of 60 passes on the same dev rows. The tail has no
  selection bias. This is the cheapest falsifier of the whole route.
- Rule (preregister first): Δtail = DeMol − GPT-T.
  - Δtail ≤ −2.0 meV: submit J1, J2, J3, J4 (and J5 if approved).
  - −2.0 < Δtail ≤ −1.0: submit J1 and J2 only.
  - Δtail > −1.0: submit nothing; record NO_TRAIN ("dev gain did not survive on the tail").
- Init file: `load_family_model` checks the state sha (`family_training_kernel.py:23-24`).
  Both 100K s42 recipes, DeMol included, bind state sha a65034da. That file is
  `G:\我的云端硬盘\MolGap\code\gptrans_init.pt` (file sha 27590c1a, 23,606,979 bytes; listed in
  `D:\molgap-artifacts\initializations\gptrans-width-500k-20261117\record.json`). It exists
  (checked). Copy it, check the sha, and pass it as `--initial-state` for both runs. If it fails
  the sha check, F0 cannot run, and J3-J5 are not submitted.
- Owner: the 500K branch (`codex/exp/demol-cin-500k-ims-20261120`), because F0 gates both branches.
- Cost: local 5060 eval only, about 10-20 min. It needs the scorer fix (P1a). No training,
  no DataLoader workers. The no-local-DeMol rule covers training through spawn workers, not
  this in-process eval, but watch handle count with `scratchpad\guard.py` anyway.

### J1 (rank 1): DeMol-CIN 100K, seed 43, on IMS
- Hypothesis: the 100K gain survives a new init, data order and dropout stream (court Route 1a).
- Spec: `ogb-train-100k`, standalone (placeholder init), ncpus 8, preflight 00:45:00,
  train walltime 06:00:00, `--wall-s 19800`.
- Est. A100 time: 1.0-1.8 h. DeMol is launch-bound (about 6,400 kernels per update; 87-105
  ms/update measured on the 5060 in
  `C:\Users\17449\.codex\worktrees\demol-cin-execution-5060\molgap\experiments\demol_cin_execution_5060_a2\decision.md`),
  so T4-to-A100 ratios do not apply. Cache about 6 GB RAM.
- Code: none new beyond P1-P3. Recipe = r2 demol_cin recipe with seed 43, torch 2.7.1,
  IMS 100K manifest, no auxiliary EMA, and the placeholder state sha.
- Rule: see J2 (paired with J2).
- Risk: first DeMol run on IMS (`ogb` via `--ogb-zip`, cache build in preflight). The
  preflight stage (2 real steps, chained `afterok`) is the in-job early check.

### J2 (rank 2): GPT-T pair_update_norm 100K, seed 43, on IMS (same-platform reference for J1)
- Hypothesis: needed as J1's matched reference, and as the missing same-recipe N=2 GPT-T
  price at 100K (court Route 1b; lit_C §2 says it was never measured).
- Spec: `ogb-train-100k`, `--initial-state` = 500K cohort `s43.pt` (same mode
  `degree_bond_local_pair_update_norm_ema999`, so the same architecture; verify by loading),
  ncpus 8, preflight 00:45:00, train 04:00:00, `--wall-s 12600`.
- Est. A100 time: 0.6-1.5 h.
- Code: none new. The init file must reach IMS (P5). This is the first non-standalone run
  through the driver, so a local `--max-steps 20` GPT-T dry run is required (safe; no
  `attach_inputs`).
- Preregistered rules (J1+J2; dev rescored offline with the fixed scorer; paired row
  bootstrap 2000, seed 42):
  - (a) Δ43 = dev MAE(J1) − dev MAE(J2).
    - Δ43 ≤ −3.0 and CI upper < 0: replicated.
    - −3.0 < Δ43 ≤ −1.5: inconclusive; no DeMol scale-up from 100K evidence.
    - Δ43 > −1.5: noise plus selection; close the DeMol 100K lead. If J3/J4 ran, read them
      under their own rule but flag the conflict.
    - Report the two-seed mean (with s42) only.
  - (b) Four models: G42 (Kaggle), G43 (J2), D42 (Kaggle), D43 (J1).
    P = mean MAE of the two same-recipe pairs (G42+G43, D42+D43) − mean MAE of the four
    cross pairs (equal-weight prediction mean, dev rows).
    - P ≥ 3.0: hetero ensembling worth a 500K test.
    - 1.0 ≤ P < 3.0: inconclusive, no scale-up.
    - P < 1.0: close the hetero line.
    Thresholds as in court `verdict.md` Route 1. Mixing platforms inside a pair is allowed.
    Single-model verdicts across platforms are not (s42 is Kaggle T4 torch 2.11; J1/J2 are
    IMS A100 torch 2.7.1).
- Risk: untested non-standalone driver path; init transport.


## Data roles and sealed roles
Train rows < 3,250,000 only. Dev = the view's own 50K rows. `tail_holdout_v1` (rows 3,250,000..3,378,605) is read only by the offline scorer, once per selected checkpoint, never to train, tune or select. Official valid and test: not read, not authorised. `sealed_roles` are all false in every run spec.

## Non-strict statement
IMS V4 has no Spec/package/runtime-certificate layer. Every verdict is a non-strict nomination. Single-model comparisons across platforms (Kaggle T4 torch 2.11 vs IMS A100 torch 2.7.1) are not allowed.

## Rulings (verbatim, Opus adjudicator 2026-11-20)

# IMS tonight rulings, 2026-11-20 (Opus adjudicator; user asleep, no questions allowed)
Base: ims_weekend_plan.md. Authority: user "do everything, then shut down"; 11-12 infra pre-approval; 11-19 Opus-check rule. No push, no official valid/test, no frozen-data rebuild, no spend beyond plan (≤ 5 runs, plan walltimes).

## 1. Rulings
- Q1 Jobs: J1+J2 when F0 Δtail ≤ −1.0 or F0 is unavailable (steps 4/5); add J3, J5, J4 (in that order) only when Δtail ≤ −2.0 and Q2-Q4 gates pass. Reason: the plan's preregistered rule, within its stated 13-25 A100-h estimate.
- Q2 Cap: non-terminal runs in our account (`jobinfo -c`; each preflight+train chain counts once, held/queued/running all count) + new runs ≤ 4. Live Cutlass+GRIT as 2 chains → J1+J2 only; as 1 chain → plus J3; as 0 → plus J3, J5. Reason: only rule enforceable from jobinfo without cross-run chaining; matches REMOTE_HANDOFF's cap of 4.
- Q3 Credits: read `showlim -c` and past-job point use from our records; submit a job only if (remaining − hard line) ≥ requested walltime × measured rate summed over all jobs. Rate unreadable → J1+J2 only, and only if `showlim -c` shows remaining above the hard line; output unreadable → submit nothing. Reason: unknown is not zero; no invented rate.
- Q4 RAM: J3/J4 need a measured fact: an IMS job log or provenance under our root showing ≥ 64 GB at ncpus 16, AND a local CPU RSS of one 50K-shard `_CellDataset` (scaled ×11) ≤ 0.5 × that. Otherwise drop J3, J4, and J5 (no value alone). Reason: estimates do not count.
- Q5 Infra: P1a (scorer moves `extra_batch_fields`) approved. P1b approved only if ≤ 10 lines, logs the already-computed primary dev MAE, writes `selected_eval.json` for the primary selection only, and passes `tests/test_ims_v4_submit.py` once; else cut it (cut-list item 1). P1c (required): `--training-workers N` on `family_recipe_run.py` (default 1, so IMS behavior is unchanged), passed to `ExecutionTuning`; the driver now hardcodes 1 spawn worker. Reason: extension/speed infra, pre-approved 11-12; P1c is what makes step 7 legal on Windows.
- Q6 Init transport: `git add -f` s43.pt into the replication branch (J2) and s44.pt into the 500K branch (J5), exp branches only (*.pt is gitignored; no `.gitattributes` edit); pass `--initial-state {code_root}/experiments/<q>/init/s4N.pt`. Reason: the V4 helper `git archive`s the commit, so the file is archive-hashed with no new remote step; scp would add an untested one.
- Q7 Skip 100K-first gate: allowed, with F0 Δtail ≤ −2.0 as the unbiased falsifier. Every IMS verdict is non-strict nomination only (Cutlass/GRIT precedent). Moot if Q2 admits only J1+J2. Reason: saves a weekend; F0 has no selection bias.

## Tonight's ordered steps and STOP rules (verbatim)

## 2. Tonight, strictly ordered (Sonnet executor)
1. Read REMOTE_HANDOFF and the `ims-molgap-workloads` skill. Lane B (below) starts now.
2. Lane A: P1 branch `codex/infra/ims-family-dev-scoring-20261120` from desktop b028886c6; P1a, P1b, P1c; one test run; `infra_budget.py`; `audit` agent; merge back to desktop (no push).
3. P2: create both exp branches from the new desktop tip; import `demol_cin.py` + catalog entry from f7f97e68d (sha256 must equal the s42 file); RML review; `protocol.md` with the section 2 rules and these rulings verbatim; prospective `trajectory.json` per arm plus F0. COMMIT before step 5. STOP if RML shows a closed duplicate → submit nothing.
4. Copy `G:\...\code\gptrans_init.pt`; check file sha 27590c1a and state sha a65034da. Mismatch → skip F0; J1+J2 only (J3-J5 never).
5. F0 on the 5060 (serial; nothing else on the GPU). Already verified (Opus): the scorer has no `num_workers`/`multiprocessing`/`spawn`/`persistent_workers`, so the PyG DataLoader runs in-process with 0 workers; re-grep only if P1a touched the loader. Run under `guard.py --max-handles 20000 --max-ws-gb W`, where W = physical RAM − 6 GB (check it first). The guard log must show peak handles < 20000. Dev check DeMol 0.134635 / GPT-T 0.139440 eV (±1e-4) → then tail once each; record Δtail in protocol.md; add both to the tail leaderboard.
   STOP: dev mismatch → submit nothing (scorer wrong). Guard trip on handles or child processes → kill, no retry, treat as F0 unavailable (= step 4 mismatch). Working-set trip → batch-size-only rerun allowed once. Δtail > −1.0 → NO_TRAIN, submit nothing, go to step 11.
6. P4: J1/J2 recipes (always); J3-J5 only if Δtail ≤ −2.0. DeMol `initialization_sha256` = the Cutlass 500K recipe value f504fa4e. The placeholder is a fixed `torch.zeros(1)` and the kernel checks the state (tensor) sha through `read_frozen_state_artifact`, so the value is model- and torch-version-independent. `checkpoint_every` 512 at 500K. Unique run ids from the plan.
7. P6 local checks, serial on GPU after F0: `read_recipe` + `mode_policy` CPU load for all; GPT-T `--phase preflight --max-steps 20 --training-workers 0` dry run with s43.pt under guard (without P1c, skip it and accept the in-job preflight as J2's only check, at the cost of at most one queue cycle). No DeMol driver run on Windows. DeMol cell RSS (CPU, guard) only if J3 is still in play.
   STOP: GPT-T dry run fails → fix (≤ 3 attempts, then Opus); still failing → drop J2 and J5, and therefore J1 → submit nothing.
8. Lane B results must be in: `jobinfo -c`, `showlim -c`, Cutlass/GRIT preflight state, RAM facts, ogb zip and the 500K dev shards present. Apply Q2-Q4. STOP: any command fails or is unreadable → submit nothing. Cutlass/GRIT preflight failed from a shared driver cause → fix, then re-review.
9. Opus pre-submit review (section 4) of the final commit. Blocker → fix and re-review; 3 failed rounds on one job → drop it (with J1/J2 as a pair).
10. Submit serially in the order J1, J2, J3, J5, J4 (only the admitted set), one `submit_from_windows.py` call each. After each: `jobinfo -c` shows the preflight + held train; log the job ids in `ims_submission.md`; commit. Error → no blind retry; check jobinfo; on a clear pre-queue refusal, fix and re-review; otherwise record UNKNOWN and stop further submits.
11. Write the state files and the morning report (section 5), then shut down.
