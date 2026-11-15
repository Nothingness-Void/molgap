# Decision: arm A, same-recipe B1024 four-init ensemble (2026-11-16)

Rules: [protocol.md](protocol.md), frozen before any result of these runs. The rule mapping, [verdict_A.py](verdict_A.py), was committed in e5cf2da8 before the s44/s45 tail scores were computed.
Each new endpoint was scored once on tail_holdout_v1 (128,606 rows). No snapshot was chosen. Official valid/test and sealed labels were not read.
This is the **arm A part** of the lever decision only. Arms B256, C and D are still running (jobs 2004612-2004615), and E is scored after them; they are decided in later records.

## Outcome
- **A: PASS.** The equal-weight four-init ensemble (s42, s43, s44, s45) scores 83.992 meV. That is 1.357 meV below the s42 + s43 pair (85.349), with a one-sided 99% paired-bootstrap upper bound of -1.207 meV (threshold <= -1.0 and upper < 0).
- Under the protocol, "all B1024 inits, equal weights" becomes a **candidate**. The final candidate is chosen only after B/C/D/E.
- Both new inits are inside the 87.89 +/- 1.5 meV band, so no member is INCONCLUSIVE.

## Single models (tail MAE, meV)
| Init | Job | Tail MAE | In band | Dev rows 0:50,000 (in-train) / pass-60 probe | `ema_model.pt` sha256 |
|---|---|---|---|---|---|
| s42 | 1998627 | 87.941 (scored 2026-11-15) | yes | 66.67 / 66.33 | f6e8eaf4...dd7a617c1 |
| s43 | 1998850 | 87.846 (scored 2026-11-15) | yes | 66.70 / 66.99 | a639c9c9...19001246 |
| s44 | 2004416 | 87.891 | yes | 66.78 / 66.27 | 6793390fb4d4f148bf703e40c1cef5320ae97cc2baa1f0eba38628ea9bc402d7 |
| s45 | 2004437 | 88.075 | yes | 66.37 / 66.25 | 5ebc5dd0a45f58f353f5f2a83f6294755d001b45bef1e953a6fd08d251db8728 |

Noise floor (four inits): sample std (ddof 1) 0.099 meV, range 0.229 meV, mean M 87.938 meV; N = max(1.08, 0.229) = 1.08 meV.

## Ensemble curve (equal weights, descriptive)
| K | Mean over subsets | Min | Max |
|---|---|---|---|
| 1 | 87.938 | 87.846 | 88.075 |
| 2 | 85.357 | 85.282 | 85.441 |
| 3 | 84.454 | 84.427 | 84.477 |
| 4 | 83.992 | - | - |

Gains per added member: 2.58, 0.90 and 0.46 meV. The baseline pair s42 + s43 (85.349) sits mid-range among the six pairs (85.282-85.441), so the post-hoc choice of baseline barely moves the result.
Residual correlation between GPT-T inits: Pearson 0.903-0.911, Spearman 0.839-0.843. GPT-T against K1: Pearson 0.837-0.850.
Heterogeneous combinations with K1 s42/s43 (descriptive, not a criterion): six-member equal weight 85.112, half four-init GPT-T + half K1 pair 86.991; both are worse than the four-init GPT-T ensemble.

## Confounds and limits
- **Seed scope.** All four inits change only the init file. Data order (sampler), dropout and global RNG are seed 42 in every run (code review; s43 precedent). The noise floor therefore measures init noise only and understates full-seed training noise. B/C/D thresholds that use sigma or M from these four runs (`max(1.5, 3 sigma)` = 1.5 meV here) inherit this; a lever effect near the bar should be read with that in mind.
- 85.349 is a post-hoc 1-of-4 baseline. All four scores come from the same 128,606 tail rows, so the row bootstrap covers row sampling, not training stochasticity.
- Ensemble cost is 4x inference and 4x training (about 34 A100 h for the four inits).
- Endpoint reading: `selected_model.pt["model"]` (weights=ema, final-ema) has tensors identical to `ema_model.pt["model"]` (337 tensors, checked for s44 and s45). `ema_model.pt` was scored with `--weights-key model`, the same file and key as s42/s43. `live_model.pt` stores EMA weights (code review) and was not used.

## Acceptance of the retrieved runs
Both runs are under `D:/molgap-artifacts/retrieved/pcqm-gptrans-b1024-minus-tail-s4{4,5}-20261115/`:
- Every file matches its remote sha256 (`remote_sha256.txt`; 10 files each).
- `training_complete.json`: COMPLETE, 60 passes, 190,380 steps, selection final-ema. Scheduler: both jobs had left the queue when checked.
- `run_provenance.json` is byte-identical to s43: rows [0, 3,250,000), tail not read and not used for selection, valid/test not read.
- `run_config.json` differs from s43 only in `init_file`/`init_sha`. s44 state sha ff9496504ff1...1728b, s45 58c48765ec58...26b7b; file shas 2415cdbf...c2573 and f653837b...12486 equal the submission record.
- nonfinite_params = 0 in all 60 log records each. Last-pass probe 0.066269 (s44) and 0.066252 eV (s45). Runner sha 120d5571... in `sessions.jsonl`; IMS A100-SXM4-80GB, torch 2.7.1+cu126.
- The scorer (`pcqm_tail_holdout_eval.py`, sha 756227a3, unchanged from s42/s43) loads strictly and rebuilds the recipe from `run_config.json` (`--run-config`, `--initial-state`, `--weights-key model`, `--dev-range 0:50000`).

## Evidence (all under `D:/molgap-artifacts/`)
| File | sha256 |
|---|---|
| tail_eval_full_levers_20261116/gptrans_s44.json | 4edea44a3f40948bcc9c7d6fde085103087c037f80167dbb9909278e42d4f23a |
| tail_eval_full_levers_20261116/gptrans_s45.json | 21af1123731dd910d25ea333c2a0440d3a203e99498b91e4cf1497357fe22d04 |
| tail_eval_full_levers_20261116/preds/gptrans_minus_tail_s44_tail_holdout_v1.pt | 697a415a2b6cb1b8194e1d028656bdd29973117afd10c55a80c9edc5d8a941bf |
| tail_eval_full_levers_20261116/preds/gptrans_minus_tail_s45_tail_holdout_v1.pt | 42e99f42d99520963293b292f1656ff602213cbdaaa253431a9e5bd3f6ae6cf7 |
| tail_eval_full_levers_20261116/verdict_A.json (script sha 2709e5af...70d54) | 444050e810182fdb13a5c35372e48ab60f7105ffad58bb06696ce5f7df8aa1d5 |
| tail_holdout_master_leaderboard.json (after adding s44, s45; 64 entries) | 06eecd9a2ebe32656d01acd4e5c190ca14c597f58cfda8217a098d44e4ef8df7 |

The s42/s43/K1 prediction inputs and their hashes are in the `inputs` block of `verdict_A.json`. Reproduce:
`python experiments/pcqm_gptrans_full_levers_20261115/verdict_A.py --out OUT.json S42=<pred> S43=<pred> S44=<pred> S45=<pred> [K1S42=<pred> K1S43=<pred>]`.

## Cost
A100 hours measured from `sessions.jsonl` (one session each): s44 8.407 h, s45 8.576 h, total 16.983 h. Queue and CPU hours are not measured. Points are not separated per job here.
