# Decision: full-scale minus-tail K1 + GPT-T fusion (2026-11-15)

Rules: [protocol.md](protocol.md) and [protocol_supplement_second_init.md](protocol_supplement_second_init.md), both frozen before
the GPT-T scores. The code mapping of the rules ([fusion_verdict.py](fusion_verdict.py)) was committed in 292ceb39, before any fusion number was computed.
Each endpoint was scored once on tail_holdout_v1 (128,606 rows). No snapshot was chosen. Official valid/test and sealed labels were not read. The production registry was not touched.

## Outcome
- **Original protocol: FAIL.** The equal-weight K1 s42 + GPT-T s42 fusion is 0.229 meV *worse* than GPT-T s42 alone.
- **Supplement (a): not passed** ("ensemble gain only" in the table's wording, though the hetero pair shows no gain over its best member).
  The hetero pair is 2.821 meV worse than the GPT-T homogeneous pair.
- **Supplement (b), (c): INCONCLUSIVE** under the sanity rule, because K1 s43 falls outside the s42 +/- 1.5 meV band and both comparisons include it.
  The descriptive numbers miss their thresholds anyway (-0.520 vs -3.0; -0.844 vs -1.0), so the consequence is the same:
  the 4-member ensemble is not a candidate, and no 2-member fusion passed either.
- (a) is not INCONCLUSIVE. K1 s43 only enters through the choice of the better homogeneous pair, and the GPT-T pair (85.349) beats the K1 pair (96.256) by 10.9 meV, far more than the 1.98 meV K1 band excess.
- Single-model result: GPT-T B1024 is about 9 meV better than K1 on tail at full scale, in both seeds.

## Single models (tail MAE, meV)
| Member | Endpoint | Tail MAE | Checkpoint sha256 | Platform |
|---|---|---|---|---|
| K1 s42 | final-live `selected_model.pt` | 97.026 (scored 2026-11-14, not rescored) | 344f32a8...6a1eff2 | Colab L4, torch 2.14.1+cu130 |
| K1 s43 | final-live `selected_model.pt` | 99.006 | d405f7e2...46901609 | IMS A100, torch 2.7.1+cu126, job 1999135 |
| GPT-T s42 | pass-60 EMA `ema_model.pt` | 87.941 | f6e8eaf4...dd7a617c1 | IMS A100, torch 2.7.1+cu126, job 1998627 |
| GPT-T s43 | pass-60 EMA `ema_model.pt` | 87.846 | a639c9c9...19001246 | IMS A100, torch 2.7.1+cu126, job 1998850 |

Noise floor |s42 - s43|: K1 1.981 meV (outside the 1.5 band; 500K prior 0.18), GPT-T 0.095 meV (500K prior range 1.08).

## Fusion (equal weights, paired row bootstrap 2000, seed 42)
| Comparison | Diff (meV) | Upper bound used | 95% one-sided | Threshold | Result |
|---|---|---|---|---|---|
| Original: hetero (88.170) - GPT-T s42 | +0.229 | +0.444 (97.5th pct, two-sided 95%) | +0.405 | <= -3, upper < 0 | FAIL |
| (a) hetero - GPT-T pair (85.349) | +2.821 | +3.065 (98.33th pct) | +3.010 | <= -3, upper < 0 | not passed |
| (b) 4-member (87.326) - GPT-T s43 | -0.520 | -0.259 (98.33th pct) | -0.316 | <= -3, upper < 0 | INCONCLUSIVE (would not pass) |
| (c) 4-member - hetero | -0.844 | -0.720 (98.33th pct) | -0.747 | <= -1, upper < 0 | INCONCLUSIVE (would not pass) |

The rule readings are in the docstring of `fusion_verdict.py`. The original "95% upper bound" is read as the two-sided 95% interval's upper end, as `pcqm_tail_fusion_eval.py` computed it. The supplement's "one-sided 98.3%" is read literally (98.33th percentile). The two-sided 98.3% ends are also in the JSON; no verdict depends on the reading.
K1 pair (K1 s42 + s43) = 96.256 meV; descriptive only.

## Residual correlation (descriptive; optimal weights are fitted on the same rows, not a selection basis)
| Pair | Pearson | Spearman | 50/50 MAE | Fitted w(first) / MAE |
|---|---|---|---|---|
| K1 s42, GPT-T s42 | 0.847 | 0.757 | 88.170 | 0.432 / 87.564 |
| K1 s43, GPT-T s43 | 0.850 | 0.751 | 88.768 | 0.443 / 88.110 |
| K1 s42, K1 s43 | 0.936 | 0.906 | 96.256 | - |
| GPT-T s42, GPT-T s43 | 0.912 | 0.843 | 85.349 | - |

## Attribution
- Hetero fusion fails because the members are too unequal. The ~9 meV gap between K1 and GPT-T outweighs the extra diversity (cross-family residual r 0.85 vs 0.91 within GPT-T).
  At 500K the two families were close and hetero gained 4.38 meV; at full scale GPT-T B1024 pulled ahead.
  Even the in-sample fitted weight gives only 87.56, 0.38 meV below GPT-T s42.
- Two GPT-T inits (85.349) beat every pre-registered candidate. This pair was not a pre-registered candidate, so it is reported, not adopted.
  Any use of it needs a new frozen question.
- The K1 s43 band excess is not a scoring-infrastructure fault. The two K1 configs are identical except the init.
  The in-train probe (20,480 train rows) at pass 60 is already 2.39 meV worse for s43 (85.56 vs 83.17 meV), and its live param norm is higher (955.6 vs 938.6).
  The cause cannot be split between init and the stated platform/runtime confound with two runs (insufficient_evidence).

## Confounds (stated, not corrected)
- K1 s42: Colab NVIDIA L4, torch 2.14.1+cu130 (`sessions.jsonl`; the supplement's "2.11.0" is superseded by this record).
  K1 s43: IMS A100-SXM4-80GB, torch 2.7.1+cu126, Colab src patches by `platforms/ims/k1_minus_tail/colab_src_patches.py`.
  For K1 the s42-s43 difference therefore mixes init with hardware/runtime.
- GPT-T s42/s43 are both IMS A100 torch 2.7.1. s43 changes only the init (sampler, dropout and global RNG stay seed 42), as with K1 s43.

## Acceptance of retrieved runs
For all three new runs (`D:/molgap-artifacts/retrieved/{pcqm-k1-b1024-minus-tail-s43-20261115-k1_muon, pcqm-gptrans-b1024-minus-tail-20261114-s42, pcqm-gptrans-b1024-minus-tail-s43-20261115}`):
- Every file matches its remote sha256 (`remote_sha256.txt` in each directory).
- `training_complete.json`: COMPLETE, 60 passes, 190,380 steps. `run_provenance.json`: rows [0, 3,250,000); tail, valid and test not read.
- nonfinite_params = 0 in all 60 log records. Init shas match the submission records.
- Weights: every endpoint file stores them under `model`. GPT-T `selected_model.pt` (weights=ema, final-ema) holds tensors identical to `ema_model.pt`. K1 `selected_model.pt` is weights=live, final-live.
- The scorer loads strictly. GPT-T recipes were rebuilt from each run's `run_config.json` by the registered runner's `make_recipe` (`pcqm_tail_holdout_eval.py --run-config`, `--weights-key model`).
- In-train wiring check, rows 0:50,000: GPT-T s42 66.67 / s43 66.70 meV against the pass-60 in-train probes of 66.33 / 66.99 meV.
- Recording note: `k1_s43.json` lists scorer sha 756227a3. The scorer hashes itself at write time, after the `--run-config` edit was saved. The code that ran was the committed f8a5dcf9 version, the same as the K1 s42 scoring; K1 does not touch the new path.

## Evidence (all under `D:/molgap-artifacts/`)
| File | sha256 |
|---|---|
| tail_eval_full_minus_tail_20261115/k1_s43.json | 2f0eccef...41339280 |
| tail_eval_full_minus_tail_20261115/gptrans_s42.json | 924bc29e...b49e66018 |
| tail_eval_full_minus_tail_20261115/gptrans_s43.json | e9553a15...143f00f10 |
| tail_eval_full_minus_tail_20261115/fusion_verdict.json | b860ac18...b266018 |
| tail_eval_full_minus_tail_20261115/residual_report.json | 7bd753e5...d455000 |
| tail_eval_k1_minus_tail_20261114/preds/k1_muon_minus_tail_s42_tail_holdout_v1.pt | c3381120...712fed3 |
| tail_eval_full_minus_tail_20261115/preds/k1_muon_minus_tail_s43_tail_holdout_v1.pt | a50a625d...0950b |
| tail_eval_full_minus_tail_20261115/preds/gptrans_minus_tail_s42_tail_holdout_v1.pt | fb172dbd...91b5 |
| tail_eval_full_minus_tail_20261115/preds/gptrans_minus_tail_s43_tail_holdout_v1.pt | 38add7b0...6671 |
| tail_holdout_master_leaderboard.json (after adding the four full-scale entries) | 13e2970c...5000 |

Full hashes are in the JSON `inputs` blocks and in `sha256sum` of these paths. Reproduce:
`python experiments/pcqm_full_minus_tail_fusion_20261113/fusion_verdict.py --out OUT.json K1S42=<s42 pred> K1S43=<...> GPTS42=<...> GPTS43=<...>`.

## Cost (A100 and points kept separate)
- A100 hours, measured (`sessions.jsonl`): K1 s43 3.657 h, GPT-T s42 8.620 h, GPT-T s43 8.410 h; total 20.687 h.
- Points (`showlim -c`): from 03:00 to the reading after all three jobs ended, group TOTAL-USE rose 2,169 (1,456,210 to 1,458,379) and user sm2 rose 1,513 (70,853 to 72,366).
  1,513 / 20.687 = 73.1 points/A100-h is an **upper bound**, because job 1996402 (16 CPUs, not ours) accrued over the same window. The exact A100 rate stays unmeasured.
