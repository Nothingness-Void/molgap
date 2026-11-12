# GPT-T 500K: batch 1024 + TF32 against the frozen batch-128 fp32 recipe

Prospective desktop question, 2026-11-12 JST. Colab L4 runs by the user. Branch
codex/exp/gptrans-b1024-precision-500k-20261112.

## Question

Does GPT-T 12x256 PairNorm (500K, first 10 shards, 60 passes) trained on Colab with batch 1024,
TF32, lr 2.828e-3, EMA 0.999 and no compile reach tail_holdout_v1 accuracy comparable to the
frozen recipe (batch 128, lr 1e-3, fp32)?

## Evidence

- Reference: same model, batch 128, init seeds 43-46, tail_holdout_v1 MAE in meV:
  108.357 / 107.273 / 108.091 / 108.037. Mean 107.94, sample SD 0.47 (init variation only).
  Source: D:/molgap-artifacts/tail_holdout_eval_gptrans_500k_batch.json (numbers re-checked
  against the file: 0.10835716, 0.10727309, 0.10809141, 0.10803716 eV; 128,606 rows each).

## Design

- Runner: platforms/colab/pcqm_k1_b1024_production/train_full_b1024.py at commit 39660d93.
  GPT-T knobs: --gptrans-batch-size --gptrans-lr-mult --gptrans-ema-decay (parser lines 738-740).
- Arms (primary): init seed 42 (run runs/gptrans-b1024-500k-s42-20261112) and init seed 43
  (runs/gptrans-b1024-500k-s43-20261112). Batch 1024, --gptrans-lr-mult 2.8284271 (lr = 1e-3 x
  sqrt(8) = 2.828e-3, Muon+AdamW groups the same multiple), --gptrans-ema-decay 0.999, no compile,
  TF32. 500K rows (--max-shards 10), 60 passes, warmup 4 passes then cosine to 1e-6
  (runner lines 195-197), selection final-ema.
- Training platform: Colab L4 (not A100), torch 2.11.0 pinned. These do not change the decision rule.
- Scoring: each ema_model.pt is scored once on tail_holdout_v1 with
  platforms/local/pcqm_tail_holdout_eval.py, --weights-key model, one run per checkpoint.
  At most 5 scorings in total (2 primary, 2 B512, 1 seed 44). Official validation does not take part.

## Frozen decision rule (Fable adjudication; no change after the first scoring)

- Metric: tail_holdout_v1 MAE (meV) of each seed's ema_model.pt. Comparison anchor 107.94.
- ADOPT: mean of the two seeds <= 109.44 (anchor + 1.5) AND each seed < 110.94 (anchor + 3.0).
- REJECT: any seed >= 110.94. Then the B512 fallback below runs on the same two seeds with the
  same thresholds.
- KEEP (inconclusive): every other case (both < 110.94 but mean > 109.44). Then add seed 44
  (same B1024 settings), scored once; no further rule change.

## B512 fallback (exact arguments)

Same runner, same data, same init files, only these GPT-T knobs change:

- --gptrans-batch-size 512
- --gptrans-lr-mult 2.0   (lr = 1e-3 x 2.0 = 2.0e-3; runner line 195 `learning_rate=1e-3 * gptrans_lr_mult`)
- --gptrans-ema-decay 0.9960   (0.999^4 = 0.99600, per-sample EMA horizon matched to batch 128;
  applied by runner lines 210-211 and 536-538)
- --arm gptrans_pairnorm --max-shards 10 (500K rows); no compile; init seeds 42 and 43 via --init-file/--init-sha.

Verified at batch 512 from the runner: steps/epoch = 500000 // 512 = 976 (lines 70-74, 183, 519),
total steps = 976 x 60 = 58,560 (line 184, 520). Warmup is 4 passes (warmup_units=4, line 197),
counted in passes, not steps, so it is unchanged (3,904 steps at B512; 1,952 at B1024 with 488
steps/epoch, 29,280 total). Checkpoint every pass (line 200). The lr 2.0e-3 and EMA 0.9960 targets
are the sqrt-batch lr scaling and 4-step EMA compounding relative to batch 128 (and 0.999^8 =
0.9920 would be the matching B1024 value; B1024 deliberately keeps 0.999 as stated above).

## Timeline (honest record)

Training was started by the user before this rule was written to disk: the seed 42 run had
already reached at least pass 10. No tail scoring has been done and no B1024 result of any kind
(tail, official validation, or dev) has been seen by the planner. This file is committed before
the first scoring.

## Descriptive items (not part of the decision)

- train_log.jsonl loss curves of both seeds against the batch-128 logs.
- Per-molecule residual report (--residual-report of the tail fusion scorer); descriptive only,
  not used for selection.

## Limits

- Two seeds; init SD about 0.47 meV from the reference only. The 1.5 meV mean margin is about
  3 SD of a single run; the 3.0 meV per-seed cap guards against one bad run.
- Hardware and torch differ from the reference (L4/torch 2.11.0 vs RTX 5060/2.7.1); confounded
  with batch and precision by design, since the question is the practical Colab recipe.
- No retry, continuation, adoption into a deliverable or use of official validation.
