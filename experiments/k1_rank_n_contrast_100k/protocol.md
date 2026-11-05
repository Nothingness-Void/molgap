# K1 post-BN 100K Rank-n-Contrast screen: frozen protocol

Prospective server screen, designed 2026-11-06. One same-run seed-42 pair on the
frozen K1 post-BN recipe; every value below is frozen before any training.

## Question and prior art

Does a joint Rank-n-Contrast (RnC) term on the pooled embedding improve the K1
Gap MAE? RML (local and `origin/molgap-desktop`) and a text search of every remote
branch contain no RnC, RankSim, contrastive, ordinal or label-distance loss. The nearest
records constrain something else: same-molecule dropout consistency
(Desktop `pcqm_k1_dropout_consistency` terminal decision, +1.5 meV, below gate), joint
atom reconstruction ([decision](../pcqm_k1_joint_atom_reconstruction_100k/decision.md))
and the combination screen ([decision](../k1_combination_100k/results/completed/decision.md)).
None orders different molecules' embeddings by Gap distance.

## Source facts and what is only a hypothesis

- Paper (arXiv 2210.01189, read 2026-11-06): loss over anchors i and positives j with
  S_ij = {k != i : d(y_i,y_k) >= d(y_i,y_j)}; negative-L2 similarity; tau 2.0 best on
  {0.1,0.2,0.5,1,2,5}; batch 256 with two augmented views; tasks are image and EEG
  regression. Table 5c (AgeDB MAE): linear probe 6.14, fine-tune 6.36, joint
  regularisation 6.42, plain L1 6.63. The joint scheme is the weakest RnC scheme and the
  paper gives no joint weight. Appendix G.3 shows augmentation matters.
- RankSim (arXiv 2205.15236) matches full neighbour rankings through blackbox
  differentiation of a ranking function; its gradient is not RnC's softmax gradient. It is
  not used here and is not a successor.
- Transfer to PCQM4Mv2/K1 (one view, batch 128, no augmentation, joint scheme, an MLP
  head already trained with L1) is a hypothesis. A fall in the RnC value or a tidier
  embedding is not a success criterion; only paired development MAE is.

## Design

- One kaggle3 (`nvoid912`) T4x2 session, one arm per device, strict same-run pair:
  `reference` (the frozen preset retrained, mode `reference`, one training forward) and
  `rnc` (addon `k1_rank_contrast/1`, mode `rnc`). The recipe is not mean2: the single
  existing training forward supplies both the Gap prediction and the embedding.
- Both arms use `experiments/presets/k1-post-bn-100k-v1.json`, initial state
  `k1_reference_s42.pt` (8ef6d4ba...), seed 42, 100K train / 50K development, 40 passes
  (31,240 updates), FP32, physical batch 128, selected-live development policy. Data
  order, dropout RNG consumption, optimizer, BN, clip 1.0 and selection are identical.
- Loss: `L = L1(head(z), y_norm) + 0.007 * L_RnC(z, y_norm)`, `z` = `encode()` output
  (192-d mean-pooled embedding), not detached, no projection head. L_RnC: unit-L2
  normalised z, similarity -||z_i - z_j||_2, tau 0.1, S_ij as in the paper (ties inside),
  mean over positives then anchors, N=128 distinct molecules per step. Normalisation
  (scale cannot be inflated) and tau 0.1 (unit-sphere distances lie in [0,2]; the paper's 2.0
  was tuned on raw features) deviate from the paper by choice, not by measurement.
- Lambda rule, applied once (`lambda_preflight.py`, result recorded): the median over the
  first 8 consecutive 128-row batches of train shard 0 of
  ||grad_encoder(RnC)|| / ||grad_encoder(L1)|| at the frozen initial state in train mode
  gives 34.03 for unit weight, so lambda = 0.25 / 34.03 = 0.00735, rounded to 0.007
  (the review range was [0.002, 0.2]). It is frozen; no later tuning.
- No inference change: the head and encoder are untouched, state keys and parameter
  count (3,658,817) equal the reference; the term is training-only.

## CPU qualification (2026-11-06, synthetic 128-graph batches; see `qualify_cpu_result.json`)

Vectorised loss equals the naive triple loop including tied labels; gradients finite,
also for identical embeddings and identical labels; RnC gives no gradient to the head and
a nonzero gradient to the encoder; an eval embedding is bit-equal when the other 127
molecules change (rows are isolated; train-mode BN still couples rows by design, as in the
reference); train forward equals head(encode); parameters, state keys and optimizer
ownership equal the reference; a model+optimizer save/reload mid-run resumes bit-equal;
the existing kernel workflow test passes for mode `rnc`. Measured cost on CPU: loss
11.7 ms per step (N=128), 1.007x step time of the reference on synthetic graphs. This is not
a T4 measurement and not a statement about the real-data memory peak (the mask is
128^3 = 2.1M entries per tensor, about 8 MB fp32).

## Cost

Same class as the combination session (6,717 s wall, 3.73 T4-hours). Estimate 1.95 h
wall, 3.9 T4-hours. Authority: 7,680 s wall x 2 devices, ceiling 4.27 T4-hours.

## Frozen gate

Effect = rnc minus reference selected development MAE. The arm retains a material lead
only if all hold: (1) effect <= -3.0 meV; (2) the 2,000-draw paired row bootstrap 95%
upper bound < 0; (3) final-live delta < 0; (4) mean delta over the last five
observations < 0. Labels as measured: `POSITIVE_BELOW_GATE` (effect < 0 and upper bound
< 0, not all of 1-4), `INCONCLUSIVE` (interval includes 0), `NEGATIVE_UNDER_CONTRACT`
(effect > 0 with lower bound > 0). Scale only: init-only SD 1.074 meV, paired SD 1.519
meV ([decision](../k1_init_noise_server_100k/results/completed/decision.md)); the gate is
1.98 paired SD, and the prior K1 screens' mean -1.98 meV (SD 0.66) means a below-gate
negative-direction result is expected even from a null mechanism.

## Stop points

- Cost: the frozen 7,680 s wall (`STOP_FOR_COST`); no extension, continuation or retry
  without new authority.
- Science: |effect| < 1.5 meV closes RnC at 100K. Any result below the gate closes it with
  no tau, lambda, projection-head, normalisation, two-view or RankSim successor without a
  new user decision.
- Engineering: non-finite loss; a T4 step-time ratio above 1.15 is reported as a cost
  finding. The reference is expected at 141.2945 meV bit-identically; drift is reported and
  the gate is read against the same-run reference.

## What is reported

Selected development MAE per arm, the effect with its row interval, final-live and
last-five deltas, reference drift, wall and T4-hours from the allocation ledger, and the
per-arm worker time. The embedding-geometry probe (Spearman of embedding distance versus
Gap distance, clip-norm share, RnC value) needs a retained checkpoint and is a separate
NO_TRAIN analysis, not part of this gate.

## Limits

No automatic seeds 43/44, no 500K or full training, no protected roles, no retry or
successor. One initialization and one seed; the row interval does not include
initialization variance, and the development rows are reused for selection.
