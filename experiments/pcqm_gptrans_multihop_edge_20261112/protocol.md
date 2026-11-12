# GPTrans multi-hop path edge encoding: frozen protocol (E1 100K, E2 500K)

Desktop question, 2026-11-12. Owner branch `codex/exp/gptrans-multihop-edge-20261112`.
User authorized 30 Kaggle hours for evidence search; E1 submission needs no further request.
E2 is frozen here but is submitted only if E1 passes its gate.

## Question

Does the official GPTrans multi-hop path edge encoding lower the error of the retained
GPT-T (BondLocal + pair-update-norm) at matched compute?

## Evidence review (RML, 2026-11-12, worktree `--repo-root .`)

- `lookup --mechanism multi hop`, `--term multihop`, `--term geodesic`: no GPT-T path-edge
  trajectory. `--term hop` finds only K1 `pcqm-k1-hop-shell-100k` (hop_ordered, -1.12 meV,
  NEGATIVE_UNDER_CONTRACT; node content partition, not pair-stream bond sequences).
- `--mechanism edge`: `gptrans-t-local-bias-rwse16-local-edge` (sparse true-bond bias,
  NEGATIVE_UNDER_CONTRACT) and K1 edge-state families. They use direct bonds only; none
  encodes the ordered bonds along shortest paths for non-adjacent pairs.
- Deficiency: our GPT-T adds the bond embedding only to directly bonded pairs (gptrans.py
  `_dense_inputs`); pairs at hop distance >= 2 see only a distance embedding. Official GPTrans
  (models/gptrans.py L107-134, `multi_hop_max_dist=20`) also injects ordered path bonds.
- Alternative explanations: BondLocal plus 12 pair-propagation layers already recover path
  content; the distance table is sufficient for PCQM4Mv2; any gain is below init noise
  (K1 100K init spread 1.63 meV; GPT-T 500K init noise is in `pcqm_gptrans_pairnorm_init_noise_500k`).

## Mechanism (candidate arm)

For every ordered real-atom pair (i, j) with 1 <= d <= 20 hops: take one shortest path, embed the
k-th bond (k = 0..d-1) with the shared OGB bond encoder averaged over its three slots
(sum / 3), multiply by position-specific 32 x 32 matrix `path_mix[k]`, sum over k, divide by d,
and add to the real-atom block of the pair stream before the first block. 20 matrices,
20,480 new parameters (reference 5,871,201). Differences from official, all deliberate or forced:

- Tie-break among equal shortest paths: lowest atom index at each step (official uses its
  precomputed Floyd-Warshall path). Pairs farther than the model distance cap 20 and unreachable
  pairs get no term (the model's `shortest_path_cap` is 20, so d > 20 is not distinguishable).
- Official keeps its own bond embedding; ours reuses the shared `bond_encoder` (also feeds the
  direct-bond term). Divisor is the hop count d (official `spatial_pos` is d + 1 in the model).
- `path_mix` starts at zero, so the candidate's initial output equals the reference
  bit for bit; the frozen candidate state is the reference state plus that zero tensor.
- Implemented as an addon (`src/molgap/gptrans_multihop_edge.py`) overriding `forward` only;
  shared trainer, validator, backend are unchanged (`_dense_inputs`, `_shortest_path` stay
  the prepared fast path).

## E1 (100K) design

Two same-job pairs, one Kaggle T4x2 kernel each (one arm per device):

| Pair | Init (torch seed) | Account | Run |
|---|---|---|---|
| s43 | 43 | kaggle2 (kaseichou) | `pcqm-gptrans-multihop-edge-20261112-s43`, arms `ref_s43`, `mh_s43` |
| s44 | 44 | kaggle1 (nothingnessvoid) | `pcqm-gptrans-multihop-edge-20261112-s44`, arms `ref_s44`, `mh_s44` |

Reference = accepted 100K BondLocal + pair-update-norm recipe (`e1/*/recipe.json`: copy of the
norm-EMA 100K recipe without the auxiliary EMA): 60 passes, 46,860 steps, batch 128, AdamW LR 1e-3,
WD 0.05, warmup 4 + cosine, EMA 0.999, best-development-EMA selection, FP32. Spec seed label 42
(sampler and dropout RNG), so within a pair both arms see identical data order and dropout streams
and differ only in the mechanism (plus its zero tensor); across pairs only the init differs.
Init states for s43/s44 equal the retained 500K init-noise states (same builder).

Roles and metric: train source_idx [0, 100000) and the contract's development role
(`fixed100k-next50k`, source_idx [100000, 150000), 50K rows; it is a selection-used role, not held
out). Metric: development MAE of the best-development-EMA endpoint, in meV. Official validation,
test_dev, test_challenge and protected roles are not read.

Gate. d_s = MAE(mh_s) - MAE(ref_s) in meV for s in {43, 44}. Pass iff mean(d_43, d_44) <= -3.7 meV
and both d_s < 0. Otherwise both candidate arms close NEGATIVE_UNDER_CONTRACT and there is no
500K. Derivation: the 500K gate is 3.0 meV; assuming the most optimistic transfer in which 82%
of a 100K gain survives (18% attenuation), the 100K gain must be >= 3.0 / 0.82 = 3.66, held at
3.7. No interval gate (two pairs); row bootstrap is reported but is not seed variance.

Cheapest falsifier and decision changed: two same-job pairs at 100K (about 6.8 T4 device hours
per pair). If it fails, the multi-hop path mechanism is closed for GPT-T at this contract and
a 500K triplet (about 35-40 T4 hours, from the 21.1 h two-arm init-noise kernel) is not spent. If it passes, E2 is released.
Cost/time are estimates, not measurements, until retrieval.

## E2 (500K) frozen recipe, submitted only after E1 passes

Arms: `ref_s47`, `mh_s47` (same-job pair, init s47), `mh_s48` (init s48), recipes in `e2/*`,
copies of the accepted 500K PairNorm recipe (60 passes, 234,360 steps) with only the init and mode
differing. Baseline pool: existing 500K PairNorm inits 43-46, tail-holdout mean 107.94 meV.

- Metric: tail_holdout_v1 MAE, each selected checkpoint scored once, offline. Official validation not used.
- Precondition: `ref_s47` tail MAE in [105.3, 110.6] meV; otherwise INCONCLUSIVE (infrastructure).
- Pass: mean of candidate tails (`mh_s47`, `mh_s48`) <= 107.94 - 3.0 = 104.94 meV.
- Also produce `--residual-report` (descriptive; no gate).
- Fail: NEGATIVE_UNDER_CONTRACT; no adoption, full training or retry.

## Cost and authority

E1: two kernels, estimate 3.4 wall hours each (reference arm about 2.8 h; candidate arm
slowed by an estimated 15-20%), 2 arms x 3.4 h = 6.8 T4 device hours per kernel, 13.6 total,
inside the 30 Kaggle hours the user authorized for evidence search. Wall ceiling 4.5 h per kernel.
No retry or automatic continuation; a cap leaves STOP_FOR_COST. E2 estimate about 35-40 T4 hours
and needs its own authority record when released.

## Local pre-checks (not scientific evidence)

BondEncoder sums its three slot embeddings (so mean = sum / 3); PCQM edge_index is stored in both
directions (2,000 of 2,000 sampled graphs); training `shortest_path_cap` is 20; the baseline pool
is BondLocal + pair-update-norm (mode `degree_bond_local_pair_update_norm_ema999`, 500K recipe WD 0.05).
Smoke results (brute-force reconciliation, zero-start equality, speed, two-pass loss) are in STATUS.md.
