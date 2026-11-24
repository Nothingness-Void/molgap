# GPS++-style pure MPNN at 100K: protocol

Frozen 2026-11-24 JST before any Kaggle work. Desktop-owned Track B, branch
`codex/exp/gpspp-mpnn-100k-20261124`, one T4x2 pair on `nothingnessvoid`.

## Question

At equal parameters, equal exposure and the same recipe, does a pure 2D
GPS++-style message-passing network (no attention) beat BondLocal+PairNorm
GPTrans-T on selected EMA 0.999 development MAE?

A 100K win is necessary, not sufficient: earlier gains shrank to 0.21-0.37x at
500K. This run is a cheap falsifier plus a speed and implementation check, not a
scale decision.

## Arms (one job, one arm per T4)

- `pair_update_norm`: same-job control. GPTrans-T BondLocal+PairNorm from
  `gptrans.pt`, recipe `experiments/pcqm_gptrans_norm_ema_100k_20261104/recipe.json`
  unchanged. Known reference: selected EMA dev MAE 139.440 meV (pass 42), bit-reproducible at seed
  42. The 156.627 meV V4 run is not the comparator. The control still runs in the
  same job.
- `gpspp_mpnn`: standalone model, same recipe bytes, seed 42, no auxiliary loss
  (the control has none). The construct hook seeds weights from `recipe.seed`
  (same route as the accepted standalone candidates).

Fixed recipe for both: split `pcqm_fixed_100k_v1` (100K train, 50K development),
AdamW lr 1e-3, wd 0.05, 4 warmup passes then 60-pass cosine to 1e-6, batch 128, 46,860
steps, FP32 without TF32, EMA 0.999 plus auxiliary 0.9999, gradient clip 1.0,
`allocation_wall_limit_seconds` 14,400, cleanup reserve 180 s, T4 hours ceiling 8.
Official valid, test and tail_holdout_v1 are not read.

## Candidate architecture (`src/molgap/gpspp_mpnn.py`)

- Pure 2D. OGB atom and bond encoders into atom width `d_n` and bond width `d_e`.
  Molecule vector `g0` = learned vector + Linear(mean atom embedding).
- RWSE16 (diagonal of P^k, k=1..16, P = D^-1 A) is computed in the model from the
  batch edges under no_grad and added to the atom embedding through one Linear.
  No cache, no LapPE, no 3D.
- 12 pre-LN residual layers, GELU two-layer MLPs with expansion 2, no dropout:
  - Bond: `e' = e + MLP(LN[x_i, x_j, e_ij, g])`.
  - Atom: `x' = x + MLP(LN[x_i, sum_in e', sum_out e', sum_nb x_old, g])`.
  - Molecule: `g' = g + MLP(LN[g, sum_atoms (x'-x), sum_bonds (e'-e)])`.
- Readout: `MLP(LN[g_L, mean x_L, sum x_L])` to one scalar; shared normalized-gap L1
  objective.
- Scatter sums use `index_add_`, the primitive BondLocal already uses under
  deterministic FP32. No custom CUDA or Triton, no `torch.compile`.
- Widths `d_n : d_e : d_g = 4 : 2 : 1`. **Frozen `d_n = 192`** (`d_e` 96, `d_g` 48).

### Parameter match

Both counts are read from the built models: control 5,871,201; candidate 5,819,425
(-0.88%, inside the +/-5% rule). The plan's earlier 5.25-5.28M control estimate was
wrong; the measured count governs and moved `d_n` from the planned 176-184 to 192
(184 gives 5,348,053, -8.9%, outside the rule).

### Known asymmetries (part of the backbone package, not a bug)

- Candidate has in-model RWSE16; the control has shortest-path distance features.
- Candidate has no attention and no dense pair state; the control has both.
- Candidate has direct bond-to-molecule and molecule-to-bond paths. A later
  ablation (`gpspp_mpnn_nobg`, both paths removed, same widths) is not built and
  not run here; it needs its own addon change and a passed nomination first.

## Gates (frozen before submit)

Primary metric: selected EMA 0.999 development MAE on the 50K development rows.
Effect = `gpspp_mpnn` minus same-job `pair_update_norm`. Uncertainty: 2,000-draw
paired row bootstrap, 95% interval.

| Seed-42 effect (meV) | Verdict |
|---|---|
| `<= -4.6` and CI upper `< 0` | Nominate. An ablation kernel is allowed. A 500K run needs a separate decision. |
| `-4.6 < effect <= -1.4` | Borderline. Run one pre-registered s43 pair (control + candidate). Nominate only if the mean of the two effects is `<= -3.0` and both CI uppers are `< 0`. |
| `> -1.4` | Close as POSITIVE_BELOW_GATE (if negative) or NEGATIVE_UNDER_CONTRACT. No s43, no ablation, no 500K. |

Kill rules: development EMA still worse than the control at pass 30 and at the end,
or a late fade as seen in `real_tri_pair`.

Cost: report measured candidate/control wall time. A ratio above 1.7 risks the
14,400 s cap and gives STOP_FOR_COST; no continuation without explicit user authority.

Replacement value exists only if a later 500K same-job pair against
`pair_update_norm` gives `<= -3.0` meV with CI upper `< 0`, followed by one tail
score. A 100K effect weaker than about -8 meV probably will not clear 500K.

## Acceptance path (decision 4.1(b), pre-registered)

Every recent standalone candidate hit `SCIENTIFIC_CONTRACT_MISMATCH` on
`runtime_certificate_scope`. This protocol takes route (b): the non-strict
mechanical closure path counts for nomination when `runtime_certificate_scope`
is the only mismatch and every recipe, data, row, target and seed field matches.
Route (a), fixing the hash split, is a separate infrastructure commit and is not
done here. Any other mismatch voids the comparison.

## Not run

- S2GNN: NO_TRAIN, deferred. The paper trails GPS/GRIT, local spectral evidence is
  null or negative (Gaussian spectral -0.07, TB-EigenGNN +12.4 meV and 4.1x slow,
  GRIT-SPSE +14.5, Cutlass-GT +11.3), and it needs an eigenpair cache that does not
  exist. It takes no slot.
- Prior for this run is weak: K1 GPS++ adapter -0.206 meV worse, GPS9 -0.630 worse.

## Slots

A now: kaggle1, s42 pair. B only in the borderline band: kaggle2, s43 pair. C only after
nomination: kaggle3, `gpspp_mpnn` against the ablation. No S2GNN, GraphGPT,
ablation before a pass, 500K or filler.
