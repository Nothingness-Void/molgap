# GPT-T with DFT distances as a training-time input (P1): protocol

Frozen 2026-11-23 JST before any diagnostic or training. Desktop-owned, Track B,
branch `codex/exp/gptrans-dft-input-p1-20261123`, forked from
`codex/exp/gptrans-dft-distance-aux-20261121` (429cb0225). Source of the design:
the court ruling `final_ruling-1123.md`, section 5 ("P1 execution spec", agreed
by both sides and accepted by the judge). The ruling is not a training grant;
the user released this build and local rehearsal only. **Nothing here submits a
job.** An Opus pre-submit review and an explicit release come first.

## Contract class: 3D-supervised training, pure-2D inference

Training reads the DFT coordinates that ship with the official PCQM4Mv2 training
set. For a share of samples the coordinates enter the model as distances, added
to the initial pair state. **Inference never reads coordinates**: no sidecar, no
3D construction, no ETKDG. This is a different contract from strict pure 2D. Its
results are booked in the separate 3D-supervised ledger and never ranked against
strict 2D results as if they were the same contract. OGB allows DFT 3D in
training and forbids explicit 3D at test time (ruling section 3).

## Evidence review (RML, 2026-11-23)

- `lookup --mechanism dft`: one trajectory, `T-pcqm-gptrans-dft-distance-aux-20261121`
  (C1). It used DFT distances as an auxiliary **label** only. 100K -3.70 meV,
  500K -0.14 meV (INCONCLUSIVE; interval crosses 0). Its reopen condition asks for
  new user authority for another distance arm and for coordinates beyond row
  499,999. This work has that authority (user brief, 2026-11-23) for the P1
  design and for dev-row coordinates used as scoring input only.
- Ruling prior: tail median about -0.8 meV expected; 15-20% chance to reach the
  -1.5 meV full-scale line. Main risks: scale erosion (100K to 500K keeps 4-37%
  of a gain in this project), 2D starvation, pair-state capacity (32 channels).
- Nearest closed work with a different mechanism: ETKDG geometry as input
  (`TB-gptrans-geometry-distance-*`, negative), Noisy Nodes (100K positive, 500K
  negative), C1 (above).

## Hypothesis and falsifiers

- Deficiency: GPT-T infers conformation-dependent effects from topology only.
  C1 gave it a distance *label* and gained nothing at 500K. Transformer-M reports
  that most of its 3D gain comes from the 3D *input* mode, not the target.
- Change: DFT distances enter the pair state (training only, for 60% of samples),
  with modality dropout so the model also learns a good 2D path, plus coordinate
  noise and a denoising loss.
- Alternative explanations for any gain: extra parameters or modality dropout
  acting as a regularizer (the zeros arm tests this); seed noise; scale erosion.
- Cheapest falsifier already run: C1 at 500K. The cheapest new one is gate (i):
  if the model cannot use true distances at all (3D mode not 10 meV better than
  2D mode), the mechanism is dead and P1/P2/P6 close together.

## Arms (6 runs, 500K, A100)

All six share one recipe frame: GPT-T BondLocal pair-update-norm, batch 128
(FP32), learning rate 1e-3, EMA decay 0.999, weight decay 0.05, 60 passes,
warmup-cosine, train rows [0, 500,000) of the fixed `ogb-train-500k-scnet-v1`
store, 234,360 optimizer steps. Within a pair the three arms share the init
file, the data order (sampler seed) and the dropout seed.

| Arm | Model | Distance input | Denoising loss |
|---|---|---|---|
| control | existing GPT-T pair-update-norm, pure 2D | none | none |
| P1 | control + distance branch + denoising head | true DFT distances (noised) | 0.1 x L1, clean distances |
| zeros | same structure and mode schedule as P1 | all-zero RBF features | none |

| Pair | Recipe seed (data order, dropout) | Init file | Same as |
|---|---|---|---|
| 42 | 42 | `init/s44.pt` (state sha ff949650...) | DFT-aux c500/d500 |
| 43 | 43 | `init/s43.pt` (state sha c95da412...) | DFT-aux c100/d100 init |

Decision (2026-11-23, no user answer available): the ruling's "init seed pairs
42 and 43" is read as the recipe seed. Pair 42 is exactly the c500/d500 setup
(init s44, seed 42), so the pair-42 control repeats finished work apart from the
readout (below). Pair 43 uses the second init file already in the DFT-aux
`init/` folder (s43.pt) with seed 43. The two pairs differ in both init and
seed. Reviewers may swap in other inits; nothing else changes.

Chain order (one serial chain, ruling and user brief): pair 42 control, P1,
zeros, then pair 43 control, P1, zeros.

## P1 constants (frozen here; the ruling left these open)

- **RBF.** K = 64 Gaussian centres, evenly spaced from 0 to 10 A (spacing
  10/63 A). Width sigma = 10/63 A: `phi_k(d) = exp(-0.5 ((d - mu_k) / sigma)^2)`.
  No clipping: a distance above 10 A just excites no centre.
- **MLP.** `Linear(64,32) -> GELU -> Linear(32,32)`; output channels equal the
  32 pair channels. The last layer starts at zero, so the branch adds exactly
  zero at step 0. Weights are drawn from the module's own generator (the
  global RNG stream, hence dropout and data order, matches the control).
- **Where.** The MLP output is added to the initial pair state (after SPD and
  bond encodings, before block 0) at atom-atom positions. All real atom pairs
  including the diagonal (d = 0) get a feature. The graph-token row and column
  get none. Both directions get the same value (distance is symmetric).
- **Modes, per sample.** 2D 0.40, 3D-only 0.20, 2D+3D 0.40, drawn independently
  per sample. "2D" = no distance input and nothing masked. "3D-only" = distance
  input, with the SPD and the bond encoder outputs of that sample set to zero.
  "2D+3D" = distance input, nothing masked. A sample without valid DFT geometry
  is always 2D (the draw is made first, then overridden, so all arms see the
  same schedule).
- **Noise.** Gaussian, 0.2 A per axis, independent per atom and per step,
  added to the DFT coordinates before distances are computed. The denoising
  target uses the clean distances.
- **Denoising loss.** 0.1 x mean L1 (A) between the head output and the clean
  distance over real i<j pairs of samples with distance input (3D-only and 2D+3D;
  not 2D samples, which would re-test C1). The head is the C1 head (LayerNorm,
  Linear, GELU, Linear on the final pair state, symmetrized). The Gap loss is the
  control's normalized Gap L1 over all rows of every batch.
- **Sampling.** One CPU generator per batch, seeded from SHA-256 of
  (recipe seed, the `source_idx` values of the batch rows). It draws modes first,
  then noise. It never touches the global RNG, and it needs no checkpoint state:
  a resumed run draws the same numbers as an uninterrupted one. P1 and zeros use
  the same code path, so they see identical modes.
- **Zeros arm.** Same modules, same masking, same schedule. The RBF features are
  multiplied by 0, so the MLP sees an all-zero vector. No denoising head, no
  denoising loss.
- **Inference and evaluation.** No distance state is set. Hooks return their input
  unchanged, so the forward pass is the control's forward pass over the same
  shared weights. Nothing reads a sidecar.

Caveat for review: in 3D-only mode the model still sees the true bond graph
through the degree encoders and the BondLocal bond messages. The spec masks
only the SPD and bond encodings, and this protocol keeps it that way. "3D-only"
therefore means "no pair-level topology encoding", not "no topology".

## Data and roles

- Train: official train rows 0..499,999 with the C1 v3 sidecar (pinned, as C1;
  one row, 140686, has no valid geometry). The zeros arm pins the same sidecar so
  all three arms load rows the same way (the control pins none).
- **Readout role: development rows 500,000..549,999 only.** The development rows
  need DFT geometry for the 3D-mode score of gate (i). Dev geometry is built with
  the same builder (`platforms/local/pcqm_dft_geometry_sidecar.py`) from the same
  SDF archive and stays local: it is a scoring input, never a training or
  selection input, and it is never uploaded to IMS. Topology match for the dev
  rows must be at least 99.9% (identity + isomorphism over 50,000 rows); the
  builder's own limit is 1% and is not the requirement.
- The tail holdout (rows >= 3,250,000), official valid and test are not read.
- The in-job run has no development role (see Readout). The job writes
  `selected_model.pt` (the final EMA); dev scoring is offline.

## Readout

Final EMA at pass 60, no best-epoch selection. The recipes use selection
`final-ema` with no in-job development evaluation, so a best epoch cannot exist.
All dev MAEs (2D, 3D-only, 2D+3D) come from the offline scorer on the retrieved
`selected_model.pt`. 2D numbers use all 50,000 dev rows. Gate (i) uses the rows
with valid geometry for both modes.

## Gates (for the later analysis; fixed now)

Deltas are meV, dev MAE, final EMA. "Both pairs" means pair 42 and pair 43 each.

- (i) P1 3D-only dev MAE <= P1 2D dev MAE - 10, both pairs. 2D+3D is descriptive.
  If it fails, P1, P2 and P6 close together.
- (ii) P1(2D) - control <= -2.0 on the mean of the two pairs, and negative in
  both pairs.
- (iii) P1(2D) - zeros(2D) <= -1.0, both pairs.
- (iv) Measured **median step time** of P1 and of zeros <= 1.5 x control, and total
  accelerator wall clock <= 2.0 x control, in both pairs. Median step time = the
  median, over consecutive `train_log.jsonl` rows of one process, of (elapsed
  seconds / optimizer steps between the rows); one row per pass, 3,906 steps per
  pass. A row after a resume restarts the clock and is skipped. Wall clock = the
  sum of those intervals over the item. Sidecar build time is CPU time booked
  separately.
- All of (i)-(iv) must pass to release a full-scale run. The full-scale gate is
  a single-init tail delta <= -1.5 meV against s42/s43, with one more init
  before adoption. Not part of this experiment. No tail score is released here.

Paired 2,000-draw bootstrap over the 50,000 dev rows is reported next to each
delta. It measures row noise only. It is not init or seed variance.

## Budget and cost

Measured on A100 (C1 record): a 500K control run took 29,984 s (8.33 h), with
an in-job development pass each epoch; the C1 head run took 31,043 s (1.035 x).
**A100 hours of P1 and zeros are not measured.** Local RTX 5060 marginal step
time (`step_time_record.json`, 2 rounds, 300-step difference): P1 1.031 x and
zeros 1.009 x the control. Per-item `budget_s` in the serial spec are estimates
used only for time skips, not limits: 30,000 s for a control, and
30,000 x local ratio x 1.05 rounded up to 100 s for P1 (32,500) and zeros
(31,800). Estimated A100 time per run, from the C1 control and the local
ratio, is about 8.3 h (control), 8.6 h (P1) and 8.4 h (zeros): about 51 h for the
chain, an estimate only. A job walltime is at most 24 h, so the chain needs three
jobs resumed with exit code 2 (job 1: pair-42 control and P1, zeros skipped for
time; job 2: zeros and pair-43 control; job 3: pair-43 P1 and zeros). A
skipped-for-time item is resumed, not redone. Gate (iv) is read from the A100
`train_log.jsonl`, not from the local ratio.

## Attribution plan (before another module in this family)

Read after retrieval, in this order:

1. Gate (i). Fails: the model does not turn true distances into a 10 meV gain, so
   a distance input at this capacity is dead; stop the family.
2. Gate (ii) with (iii). (ii) passes and (iii) fails: the gain comes from
   modality dropout or extra parameters, not geometry. (ii) fails with (i) passing:
   2D starvation (only 40% of samples train the pure-2D path) or pair-state
   capacity (32 channels); check P1(2D) against control per pair.
3. P1(2D) - P1(2D+3D) gap: how much geometry the model uses when given it.
4. Scale erosion: no 100K companion exists in this experiment, so a null cannot be
   split into "needs more data" vs "does not transfer"; say so, do not infer it.
5. Seed context: two pairs differ in init and seed; a pair-to-pair sign flip is
   noise, not a mechanism.

## Caveats (ruling section 6, plus build facts)

- Scale erosion, 2D starvation, capacity and the zeros reading are ruling risks.
- The pair-42 control repeats the C1 c500 setup apart from the readout; a
  matching final EMA is an extra determinism check, a mismatch is not a failure.
- Only the distance branch separates P1 from zeros. The zeros arm still pays the
  masking and mode schedule, so it tests the whole regularizer package.
- Dev geometry comes from the same third-party mirror archive as C1 (identity
  proven by topology match, not by an official hash).
- 3D-only mode keeps topology through degree encoders and BondLocal (above).
