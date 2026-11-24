# DFT distance auxiliary: 100K gain does not survive at 500K

2026-11-23, Track B, 3D-supervised training with pure-2D inference (separate
contract from strict 2D). IMS run `gptdft-chain-20261121` (job 2111727, resume
job 2134349) finished COMPLETE. Rules: [protocol.md](protocol.md), "Decision
rules (after retrieval)".

**Verdict.** 100K: positive under this contract. 500K: **INCONCLUSIVE**
(delta -0.14 meV, between 0 and -3.0; the interval crosses 0). The 500K gate is
not met, so no tail score is released and no leaderboard row is added. Nothing
is adopted; no full-scale run follows.

## Development MAE (rows 100,000..149,999 and 500,000..549,999, selected EMA)

| Item | Dev MAE (meV) |
|---|---:|
| c100 | 139.185 |
| d100 | 135.485 |
| c500 | 100.376 |
| d500 | 100.236 |

Paired bootstrap, 2,000 draws over all 50,000 dev rows (same indices for both
arms, seed 20261123). The protocol text says "2000-row"; this record uses 2,000
draws.

| Pair | Delta (meV) | 95% interval (meV) | Gate |
|---|---:|---|---|
| d100 - c100 | -3.700 | [-4.600, -2.772] | <= -3.0 and upper < 0: met |
| d500 - c500 | -0.140 | [-0.687, +0.384] | <= -3.0 and upper < 0: not met |

The scorer rebuilt each item from the retrieved `selected_model.pt` and matched
the job's own dev MAE within 1e-4 eV for all four items (largest gap 6e-8 eV).
It built no sidecar and passed no coordinate field, so this is also the
protocol's inference check: the d-arm checkpoints score with pure 2D input.

## C1 prerequisites (court ruling; separate from the gates above)

- (a) Aux gain at 500K on dev: -0.140 meV; needs <= -1.0. **Not met.**
- (b) Head distance error, L1 in angstrom over real heavy-atom pairs i<j,
  pair-count weighted, 50,000 rows each:

| Checkpoint | Train rows 0..49,999 | Held-out rows 150,000..199,999 |
|---|---:|---:|
| d100 | 0.2292 | 0.2423 |
| d500 | 0.2050 | evidence missing |

Needs < 0.20 on held-out rows. d100 held-out 0.2423: **not met** (train-held-out
gap 0.0131). d500: the v3 sidecar covers only rows 0..499,999, which is the
whole d500 train role, and the protocol gives development rows no coordinates.
No unseen row has coordinates, so the held-out figure is **evidence missing**.
Even the d500 train-row figure (0.2050) is above 0.20.

## Attribution

- Retention: 0.140 / 3.700 = 3.8% of the 100K gain remains at 500K. The 100K
  gain is outside the paired interval at 500K.
- Noise context: the paired 500K interval is [-0.687, +0.384] meV, so the
  delta cannot be told from zero with one init and one seed. The protocol cites
  seed noise of about 1-2 meV at 100K; the memory notes give a 1.63 meV init
  spread for K1 at 100K (not GPT-T, not 500K). The brief cites an init-noise std
  of 0.099 meV at full scale; no RML source for that figure was found here, so
  it is only quoted. The 100K delta of -3.70 meV is larger than the 1-2 meV
  noise figure. Ensembling is not tested here.
- Mechanism reading (not proven): the pattern matches the earlier Noisy Nodes
  auxiliary (positive at 100K, negative at 500K): a regularizer helps the small
  run and fades when more data does the same job.
- Cost: A100 wall per item (s) and ratio d/c: 100K 7,319.1 / 6,887.6 = 1.063;
  500K 31,043.4 / 29,984.3 = 1.035. Total item wall 75,234.4 s (20.90 h);
  estimate was 11.6 h. c500 took 29,984 s against a 15,300 s budget and d500
  31,043 s against 22,000 s (budgets are time estimates, not limits).
- Determinism: c100 is byte-identical to j2 of job 2097505 (selected model
  sha256 3840b378...bccb and dev MAE 0.13918505609035492 equal). c500 vs j5:
  not comparable, j5 of 2097505 is paused (step 124,992, dev 0.10276), not
  finished.

## Records and custody

- Tail rows, official valid, test and the leaderboard file were not read or
  changed. Tail scoring is not triggered.
- The stage record `output/train/stage_status.json` (47 bytes) holds only
  `{"stage": "train", "status": "complete"}`: the paused job's worker came from
  the earlier launcher and wrote no `outputs_sha256`. `submit_from_windows.py
  --retrieve-to` therefore refuses ("no complete worker digest record"). Fetch
  used [retrieve_legacy_outputs.py](retrieve_legacy_outputs.py): remote
  `sha256sum` over 25 files (checkpoints, logs, summary; the fixed-dataset
  mount copies are skipped), local hashes equal. Receipt:
  `D:/molgap-artifacts/retrieved/gptdft-chain-20261121/training/retrieval_legacy.json`.
- Scoring: [score_dev_and_distance.py](score_dev_and_distance.py), output
  [results.json](results.json); per-row dev errors stay in
  `D:/molgap-artifacts/retrieved/gptdft-chain-20261121/analysis/final/`.
- Job 2134319 (run `gptdft-d500-20261123`) was cancelled before it started; the
  `*_d500.json` spec files are unused.
- Payload `D:/molgap-artifacts/ims_payloads/gptdft-chain-20261121` is kept.

## Reopen conditions

Only with new user authority: a larger weight or schedule for the distance head
is not tested (one arm, no lambda grid by protocol), and held-out distance
error at 500K needs coordinates for rows outside 0..499,999.
