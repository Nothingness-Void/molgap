# Exact-2M Encoder Transplant Into Routed-v4

## Decision

**Closed negative.** Keep the production routed dual-GPS v4 unchanged and do
not spend external-evaluation or sealed-set compute on this candidate.

This experiment tested the materially new hypothesis that the exact-2M GPS7
and GPS9 encoders could replace the 500K GPS encoders while retaining the
strong 500K routed-v4 model group. It did not test or invalidate the exact-2M
pure-2D specialist.

## Controlled Setup

- Frozen 500K ETKDG SchNet embeddings and labels: 497,578 aligned rows.
- Fixed split: 398,062 train / 49,757 validation / 49,759 test, seed 42.
- Unchanged heads: gated `FusionHead`, hidden width 192, base input 192+192,
  dual input 384+192.
- Unchanged route: use the dual head when the production base predicts
  `Gap < 4 eV`.
- Paired head initialization seeds: 42, 43, 44.
- The control and candidate heads were retrained with identical optimization.
- The only experimental change was 500K GPS7/GPS9 embeddings versus the
  exact-2M GPS7/GPS9 embeddings on the same 500K prefix.

The retrained 500K control reproduced the production routed checkpoint on the
same test rows: mean control average MAE was 0.084258 eV, versus 0.084283 eV
for the production checkpoint.

## Results

Candidate minus paired 500K control; positive values are regressions.

| routing | target | mean MAE delta (eV) | seed std (eV) | seed values (eV) |
|---|---|---:|---:|---|
| production-fixed | HOMO | +0.005217 | 0.000173 | +0.005103, +0.005416, +0.005132 |
| production-fixed | LUMO | +0.003671 | 0.000105 | +0.003574, +0.003656, +0.003782 |
| production-fixed | Gap | +0.007480 | 0.000215 | +0.007274, +0.007462, +0.007703 |
| production-fixed | average | +0.005456 | 0.000121 | +0.005317, +0.005511, +0.005539 |
| candidate self-route | Gap | +0.007486 | 0.000194 | +0.007285, +0.007499, +0.007672 |
| candidate self-route | average | +0.005449 | 0.000127 | +0.005302, +0.005532, +0.005512 |

The fixed route selected 12,693 of 49,759 test rows. Candidate self-routing
selected 12,543, 12,514, and 12,536 rows, so the regression is not explained
by a large route-frequency shift.

## Interpretation

More pretraining data improved the exact-2M pure-2D specialist, but its learned
representation is not a drop-in replacement inside the 500K GPS+SchNet fusion
system. Both the candidate base and dual heads converged to worse validation
and test errors on the fixed 500K fusion distribution. Preserving the old head
architecture therefore does not recover the lost complementarity with the
500K SchNet encoder.

Any later use of the exact-2M encoders in a hybrid must introduce a different,
explicitly justified alignment or joint-training hypothesis. Repeating this
head-only transplant with external sets is not warranted.

## Artifacts

- Machine-readable metrics: `metrics.json`
- Atomic per-epoch logs and resumable checkpoints: `seed42/`, `seed43/`,
  `seed44/`
- Reusable entry point:
  `scripts/phase8/archive/archive-r07-exact2m-encoder-transplant/train_routed_v4_encoder_transplant.py`

No production registry entry or default checkpoint changed.
