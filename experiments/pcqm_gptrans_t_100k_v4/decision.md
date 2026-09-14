# GPTrans-T 100K V4 decision — 2026-09-14

## Question

Is the compact published-shape GPTrans-T a competitive pure-2D reference for
the accepted PCQM4Mv2 100K/50K architecture funnel?

## Accepted evidence

Kaggle2 kernel `kaseichou/molgap-pcqm-gptrans-t-v4-s42`, version 4, completed
and passed no-inference acceptance. It executed 60 epochs, 46,860 optimizer
steps, and 5,998,080 sample presentations on one P100. Source, immutable graph
manifest, frozen initialization, runtime certificate, final model, aligned
development predictions, and completion hashes were accepted. Official
validation and all test roles remained unread.

| Model | Development Gap MAE | Parameters | Best epoch |
|---|---:|---:|---:|
| GPTrans-T `12x256/pair32` | 0.1566272043 eV | 5,246,817 | 59 |
| frozen K1-v4, contextual reference | 0.1413736343 eV | 3,658,817 | 39 |

GPTrans-T is contextually worse by `0.0152535700 eV` while using 43.39% more
parameters. This is not a strict causal paired comparison because the published
GPTrans optimizer, schedule, EMA selection, and sample exposure differ from
K1's frozen V4 contract. It establishes poor 100K sample efficiency, not an
architecture-wide rejection.

## Attribution

The final training MAE was `0.1037558007 eV`, leaving a development gap of
about `0.05287 eV`. Development EMA improved through the final epoch, but the
learning rate had already reached `1e-6`; extending training would change the
frozen exposure and mainly chase an EMA/generalization lag rather than test a
new architecture.

The compact adaptation retains GPTrans node-to-node, node-to-pair, and
pair-to-node propagation, a virtual graph token, and shortest-path pair
encoding. It deliberately omits the two project mechanisms with the strongest
bounded-data evidence: RWSE16 and a sparse local persistent real-bond EdgeState
MPNN path. Dense all-pairs propagation therefore spends more capacity and
memory while supplying a weaker local chemical inductive bias at 100K. This is
consistent with the earlier QM9 PairGPS result, where dense pair-state repairs
lost to sparse persistent EdgeState.

The earlier accepted SCNet 500K result used the same adapted core and reached
`0.1039478481 eV`, beating its matched ESGPS6-304 scratch comparator by
`0.0037238458 eV`. It was also numerically `0.0009112507 eV` below the frozen
K1 500K score, although those K1 and GPTrans runs used architecture-specific
optimization contracts and therefore do not form a strict causal pair. The
100K result is consequently a false-negative screen for this scale-sensitive
architecture rather than evidence against its 500K/full-data potential.

## Decision

Mechanically freeze this run as the only GPTrans-T 100K V4 reference and do not
tune its width, depth, EMA, learning rate, or exposure on the consumed role.
Do not use the 100K score to close GPTrans-T: the accepted 500K evidence retains
it as a full-scale candidate. The desktop-owned matched full K1/GPTrans study is
the appropriate discriminator; server must not submit a duplicate successor.

One of the separately authorized maximum three Kaggle2 scientific rounds is
consumed. No immediate Kaggle successor is released because the next relevant
scale question is already under the desktop full-run contract.

## Evidence pointers

- No-inference acceptance: `results/kaggle_acceptance.json`.
- Retrieved record:
  `platforms/_records/kaggle/training/pcqm_gptrans_t_v4_s42_v4/`.
- Accepted 500K evidence: `../pcqm_gptrans_t_500k/results/decision.md`.
