# Protocol: K1 InducedPairToken 100K

## Question

Can four learned source and target inducing slots retain PairToken's useful
learned selection, cross-node relation and per-pair normalization while
removing its explicit all-atom-pair cost?

## Single change

The K1-v4 backbone and layer-6 insertion point are unchanged. Instead of
materializing every ordered atom pair, the candidate learns four source and
four target summaries, forms the 16 ordered latent pairs, normalizes each pair
across its 32 channels, then selects one relation token. The return projection
is zero initialized, so the candidate is exactly K1 at initialization.

## Frozen V5 screen

- Fixed official-train-derived roles: 100,000 train / 50,000 development.
- Immutable K1-v4 reference payload; no reference retraining.
- Seed 42; deterministic FP32; TF32 disabled; physical BS128; `drop_last`.
- AdamW `4e-4`, weight decay `1e-5`, clip `1.0`; cosine, 40 epochs.
- No geometry, teacher, target-derived feature or protected role.
- Official validation, test-dev and test-challenge remain unread.

Promotion requires at least `0.003 eV` gain over immutable K1-v4 with a
favorable paired interval. A miss closes this induced-pair configuration
without a second seed or scale bridge.
