# Protocol: K1 RecurrentPairBridge 100K

## Question

Can the pair recurrence, pair-to-node flow, and pair-update pre-normalization
that were individually useful in GPTrans add transferable relational context
to K1 without replacing K1's successful sparse local path or one-slot global
exchange?

## Evidence basis

- K1-v4 is the frozen strict 100K reference.
- GPTrans flow ablation showed that removing pair recurrence or pair-to-node
  flow worsened development MAE.
- GPTrans pair-update pre-normalization improved its fixed 100K reference.
- K1 PairToken proved that ordered cross-node pairs can help at 100K, but its
  single layer-6 graph token lost materiality at 500K.
- K1 allocation, scalar-gate, multi-slot, readout, and induced-pair routes are
  already closed and are not repeated here.

## Single intervention

The complete K1-v4 backbone and its layer 3/6/9 one-slot exchanges remain
unchanged. A 32-channel ordered-pair state is updated at those same three
layers. Each new pair update is normalized before recurrent addition; the
accumulated state selects source atoms, and the current normalized pair update
is returned directly to each target atom. Layer-specific return projections
are zero initialized, so the candidate is exactly K1 at initialization.

No geometry, pretraining, teacher, fusion, extra K1 slots, scalar global gate,
or protected role is used.

## Frozen screen

- Fixed official-train-derived roles: 100,000 train / 50,000 development.
- Seed 42; deterministic FP32; TF32 disabled; physical BS128; `drop_last`.
- AdamW `4e-4`, weight decay `1e-5`, clipping `1.0`; cosine, 40 epochs.
- Direct normalized Gap L1; best development checkpoint selected each epoch.
- Exactly 31,240 optimizer steps and 3,998,720 sample presentations.
- Atomic epoch checkpoints and 10-epoch recovery bundles are mandatory.
- Official validation, test-dev, and test-challenge remain unread.

Promotion requires at least `0.003 eV` improvement over the immutable K1-v4
aligned payload and a favorable paired bootstrap interval. Failure closes this
recurrent bridge without a second seed or scale-up. Passing only permits a
separate 500K cost and transfer decision.
