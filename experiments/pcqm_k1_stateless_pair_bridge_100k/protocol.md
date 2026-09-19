# Protocol: K1 StatelessPairBridge 100K

## Question

Does K1 benefit from repeated pair-to-node relational flow at layers 3, 6 and
9 without carrying pair memory across layers?

## Evidence basis

- K1-v4 is the frozen strict 100K reference.
- GPTrans flow ablation supports pair-to-node flow and pair recurrence, but
  their transfer into K1 has not been separated.
- K1 PairToken showed a 100K signal but lost materiality at 500K.
- The paired `RecurrentPairBridge` run changes pair flow and recurrence
  together. This arm isolates the cheaper repeated-flow explanation.
- K1 allocation, scalar-gate, multi-slot, readout, and induced-pair routes are
  closed and are not repeated.

## Single intervention

The complete K1-v4 backbone and its layer 3/6/9 one-slot exchanges remain
unchanged. At those same three layers, a 32-channel ordered-pair update is
normalized, used to select source atoms, and returned to each target atom. No
pair state is retained across layers. Layer-specific return projections are
zero initialized, so the candidate is exactly K1 at initialization.

The arm has exactly the same added parameters and exchange points as
`RecurrentPairBridge`; recurrence is the only mechanistic difference.

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
aligned payload and a favorable paired bootstrap interval. Failure closes
stateless multilayer pair bridging without another seed or scale-up. Passing
only permits a separate 500K cost and transfer decision.
