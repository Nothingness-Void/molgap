# Protocol: K1 selective MoSE residual on fixed PCQM 100K

## Question

Can K1 retain its stable RWSE16 path while learning a small, node-selective
residual from MoSE31, instead of forcing one structural encoding to replace the
other?

This is a distinct information-fusion question. It does not reopen the closed
MoSE replacement or hidden-normalization questions and is not plain feature
concatenation.

## Evidence

- Unnormalized MoSE replacement was directionally favorable overall, with its
  strongest effect in the lowest count-magnitude quintile, but regressed in the
  highest quintile.
- Hidden BatchNorm regressed in every quintile relative to unnormalized MoSE,
  ruling out another normalization variant.
- The MoSE study reports mixed GPS results for combined RWSE+MoSE but gains for
  several other architectures. K1 uses sparse molecular-slot exchange rather
  than the paper's dense GPS attention, so the interaction remains unresolved.
- Learnable Structural and Positional Encoding work supports keeping structural
  and positional streams distinct rather than irreversibly mixing them at the
  input.

## Single mechanism

The K1-v4 RWSE16 architecture path remains unchanged and is initialized under
the same seed contract. All parameters remain trainable. The accepted
MoSE31 cache supplies a separate residual stream:

```text
RWSE16 --------------------> K1 structural encoder ------------+
                                                               |
MoSE31 -> 31x64 -> SiLU -> 64x192 (zero initialized) -> gate --+-> K1 blocks
                         mean(MoSE31) -> 1x16 -> SiLU -> 16x1 -> sigmoid
```

The zero output projection makes the complete candidate exactly equal to K1-v4
at initialization. The scalar gate is node-local and uses only inference-time
MoSE magnitude. It cannot read the target, graph-level labels, geometry or a
protected role. The candidate adds 14,577 parameters (3,673,394 total).

## Frozen V5 screen

- Reference: immutable `neural_atom_k1_v4` accepted payload.
- Candidate: `neural_atom_k1_rwse_mose_residual_gate`.
- Fixed official-train-derived PCQM roles: 100,000 train / 50,000 development.
- Accepted RWSE16 graph cache and accepted MoSE31 cache; no graph rebuild.
- Seed 42; deterministic FP32; TF32 disabled; physical BS128; `drop_last`.
- AdamW `4e-4`, weight decay `1e-5`, clip `1.0`; cosine, 40 epochs.
- Identical row order, target transform, loss, selection and sample exposure.
- Official validation, test-dev and test-challenge remain unread.

## Gate and stop condition

Mechanical acceptance requires complete hashes, exact-K1 initialization,
separate finite gradients in both residual and gate modules, accepted runtime
and cache identities, and equality of all non-intervention contract fields.
Promotion requires at least `0.003 eV` gain over K1-v4 and a favorable paired
row interval. A pass grants shortlist status only. A miss closes this selective
residual question without another seed, wider residual, gate variant, 500K,
protected role or desktop handoff.
