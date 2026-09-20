# Protocol: chemistry-conditioned PairToken

## Question

Can deterministic local chemistry roles improve *which ordered atom pairs* the
single accepted PairToken selects, without adding another token branch or
changing its value/return path?

## Evidence basis

- Original PairToken is the strongest retained 100K K1 relation mechanism:
  development Gap MAE `0.1383300447 eV`, a `0.0030435966 eV` gain over the
  immutable K1-v4 reference.
- Standalone functional-group token exchange is directionally favorable at
  `0.1398314089 eV`, but its `0.0015422255 eV` gain is below the frozen material
  gate. That exact standalone route is closed.
- Their per-row corrections are only moderately aligned (`r ~= 0.485`), while
  both mainly help the hard K1-error tail. Chemistry roles therefore contain
  plausible selection information, but not enough evidence for a second
  parallel output branch or prediction fusion.

## Single intervention

The K1 backbone and original layer-6 PairToken are retained. The existing
12-role incidence sidecar is averaged into a 16-dimensional role vector for
each atom. Bias-free source/target role projections create one role-pair
representation. A zero-initialized role query contributes an additive bias to
the original PairToken pair-selection logits:

```text
layer-6 node states -> original ordered-pair values -----> one PairToken
                         ^
12-role incidence -> role pair -> zero-init logit bias --+
```

The value tensor, pair normalization, token FFN, and return projection remain
the original PairToken mechanism. The role query starts at zero and the token
return projection starts at zero. Initial PairToken selection is therefore
unchanged, and the complete candidate initially produces exactly K1 outputs.
Atoms with no detected role have the exact zero role vector.

This adds 752 parameters over PairToken (23,600 over K1; 3,682,417 total). It
uses no coordinates, SMILES, new target, teacher, pretrained checkpoint,
prediction fusion, or protected role.

## Fixed screen

- Accepted cross-platform PCQM V4 fixed 100K train / 50K development roles.
- Direct Gap target; official validation, test-dev, and challenge remain sealed.
- Seed 42; FP32; TF32 off; deterministic algorithms.
- Physical batch 128; 40 epochs; 31,240 optimizer steps; 3,998,720 sample
  presentations; AdamW `4e-4`, weight decay `1e-5`, cosine schedule.
- No baseline retraining. K1-v4 is the strict frozen V5 reference; original
  PairToken is a supporting mechanism control from the identical contract.

The candidate must meet the prospectively frozen K1 material gate and must not
regress against the original PairToken point estimate. One discovery seed can
shortlist only; it cannot authorize seeds 43/44, scale-up, protected roles, or
desktop handoff.
