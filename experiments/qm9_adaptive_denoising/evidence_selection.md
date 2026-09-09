# Adaptive local denoising selection — 2026-09-09

This record answers one question: which untested server-side mechanism has the
highest prior probability of surviving the Track C to Track B funnel?  It does
not report a result and does not authorize a scale-up.

## Local evidence matrix

| Mechanism family | Strongest accepted evidence | Interpretation |
|---|---|---|
| Persistent real-bond state | Large direction-consistent gains on QM9 and three PubChemQC seeds | Retain EdgeState as the downstream backbone |
| Dense pair/global attention | PairGPS lost to sparse EdgeState; full GPS lost to GraphState at three PCQM seeds | Do not add another dense attention path |
| Additional topology states | Ring, PNA, retention, LapPE, relative-value path, contact, body-order, and GAPE-lite were neutral or negative | More topology is unlikely to supply the missing signal |
| Sparse path/bond/component states | Hop path, directed bonds, and ComponentState were only weak positive and below their material gates | Useful diagnosis, insufficient promotion prior |
| Local chemical pretraining | The 10/30 atom/bond/group allocation improved its paired scratch arm by 0.00234205 eV | Local supervision can shape the encoder, but the exact schedule is closed |
| Graph-level pretraining | Composition/hashed-fragment and geometry-histogram objectives failed | Do not reconstruct global summaries |
| ETKDG geometry | Distance plus angle bottom fusion improved its paired comparator in all three PCQM seeds | Geometry can add target-relevant information when injected locally |
| Cheap geometry augmentation | Scalar angle/dihedral injection and hard contact/body-order additions failed | The next geometry question must change the learning objective, not add static channels |

## Literature intersection

DenoiseVAE, Frad, SCD, and SliDe all make the supervision local to atoms or
relative coordinates.  DenoiseVAE is the smallest causal step because it asks
whether the corruption scale should depend on the molecule; its paper reports
direct PCQM4Mv2 Gap evidence with a compact model.  The public implementation
is not reusable as-is because its geometry and software contract do not match
MolGap, so no external checkpoint, DFT coordinate, GEOM row, or paper metric is
imported.

The selected question combines only conclusions already supported locally:

1. keep the accepted sparse EdgeState backbone;
2. retain the accepted sparse wedge state and deterministic ETKDG
   distance/angle bottom fusion;
3. replace failed graph-level geometry summaries with atom-local vector
   denoising; and
4. isolate adaptive noise from ordinary fixed-noise denoising.

## Why other candidates are not next

- Fragment/BRICS work belongs to the independent desktop branch and is outside
  this server queue.
- A `(2,1)-GT`, dense atom/bond Transformer, or new GPS variant restores costly
  attention despite repeated local evidence that attention allocation is the
  problem.
- Another path, ring, wedge, directed-bond, PNA, or retention variant would
  retune a closed information family.
- SliDe and anisotropic covariance noise add multiple mechanisms and much more
  compute before fixed versus adaptive scalar noise has been isolated.
- The positive 10/30 hierarchy schedule is not rerun: its own frozen decision
  closed allocation search on that validation role.

## Admission decision

Admit one QM9-30K paired three-arm screen: scratch, fixed isotropic local
denoising, and molecule-adaptive isotropic local denoising.  All arms must share
one Kaggle task, accepted role identities, downstream architecture,
initialization, seed, optimizer, schedule, FP32 precision, physical batch 128,
and 40 encoder passes.  A result below the material gate closes the exact
adaptation without another seed or hyperparameter retry.
