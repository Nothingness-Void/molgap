# Resonance-invariant K1 representation (RIGR): planning review — 2026-11-06

No training and no submission. After an A/B/C review (proposal, opposition with new computation,
independent ruling that re-ran the motif counts) the direction closes as `NO_TRAIN` / duplicate. No
protected role and no Kaggle quota was used; only fixed-100K train shard 0 and the frozen reference
K1 were read, and no model was fitted.

## What the paper does and does not show

RIGR (ChemRxiv 2025) is a Chemprop D-MPNN featuriser: it drops formal charge, hybridization,
aromaticity, bond type and conjugation, keeps ring membership, and adds the net charge. Its
evidence for invariance is a 50,000-molecule QuantumPioneer heat-of-formation set rich in resonance
radicals. For PCQM4Mv2 (5 splits, 5-initialisation ensembles, a different architecture and split) it
reports RIGR 0.10 and Native 0.09 eV, rounded to 0.01, so the true difference can only be bounded
at roughly 0 to 20 meV; RMSE (0.16 vs 0.15) and R2 (0.980 vs 0.983) also favour Native. No
resonance-form test set exists for PCQM. This is no evidence of a K1 gain.

## Is the problem real in the current inputs? No

K1 consumes OGB `smiles2graph` features of the one dataset SMILES per molecule; the pipeline adds
no resonance standardisation. Rebuilding the 3,000 first train molecules with RDKit reproduced the
stored features exactly. 41.6% have two or more RDKit resonance structures, but those collapse to
the same aromatic graph except for 3.03% (91 of 3,000) with an alternate form whose OGB features
differ, mostly a different formal-charge pattern (188 of 197 alternate forms). On those alternates the
frozen reference's Gap prediction differs by 0.172 eV on average (median about 0, 95th percentile
0.868 eV), but they are enumerated forms the dataset never contains. Across all 50,000 train
molecules each tested motif is written in a single form: charge-separated nitro 1,223, N-oxide 82,
neutral sulfoxide 175, neutral sulfone 76, and zero hypervalent nitro, carboxylate anion, S+-O-
sulfoxide, azide, diazo, nitrile oxide or phosphate P+-O-. The frozen model's (in-sample, affine-
calibrated) error on charge-separated nitro molecules is not larger than on the rest
(difference interval [-0.007, +0.004] eV). A coarse skeleton check found 2,045 of 50,000 molecules
sharing a skeleton, net charge and hydrogen count, with Gap differences of 0.03 to 1.6 eV, and no
sign of duplicate-form inputs. Invariance to inputs the benchmark never presents cannot lower its MAE.

## Distinct question versus existing work? No

Feature deletion removes charge, aromaticity, bond type and conjugation that K1 already uses; the
chemistry-separated local and bond-type local arms were negative and the conjugated hyperedge
stayed below the gate. Training-only consistency over resonance views is the family of the Desktop
dropout-consistency result (+1.529 meV, interval [0.616, 2.349] meV, below the 3 meV gate) and of the
clean-second-view and joint atom-reconstruction screens, with an alternate view on only about 3%
of rows. Canonicalising motif forms does nothing where no second form exists and would create a
minority form for the 2.6% charge-separated nitro molecules.

## Reopening condition

Only if a future input pipeline introduces several resonance or tautomer forms per molecule, for
example a new SMILES source or a 3D-to-graph conversion. Scripts and results for the probe and the
skeleton check are in this directory. Desktop's message-statistics and history-read jobs were not
touched or duplicated.
