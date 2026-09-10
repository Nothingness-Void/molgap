# Fourier-Edge evidence selection

## Failure-driven question

The Neural-Atom experiment showed that simplifying global communication is
directionally useful, but increasing one latent slot to four adds less than
known run variation. Another global mixer, slot count, attention schedule, or
capacity variant is therefore low information.

The surviving one-slot skeleton still uses the original two-linear SiLU MLP in
every persistent EdgeState update. Prior local-operator screens changed the
neighbor aggregator (GatedGCN, edge attention, GEN, PNA, directed bonds) or
added graph relations; they did not test the nonlinear function family inside
the already useful real-bond memory. This is the remaining isolated local
bottleneck with no new molecular input.

## External mechanism evidence

KA-GNN (Nature Machine Intelligence, 2025) replaces fixed-activation MLP
transformations with learnable Fourier-series univariate functions. Its
official implementation defines coefficients with shape
`[sin/cos, output, input, harmonics]` and the released molecular configuration
uses `grid_feat: 1`. The public model also adds 5-angstrom non-covalent edges
and replaces several components at once, so its published gain cannot be
treated as evidence for any one component here.

- Paper: <https://www.nature.com/articles/s42256-025-01087-7>
- Official repository: <https://github.com/LongLee220/KA-GNN>
- Inspected repository commit: `6774c41a03a3117d67c6e60922420c0feb2b3151`

## Selected causal transplant

Use the parameter-efficient one-slot Neural-Atom/EdgeState skeleton only as a
matched control. In the candidate, replace the normalized
`Linear -> SiLU -> Dropout -> Linear` proposal in each of nine 64-dimensional
EdgeState updates with
`LayerNorm -> single-harmonic Fourier-KAN -> Dropout`. Keep source/target node
projections, edge residual, output normalization, RWSE16, local graph operator,
one-slot global exchange, pooling, head, data, and training contract unchanged.

A fresh full-GPS EdgeState arm remains an absolute anchor. The candidate must
beat both it and the one-slot control; merely stacking two sub-threshold changes
cannot nominate a transfer.

This is not the closed torsion Fourier encoding: no coordinate, distance,
angle, torsion, ETKDG, proximity edge, or new cache is introduced. It is route
2/3 after the cardinality decision. A scientific loss closes the exact
single-harmonic EdgeState-function replacement before the final route audit.
