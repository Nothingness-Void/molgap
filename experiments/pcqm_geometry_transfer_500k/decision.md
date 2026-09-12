# Geometry Transfer 500K Decision

Decision date: 2026-09-12

Both geometry-enhanced models completed the frozen 500K/50K development
protocol. GPTrans-T reached `0.1027374789 eV` and Neural-Atom K1 reached
`0.1043544337 eV`. Their aligned predictions are complementary: the equal
blend reached `0.0995931532 eV`, and the five-fold out-of-fold convex blend
reached `0.0995458298 eV` with fold weights between `0.550` and `0.567` on the
GPTrans-T branch.

The `0.0808970350 eV` per-row Oracle result is a headroom bound only. It is not
a deployable result and must not be reported as model performance. The accepted
deployable evidence is the out-of-fold convex blend. Its absolute improvement
over the stronger component is `0.0031916491 eV`, which clears the V4 material
gain threshold of `0.003 eV` on this fixed development role.

The initial fusion failures were platform-only: the SCNet CPU partition lacked
OpenMPI and HSA libraries required by the DTK PyTorch build. The final fusion
ran under a short DCU allocation without changing data, predictions, or fusion
logic. No official validation, test-dev, or test-challenge role was read.
