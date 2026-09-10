# Fourier EdgeState decision — 2026-09-10

## Question

Did replacing only the persistent real-bond proposal MLP with a
single-harmonic Fourier-KAN materially improve the efficient one-slot global
skeleton on the frozen QM9 direct-Gap screen?

## Acceptance

Kaggle2 kernel `kaseichou/molgap-qm9-fourier-edge-s42`, version 1, completed
all three 40-epoch arms. No-model acceptance passed with source commit
`50fdaf00abee6a24a7285908dd830da243de33e9`, cache aggregate
`80a2d256ccbbde8de5862e7ad4fd61ea7c1255a19217e590033a94cabe1c6340`,
split `62f1cdefdaec6877`, seed 42, FP32, and physical batch 128. All artifact
hashes and remote preflight invariants matched; no QM9 test or official PCQM
role was read.

## Result

| Arm | Validation Gap MAE | Parameters | Mean epoch |
|---|---:|---:|---:|
| Full EdgeState GPS9 | 0.1280843765 eV | 4,771,073 | 18.723 s |
| One-slot latent control | 0.1284803897 eV | 3,658,817 | 15.722 s |
| Fourier-Edge one-slot | 0.1241975799 eV | 3,658,241 | 15.815 s |

The Fourier arm improved on full GPS by `0.0038867965 eV`, exceeding the
frozen `0.003 eV` gate, and on its architecture-matched one-slot control by
`0.0042828098 eV`, exceeding the `0.001 eV` causal gate. Its epoch-time ratio
to the one-slot control was `1.00594`, below the `1.25` ceiling, with more than
15% memory reserve.

## Attribution

The matched K1 comparison isolates the gain to the EdgeState proposal function
rather than latent-global simplification, parameter count, or training time.
All arms used identical data roles, initialization policy, exposure, optimizer,
and 40-epoch schedule. The candidate finished near its best epoch and passed
the compute gates, so incomplete convergence and resource failure do not
explain the result.

This is positive evidence for a bounded nonlinear function-family change in
persistent bond dynamics. It is not evidence for KA-GNN proximity edges,
additional harmonics, wider KANs, or a PCQM improvement.

## Decision

The candidate passes Track C and is nominated for exactly one paired
PCQM-100K transfer under the existing official-train-derived 100K/10K internal
validation contract. The transfer must preserve the architecture exactly and
train fresh matched controls in the same task at physical batch 128. No shadow
read, extra seed, official validation/test-dev access, desktop/full-scale
handoff, SCNet work, or IMS work is authorized by this result.

This successful Track C result does not consume the final route-3/3 discovery
attempt. If the PCQM transfer fails, its cause must be recorded before any
materially different route is considered.
