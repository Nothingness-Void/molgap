# PCQM Fourier-Edge transfer decision — 2026-09-10

## Question

Did the QM9-positive single-harmonic Fourier EdgeState replacement transfer to
PCQM-100K after separating it from the one-slot latent-global skeleton?

## Acceptance

Kaggle2 kernel `kaseichou/molgap-pcqm-fourier-edge-s42`, version 1, completed
all three 40-epoch arms. No-model acceptance passed with source commit
`47f99cf9da7fee306f5165175b4020c6c4aa9fb3`, parent graph-cache aggregate
`eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`,
seed 42, FP32, and physical batch 128. Geometry fields were removed before
batching. All artifact hashes and remote preflight invariants matched; no
shadow, official PCQM validation, or test-dev role was read.

## Result

| Arm | Validation Gap MAE | Parameters | Mean epoch |
|---|---:|---:|---:|
| Full EdgeState GPS9 | 0.1360574961 eV | 4,771,073 | 75.233 s |
| One-slot latent control | 0.1265957952 eV | 3,658,817 | 67.328 s |
| Fourier-Edge one-slot | 0.1269524097 eV | 3,658,241 | 65.588 s |

Fourier-Edge improved on full GPS by `0.0091050863 eV` but regressed against
its architecture-matched K1 control by `0.0003566146 eV`; it therefore failed
the causal `0.001 eV` gate and did not authorize shadow access.

The valid K1-versus-full paired difference was `-0.009461689 eV`; a fixed-seed
10,000-resample paired bootstrap interval was
`[-0.01151975, -0.00748735] eV`. K1 improved 54.32% of rows and improved all
four target-value quartiles. Fourier minus K1 had interval
`[-0.00122520, +0.00190748] eV`, improved 49.70% of rows, and showed no
resolvable global advantage.

## Attribution

The large transferable effect is global-allocation simplification: replacing
nine dense atom-attention blocks with three one-slot exchanges reduced both
capacity and compute while improving validation. This agrees with the earlier
three-seed GraphState result and strengthens the conclusion that repeated
dense global attention is harmful on this bounded PCQM task.

The Fourier arm ended with lower normalized training error than K1
(`0.085685` versus `0.088912`) but worse validation. Both selected epoch 38,
the Fourier arm was slightly faster, memory was ample, and initialization and
sample exposure matched. Resource pressure or incomplete convergence therefore
does not explain the miss; the single-harmonic proposal increased fit without
transferable regularization. Its QM9 gain was dataset-specific.

## Decision

The exact Fourier EdgeState replacement is scientifically rejected for PCQM
and closed without harmonic, width, placement, seed, optimizer, or schedule
variants. It receives no shadow audit or scale-up.

The predeclared K1 causal control produced a large, internally consistent but
exploratory PCQM signal. Repeating the same seed and selection role would add
no information. Freeze K1 unchanged as the sole candidate for one independent,
label-sealed, official-train-derived shadow audit. Because K1 was a control
rather than the originally nominated Fourier arm, the shadow result must be
reported as confirmatory evidence for K1, not as a successful Fourier transfer.
No desktop/full-scale handoff follows unless that untouched shadow agrees.

The final route-3/3 architecture-discovery attempt remains unused. It cannot be
submitted while K1 has an unresolved independent-audit path.

Machine provenance: `results/gpu_seed42_launch.json`.
