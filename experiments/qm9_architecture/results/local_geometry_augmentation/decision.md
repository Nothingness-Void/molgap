# Local SchNet angle and dihedral augmentation

Date: 2026-07-28

## Question

Can the transferable part of 3DMSE, explicit bonded angle and dihedral
information, improve the deployment-like QM9 ETKDG SchNet branch without
exceeding the project's 2x SchNet compute limit?

## Protocol

- Fixed QM9 split: 100,000 train / 10,000 validation / 10,000 test, split
  seed 42.
- Geometry: the existing ETKDGv3 + MMFF200 seed-42 cache.
- Encoder seed: 42.
- Budget: 30 epochs with the existing SchNet optimizer and schedule.
- Base encoder: lightweight SchNet 176/160/6.
- Angle features: per-atom Gaussian basis over bonded `cos(theta)`.
- Dihedral features: per-atom reflection-invariant `cos(n phi)`, `n=1..4`.
- Covalent neighbors are inferred once from atomic numbers and ETKDG bond
  lengths. Features are cached and injected into the initial atom embedding.
- Acceptance gate established before the run: at least 0.003 eV average-MAE
  improvement and less than 2x SchNet epoch time.

The first one-epoch timing probe was not used as evidence because resuming it
would retain a one-epoch cosine schedule. The recorded 30-epoch runs started
from fresh model and optimizer states.

## Result

All test comparisons cover the same 9,306 ETKDG-successful molecules.

| Encoder | Parameters | Seconds/epoch | Cost ratio | HOMO | LUMO | Gap | Average |
|---|---:|---:|---:|---:|---:|---:|---:|
| SchNet | 809,668 | 17.73 | 1.00x | 0.08054 | 0.09136 | 0.11130 | 0.09440 |
| SchNet + angle | 842,580 | 20.51 | 1.16x | 0.08022 | 0.09100 | 0.11030 | 0.09384 |
| SchNet + angle + dihedral | 843,460 | 20.39 | 1.15x | 0.08065 | 0.09003 | 0.10964 | 0.09344 |

Paired average-MAE deltas versus SchNet:

| Variant | Delta | Paired bootstrap 95% CI | Molecule win rate |
|---|---:|---:|---:|
| Angle | -0.00056 eV | [-0.00165, +0.00058] | 50.27% |
| Angle + dihedral | -0.00096 eV | [-0.00206, +0.00021] | 50.42% |

## Decision

The lightweight augmentation passes the compute gate but fails the accuracy
gate. Its best average gain is the same size as the encoder-seed noise already
measured by this experiment family, and both paired confidence intervals cross
zero. Do not add more seeds and do not transfer this variant to PubChemQC 100K.

This result rejects only cheap scalar injection into SchNet. It does not
validate or falsify the paper's full equivariant message-passing architecture.
The larger local evidence remains unchanged: improving ETKDG geometry quality
has substantially more leverage than adding another geometry architecture.

Machine-readable values: `summary.json`.
