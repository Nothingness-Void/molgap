# Sparse Torsion EdgeState GPS9 Seed-42 GPU Decision

Decision date: 2026-09-01

## Question

Does one sparse persistent torsion state improve the accepted distance-plus-
angle Sparse Triangle EdgeState GPS9 on the frozen official-train-derived
PCQM 100K/10K roles?

## Accepted comparison

Kaggle1 kernel `nothingnessvoid/molgap-pcqm-sparse-torsion-s42`, version 5,
completed the version-3 atomic checkpoint after two resume-only infrastructure
steps. Version 4 verified and hydrated all eight hash-pinned artifacts but
stopped before training because CUDA map-location changed the generator RNG
state's device. Version 5 restored only the checked RNG tensors to CPU,
skipped the completed comparator, ran candidate epoch 39, and assembled the
final artifacts.

Independent no-inference acceptance passed with `resume_verified=true`. Both
traces contain exactly epochs 0 through 39, and the first and only new training
epoch in version 5 was candidate epoch 39.

| Model | Parameters | Best epoch | Validation Gap MAE (eV) | Mean throughput (graphs/s) |
|---|---:|---:|---:|---:|
| Distance + angle comparator | 4,891,057 | 35 | 0.1353926808 | 403.2454 |
| + sparse torsion state | 4,902,081 | 36 | 0.1363666952 | 295.9366 |

The torsion candidate was worse by `+0.0009740144 eV` (`+0.7194%`), while
adding 11,024 parameters (`+0.2254%`) and reducing mean throughput by `26.61%`.
Its final epoch reached `0.1371754408 eV`, so epoch 39 did not replace the
epoch-36 best checkpoint.

## Interpretation

The fixed single-conformer torsion state did not add useful predictive signal
beyond the accepted distance-and-angle bottom fusion under this contract. Its
cost came mainly from enumerating and updating sparse non-backtracking
four-atom paths, not from parameter count. The result is consistent with the
torsion feature being partly redundant with the existing local geometry and
partly sensitive to ETKDG conformer noise. It closes this specific persistent
torsion mechanism; it does not claim that every possible torsional or learned
geometry representation is ineffective.

## Disposition

The strict seed-42 advancement gate failed. Sparse torsion receives no seeds
43/44, full-data training, official-role evaluation, desktop submission, or
molecular-research-server work. The accepted distance-plus-angle Sparse
Triangle EdgeState GPS9 remains the 100K comparator. Any later architecture
question must use a separately frozen protocol and cannot tune from official
validation or test-dev.

## Evidence

- Accepted metrics and hashes: [`gpu_v5_summary.json`](gpu_v5_summary.json)
- No-inference acceptance: [`gpu_v5_acceptance.json`](gpu_v5_acceptance.json)
- Version-3 checkpoint contract: [`gpu_v3_timeout_resume.md`](gpu_v3_timeout_resume.md)
- Version-4 infrastructure evidence: [`gpu_v4_resume_failure.json`](gpu_v4_resume_failure.json)
- Version-5 launch identity: [`gpu_v5_resume_launch_manifest.json`](gpu_v5_resume_launch_manifest.json)
- Accepted torsion cache: [`cache_decision.md`](cache_decision.md)
